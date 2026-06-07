import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Sequence, List, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Make the shared helpers importable regardless of the current working directory
# (scripts are launched as `python src/exclusive_unlearning/run_exclusive_unlearning.py`).
import sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from eu_common.chat_format import build_chat_messages, tokenizer_supports_system_role


def is_rank_0():
    return not dist.is_initialized() or dist.get_rank() == 0


IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}



def prepare_tokenizer_and_model_for_padding(tokenizer, model) -> int:
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None. This script requires EOS.")

    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config") and model.config is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return int(start_id)


def prepend_start_1d(ids_1d: torch.Tensor, start_id: int) -> torch.Tensor:
    start = torch.tensor([start_id], dtype=ids_1d.dtype, device=ids_1d.device)
    return torch.cat([start, ids_1d], dim=0)


def prepend_start_2d(input_ids: torch.Tensor, attention_mask: torch.Tensor, start_id: int):
    B = input_ids.size(0)
    start_col = torch.full((B, 1), start_id, dtype=input_ids.dtype, device=input_ids.device)
    start_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([start_col, input_ids], dim=1), torch.cat([start_mask, attention_mask], dim=1)


# ========= chat_template utilities =========

def apply_chat_template_text(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("This tokenizer has no apply_chat_template(). You requested chat_template mode.")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def encode_text(tokenizer, text: str, max_length: int) -> torch.Tensor:
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return enc["input_ids"][0]


def build_chat_ids_labels_assistant_only(
    tokenizer,
    *,
    user_text: str,
    assistant_text: str,
    model_max_length: int,
    loss_on_full_sequence: bool,
    system_text: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    supports_system = tokenizer_supports_system_role(tokenizer)
    msgs_full = build_chat_messages(
        user_text=user_text,
        assistant_text=assistant_text,
        system_text=system_text,
        supports_system=supports_system,
    )
    msgs_prefix = build_chat_messages(
        user_text=user_text,
        assistant_text=None,
        system_text=system_text,
        supports_system=supports_system,
    )

    full_text = apply_chat_template_text(tokenizer, msgs_full, add_generation_prompt=False)
    prefix_text = apply_chat_template_text(tokenizer, msgs_prefix, add_generation_prompt=True)

    full_ids = encode_text(tokenizer, full_text, max_length=model_max_length)
    prefix_ids = encode_text(tokenizer, prefix_text, max_length=model_max_length)

    prefix_len = int(prefix_ids.size(0))
    prefix_len = min(prefix_len, int(full_ids.size(0)))

    labels = full_ids.clone()
    if not loss_on_full_sequence:
        labels[:prefix_len] = IGNORE_INDEX

    attn = torch.ones_like(full_ids, dtype=torch.long)
    return full_ids, labels, attn, prefix_len


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model path or ID"})
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    torch_dtype: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)
    low_cpu_mem_usage: bool = field(default=True)
    attn_implementation: Optional[str] = field(default=None)


@dataclass
class ForgetArguments:
    num_train_steps: int = field(default=10000)
    seq_len: int = field(default=256)
    eval_text_file: Optional[str] = field(default=None)

    temperature: float = field(default=1.0)
    top_k: int = field(default=0)
    train_generate_batch_size: int = field(default=4)

    output_comparison_file: Optional[str] = field(default="comparison_outputs.jsonl")

    forget_lambda: float = field(
        default=0.4,
        metadata={"help": "Weight for forget loss. Total = λ*forget + (1-λ)*retain. Must be in [0,1]."},
    )

    retain_train_data_file: Optional[str] = field(default=None)
    retain_eval_data_file: Optional[str] = field(default=None)
    retain_eval_max_samples: int = field(default=200)
    forget_sample_file: Optional[str] = field(default=None)

    wandb_project: Optional[str] = field(default="exclusive_loss_instruct_chattemplate")

    # retain loss
    retain_loss_on_full_sequence: bool = field(
        default=False,
        metadata={"help": "If True, apply CE loss on prompt+response (full sequence). If False, response-only."},
    )

    # debug prints
    debug_samples_retain: int = field(default=3)
    debug_batches_collator: int = field(default=1)
    debug_samples_forget: int = field(default=3)


class AlpaCareIFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        start_id: int,
        max_samples: Optional[int] = None,
        seed: int = 42,
        model_max_length: int = 512,
        loss_on_full_sequence: bool = False
    ):
        super().__init__()
        self.input_ids = []
        self.labels = []
        self.attn_masks = []
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.loss_on_full_sequence = loss_on_full_sequence
        self.start_id = int(start_id)

        records = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                records = obj
            else:
                raise ValueError("IFT dataset expects list JSON or jsonl")

        if max_samples is not None and len(records) > max_samples:
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)

        prompt_input = PROMPT_DICT["prompt_input"]
        prompt_no_input = PROMPT_DICT["prompt_no_input"]

        for ex in records:
            ins = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("response") or ex.get("output") or "").strip()

            src = (prompt_input if inp else prompt_no_input).format_map({"instruction": ins, "input": inp})
            tgt = f"{out}{tokenizer.eos_token}"

            tok_src = tokenizer(src, return_tensors="pt", padding=False, truncation=True,
                                max_length=self.model_max_length, add_special_tokens=False)
            src_ids = tok_src.input_ids[0]
            src_len = src_ids.size(0)

            tok_tgt = tokenizer(tgt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
            tgt_ids = tok_tgt.input_ids[0]

            ids = torch.cat([src_ids, tgt_ids], dim=0)
            ids = prepend_start_1d(ids, self.start_id)
            src_len = src_len + 1

            if ids.size(0) > self.model_max_length:
                ids = ids[: self.model_max_length]
                src_len = min(src_len, ids.size(0))

            lab = ids.clone()
            if (not loss_on_full_sequence) and src_len > 0:
                lab[:src_len] = IGNORE_INDEX

            attn = torch.ones_like(ids, dtype=torch.long)

            self.input_ids.append(ids)
            self.labels.append(lab)
            self.attn_masks.append(attn)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i], "attention_mask": self.attn_masks[i]}


class DataCollatorForIFTSupervised(object):
    def __init__(self, tokenizer, debug=False, debug_batches=1):
        self.tokenizer = tokenizer
        self.debug = debug
        self.debug_batches = debug_batches
        self._debug_count = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]

        batch_size = len(input_ids_list)
        lengths = [ids.size(0) for ids in input_ids_list]
        max_len = max(lengths)

        pad_id = self.tokenizer.pad_token_id
        input_ids = input_ids_list[0].new_full((batch_size, max_len), pad_id)
        labels = labels_list[0].new_full((batch_size, max_len), IGNORE_INDEX)
        attention_mask = input_ids_list[0].new_zeros((batch_size, max_len))

        for i, (ids, lab, L) in enumerate(zip(input_ids_list, labels_list, lengths)):
            input_ids[i, :L] = ids
            labels[i, :L] = lab
            attention_mask[i, :L] = 1

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class InstructChatRetainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_samples: Optional[int],
        seed: int,
        model_max_length: int,
        loss_on_full_sequence: bool,
        debug_samples: int = 3,
    ):
        super().__init__()
        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.attn_masks: List[torch.Tensor] = []
        self.prefix_lens: List[int] = []
        self.tokenizer = tokenizer
        self.model_max_length = int(model_max_length)
        self.loss_on_full_sequence = bool(loss_on_full_sequence)

        records: List[Dict[str, Any]] = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        records.append(json.loads(s))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, list):
                raise ValueError("retain dataset expects list JSON or jsonl")
            records = obj

        if max_samples is not None and len(records) > max_samples:
            rng = random.Random(seed)
            records = rng.sample(records, max_samples)

        for idx, ex in enumerate(records):
            ins = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("response") or ex.get("output") or "").strip()

            user_text = f"{ins}\n{inp}".strip() if inp else ins
            assistant_text = out

            ids, lab, attn, prefix_len = build_chat_ids_labels_assistant_only(
                tokenizer=self.tokenizer,
                user_text=user_text,
                assistant_text=assistant_text,
                model_max_length=self.model_max_length,
                loss_on_full_sequence=self.loss_on_full_sequence,
                system_text="",
            )

            self.input_ids.append(ids)
            self.labels.append(lab)
            self.attn_masks.append(attn)
            self.prefix_lens.append(prefix_len)

            if is_rank_0() and idx < int(debug_samples):
                toks = self.tokenizer.convert_ids_to_tokens(ids.tolist())
                loss_mask = (lab != IGNORE_INDEX).tolist()
                flags = ["L" if lm else "." for lm in loss_mask]

                print(f"\n===== [Retain Dataset chat example #{idx}] =====")
                print("system: ''")
                print("user_text (head):", user_text[:120], ("..." if len(user_text) > 120 else ""))
                print("assistant_text (head):", assistant_text[:120], ("..." if len(assistant_text) > 120 else ""))
                print(f"prefix_len={prefix_len}, total_len={ids.size(0)}")
                print("tokens :", " ".join(toks))
                print("flags  :", " ".join(flags))
                print("legend : L=loss flows (assistant), .=no loss (prompt/system/user)")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attn_masks[i],
            "prefix_len": torch.tensor(self.prefix_lens[i], dtype=torch.long),
        }



class DataCollatorForSupervisedChat(object):
    def __init__(self, tokenizer, debug=False, debug_batches=1):
        self.tokenizer = tokenizer
        self.debug = bool(debug)
        self.debug_batches = int(debug_batches)
        self._debug_count = 0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [inst["input_ids"] for inst in instances]
        labels_list = [inst["labels"] for inst in instances]

        batch_size = len(input_ids_list)
        lengths = [ids.size(0) for ids in input_ids_list]
        max_len = max(lengths)

        pad_id = self.tokenizer.pad_token_id
        input_ids = input_ids_list[0].new_full((batch_size, max_len), pad_id)
        labels = labels_list[0].new_full((batch_size, max_len), IGNORE_INDEX)
        attention_mask = input_ids_list[0].new_zeros((batch_size, max_len))

        for i, (ids, lab, L) in enumerate(zip(input_ids_list, labels_list, lengths)):
            input_ids[i, :L] = ids
            labels[i, :L] = lab
            attention_mask[i, :L] = 1

        if self.debug and is_rank_0() and self._debug_count < self.debug_batches:
            self._debug_count += 1
            print(f"\n===== [Collator debug #{self._debug_count}] =====")
            print(f"batch_size={batch_size}, max_len={max_len}")

            show_n = min(2, batch_size)
            for b in range(show_n):
                ids_row = input_ids[b].tolist()
                lab_row = labels[b].tolist()
                att_row = attention_mask[b].tolist()

                tokens = self.tokenizer.convert_ids_to_tokens(ids_row)
                flags = []
                for lab_v, att_v in zip(lab_row, att_row):
                    if not att_v:
                        flag = "P"
                    elif lab_v != IGNORE_INDEX:
                        flag = "L"
                    else:
                        flag = "."
                    flags.append(flag)

                print(f"\n--- sample {b} ---")
                print("tokens :", " ".join(tokens))
                print("flags  :", " ".join(flags))
                print("legend : P=padding, L=loss flows (assistant), .=no loss")

                first_loss_idx = None
                for idx2, (lab_v, att_v) in enumerate(zip(lab_row, att_row)):
                    if att_v and lab_v != IGNORE_INDEX:
                        first_loss_idx = idx2
                        break

                if first_loss_idx is not None:
                    first_token_id = ids_row[first_loss_idx]
                    first_token_decoded = self.tokenizer.decode([first_token_id])
                    print(
                        f"first loss token idx={first_loss_idx}, "
                        f"id={first_token_id}, "
                        f"decoded={repr(first_token_decoded)}"
                    )
                else:
                    print("first loss token: (none)")

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}



def evaluate_unconditional(model, tokenizer, start_id: int, entries, prefix="", log_to_wandb=True, output_path=None, eval_max_length: int = 512):
    model.eval()
    loss_list: List[float] = []
    stats_records: List[Dict[str, Any]] = []

    vocab_size = model.config.vocab_size
    max_entropy = math.log(vocab_size)

    for entry in entries:
        user_text = (entry.get("user_text") or "").strip()
        assistant_text = (entry.get("assistant_text") or "").strip()

        if user_text == "" and assistant_text == "":
            text = (entry.get("text") or "").strip()
            if text:
                user_text = text
                assistant_text = ""

        ids, labels, attn, prefix_len = build_chat_ids_labels_assistant_only(
            tokenizer=tokenizer,
            user_text=user_text,
            assistant_text=assistant_text,
            model_max_length=eval_max_length,
            loss_on_full_sequence=False,
            system_text="",
        )

        input_ids = ids.unsqueeze(0).to(model.device)          # (1,T)
        attention_mask = attn.unsqueeze(0).to(model.device)    # (1,T)
        labels_ = labels.unsqueeze(0).to(model.device)         # (1,T)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_,
            )
            loss = float(outputs.loss.item())
            logits = outputs.logits  # (1,T,V)

        target_mask_tokens = (labels_ != IGNORE_INDEX) & (attention_mask.bool())  # (1,T)
        pred_mask = target_mask_tokens[:, 1:]  # (1,T-1)
        logits_pred = logits[:, :-1, :][pred_mask]  # (N,V)
        token_count = int(pred_mask.sum().item())

        if token_count > 0:
            probs = F.softmax(logits_pred, dim=-1)
            entropy = -torch.sum(probs * probs.log(), dim=-1)           # (N,)
            max_prob = torch.max(probs, dim=-1).values                  # (N,)
            variance = torch.var(probs, dim=-1)                         # (N,)
            entropy_mean = float(entropy.mean().item())
            max_prob_mean = float(max_prob.mean().item())
            var_mean = float(variance.mean().item())
        else:
            entropy_mean = float("nan")
            max_prob_mean = float("nan")
            var_mean = float("nan")

        stats_records.append(
            {
                "user_text": user_text,
                "assistant_text": assistant_text,
                "loss": loss,
                "ppl": math.exp(loss) if math.isfinite(loss) else float("nan"),
                "assistant_token_count": token_count,
                "entropy": entropy_mean,
                "max_entropy": max_entropy,
                "max_prob": max_prob_mean,
                "prob_variance": var_mean,
                "prefix_len": int(prefix_len),
                "total_len": int(ids.size(0)),
            }
        )
        loss_list.append(loss)

    loss_mean = sum(loss_list) / len(loss_list) if loss_list else float("nan")
    ppl_from_loss_mean = math.exp(loss_mean) if math.isfinite(loss_mean) else float("nan")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in stats_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if is_rank_0() and log_to_wandb and loss_list:
        wandb.log({f"{prefix}loss_mean": loss_mean, f"{prefix}ppl_mean": ppl_from_loss_mean})

    model.train()


def evaluate_retained_ift(model, tokenizer, dataset, prefix="eval/retain/"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    data_collator = DataCollatorForSupervisedChat(tokenizer, debug=False)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

    for batch in tqdm(dataloader, desc=f"Evaluating {prefix}"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += float(outputs.loss.item())
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    if is_rank_0():
        wandb.log({f"{prefix}loss": avg_loss})
        print(f"{prefix}Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss


def loss_function_kl_p_theta_to_uniform(logits, attention_mask):
    logits = logits[:, :-1, :]             # (B, T-1, V)
    attention_mask = attention_mask[:, 1:] # (B, T-1)

    log_probs = F.log_softmax(logits, dim=-1)  # (B, T-1, V)
    probs = log_probs.exp()

    logV = math.log(logits.size(-1))
    per_token = (probs * log_probs).sum(dim=-1) + logV  # (B, T-1)

    mask = attention_mask.to(per_token.dtype)
    token_count = mask.sum(dim=1).clamp_min(1.0)

    per_seq = (per_token * mask).sum(dim=1) / token_count
    return per_seq.mean()


def loss_function_kl_p_theta_to_uniform_masked(logits, attention_mask, loss_mask_tokens):
    logits = logits[:, :-1, :]                        # (B, T-1, V)
    attn = attention_mask[:, 1:].to(logits.dtype)     # (B, T-1)
    lm = loss_mask_tokens[:, 1:].to(logits.dtype)     # (B, T-1)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    logV = math.log(logits.size(-1))

    per_token = (probs * log_probs).sum(dim=-1) + logV  # (B, T-1)

    mask = attn * lm
    token_count = mask.sum(dim=1).clamp_min(1.0)
    per_seq = (per_token * mask).sum(dim=1) / token_count
    return per_seq.mean()


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        tokenizer,
        seq_len: int,
        temperature: float,
        top_k: int,
        run_dir: str,
        train_generate_batch_size: int,
        forget_lambda: float = 0.5,
        forget_sample_file: Optional[str] = None,
        forget_debug_samples: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

        self.seq_len = int(seq_len)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.run_dir = run_dir
        self.train_generate_batch_size = int(train_generate_batch_size)

        if not (0.0 <= float(forget_lambda) <= 1.0):
            raise ValueError(f"forget_lambda must be in [0,1], got {forget_lambda}")
        self.forget_lambda = float(forget_lambda)

        self.forget_pairs: List[Tuple[str, str]] = []
        self.shuffle_rng = random.Random(42)
        self.sample_pointer = 0

        self._forget_debug_remaining = int(forget_debug_samples)
        self._forget_debug_batches_printed = 0

        if forget_sample_file is not None:
            with open(forget_sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    obj = json.loads(s)
                    ut = (obj.get("user_text") or "").strip()
                    at = (obj.get("assistant_text") or "").strip()
                    if ut == "" and at == "":
                        continue
                    self.forget_pairs.append((ut, at))
            self.shuffle_rng.shuffle(self.forget_pairs)

        if is_rank_0():
            print(f"[Forget] loaded pairs: {len(self.forget_pairs)} from {forget_sample_file}")

    def _debug_print_forget_batch(self, user_texts, assistant_texts, input_ids, labels, attention_mask, prefix_lens):
        if not (is_rank_0() and self._forget_debug_remaining > 0):
            return

        self._forget_debug_batches_printed += 1
        print(f"\n===== [Forget debug batch #{self._forget_debug_batches_printed}] =====")

        show_n = min(self._forget_debug_remaining, input_ids.size(0))
        for b in range(show_n):
            ids_row = input_ids[b].tolist()
            lab_row = labels[b].tolist()
            att_row = attention_mask[b].tolist()
            toks = self.tokenizer.convert_ids_to_tokens(ids_row)

            flags = []
            for lab_v, att_v in zip(lab_row, att_row):
                if not att_v:
                    flags.append("P")
                elif lab_v != IGNORE_INDEX:
                    flags.append("L")
                else:
                    flags.append(".")
            seq_len = int(sum(att_row))
            print(f"\n--- forget sample {b} ---")
            print("system: ''")
            print("user_text (head):", user_texts[b][:200], ("..." if len(user_texts[b]) > 200 else ""))
            print("assistant_text (head):", assistant_texts[b][:200], ("..." if len(assistant_texts[b]) > 200 else ""))
            print(f"prefix_len={int(prefix_lens[b])}, total_len={seq_len} (padded_len={len(att_row)})")
            print("tokens :", " ".join(toks))
            print("flags  :", " ".join(flags))
            print("legend : L=loss flows (assistant-only), .=no loss, P=padding")

        self._forget_debug_remaining -= show_n

    def _compute_forget_loss(self, model):
        if not self.forget_pairs:
            return torch.tensor(0.0, device=model.device)

        bsz = self.train_generate_batch_size

        pairs = self.forget_pairs[self.sample_pointer : self.sample_pointer + bsz]
        if len(pairs) < bsz:
            self.sample_pointer = 0
            self.shuffle_rng.shuffle(self.forget_pairs)
            pairs = self.forget_pairs[self.sample_pointer : self.sample_pointer + bsz]
        self.sample_pointer += bsz

        user_texts = [p[0] for p in pairs]
        assistant_texts = [p[1] for p in pairs]

        ids_list: List[torch.Tensor] = []
        prefix_lens: List[int] = []
        for ut, at in zip(user_texts, assistant_texts):
            ids, _, attn, prefix_len = build_chat_ids_labels_assistant_only(
                tokenizer=self.tokenizer,
                user_text=ut,
                assistant_text=at,
                model_max_length=self.seq_len,
                loss_on_full_sequence=False,
                system_text="",
            )
            ids_list.append(ids)
            prefix_lens.append(prefix_len)

        pad_id = self.tokenizer.pad_token_id
        lengths = [x.size(0) for x in ids_list]
        max_len = max(lengths)

        input_ids = ids_list[0].new_full((bsz, max_len), pad_id)
        attention_mask = ids_list[0].new_zeros((bsz, max_len))
        labels = ids_list[0].new_full((bsz, max_len), IGNORE_INDEX)

        for i, ids in enumerate(ids_list):
            L = int(ids.size(0))
            input_ids[i, :L] = ids
            attention_mask[i, :L] = 1

            pl = int(prefix_lens[i])
            if pl < L:
                labels[i, pl:L] = ids[pl:L]

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        prefix_lens_t = torch.tensor(prefix_lens, device=model.device, dtype=torch.long)

        self._debug_print_forget_batch(user_texts, assistant_texts, input_ids, labels, attention_mask, prefix_lens_t)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        B, T = input_ids.size()
        loss_mask_tokens = torch.zeros((B, T), device=model.device, dtype=torch.long)
        for i in range(B):
            pl = int(prefix_lens[i])
            seqL = int(attention_mask[i].sum().item())
            if pl < seqL:
                loss_mask_tokens[i, pl:seqL] = 1

        loss = loss_function_kl_p_theta_to_uniform_masked(outputs.logits, attention_mask, loss_mask_tokens)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        retain_inputs = self._prepare_inputs(inputs)
        outputs_retain = model(**retain_inputs)
        loss_retain = outputs_retain.loss

        loss_forget = self._compute_forget_loss(model)

        lam = self.forget_lambda
        total_loss = lam * loss_forget + (1.0 - lam) * loss_retain

        if (
            is_rank_0()
            and self.state.global_step > 0
            and self.state.global_step % max(1, self.args.logging_steps) == 0
        ):
            wandb.log(
                {
                    "train/loss": float(total_loss.item()),
                    "train/forget_loss": float(loss_forget.item()),
                    "train/retain_loss": float(loss_retain.item()),
                    "train/forget_lambda": float(lam),
                }
            )

        return (total_loss, outputs_retain) if return_outputs else total_loss


class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, start_id: int, uncond_entries=None, retain_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.start_id = int(start_id)
        self.uncond_entries = uncond_entries
        self.retain_dataset = retain_dataset

    def on_log(self, args, state, control, **kwargs):
        if self.uncond_entries is not None:
            evaluate_unconditional(
                self.model,
                self.tokenizer,
                self.start_id,
                self.uncond_entries,
                prefix="eval/uncond/",
                log_to_wandb=True,
            )
        if self.retain_dataset is not None:
            evaluate_retained_ift(
                self.model,
                self.tokenizer,
                self.retain_dataset,
                prefix="eval/retain/",
            )


# ========= main =========

def main():
    parser = HfArgumentParser((ModelArguments, ForgetArguments, TrainingArguments))
    model_args, forget_args, training_args = parser.parse_args_into_dataclasses()

    if not (0.0 <= float(forget_args.forget_lambda) <= 1.0):
        raise ValueError(f"--forget_lambda must be in [0,1], got {forget_args.forget_lambda}")

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(training_args.output_dir, f"run-{run_time}")
    training_args.output_dir = run_dir

    if is_rank_0():
        os.makedirs(run_dir, exist_ok=True)
        wandb.init(project=forget_args.wandb_project, name=training_args.run_name)

    logging.basicConfig(level=logging.INFO)

    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = "right"

    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None
    model_kwargs = dict(
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    start_id = prepare_tokenizer_and_model_for_padding(tokenizer, model)

    if is_rank_0():
        print("\n===== [Tokenizer token ids] =====")
        print("bos_token:", repr(tokenizer.bos_token), "id=", tokenizer.bos_token_id)
        print("eos_token:", repr(tokenizer.eos_token), "id=", tokenizer.eos_token_id)
        print("pad_token:", repr(tokenizer.pad_token), "id=", tokenizer.pad_token_id)
        print("padding_side:", tokenizer.padding_side)
        start_tok = tokenizer.convert_ids_to_tokens([start_id])[0]
        which = "BOS" if (tokenizer.bos_token_id is not None and start_id == tokenizer.bos_token_id) else "EOS"
        print(f"START token (compat): {which} (id={start_id}, token={repr(start_tok)})")
        print("NOTE: training/forget/eval strings are built via tokenizer.apply_chat_template (system is empty).")

    # retain datasets
    retain_train_dataset = None
    if forget_args.retain_train_data_file:
        retain_train_dataset = InstructChatRetainDataset(
            data_path=forget_args.retain_train_data_file,
            tokenizer=tokenizer,
            max_samples=None,
            seed=training_args.seed,
            model_max_length=getattr(training_args, "model_max_length", 512),
            loss_on_full_sequence=forget_args.retain_loss_on_full_sequence,
            debug_samples=forget_args.debug_samples_retain,
        )

    retain_eval_dataset = None
    if forget_args.retain_eval_data_file:
        retain_eval_dataset = InstructChatRetainDataset(
            data_path=forget_args.retain_eval_data_file,
            tokenizer=tokenizer,
            max_samples=forget_args.retain_eval_max_samples,
            seed=training_args.seed,
            model_max_length=getattr(training_args, "model_max_length", 512),
            loss_on_full_sequence=forget_args.retain_loss_on_full_sequence,
            debug_samples=0,
        )

    eval_entries = None
    if forget_args.eval_text_file:
        with open(forget_args.eval_text_file, "r", encoding="utf-8") as f:
            eval_entries = [json.loads(line) for line in f if line.strip()]

    training_args.max_steps = forget_args.num_train_steps
    training_args.report_to = ["wandb"]
    training_args.save_strategy = "no"

    if is_rank_0():
        with open(os.path.join(run_dir, "used_args.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_args": vars(model_args),
                    "forget_args": vars(forget_args),
                    "training_args": training_args.to_dict(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    data_collator = DataCollatorForSupervisedChat(
        tokenizer=tokenizer,
        debug=True,
        debug_batches=forget_args.debug_batches_collator,
    )

    trainer = UnlearningTrainer(
        tokenizer=tokenizer,
        seq_len=forget_args.seq_len,
        temperature=forget_args.temperature,
        top_k=forget_args.top_k,
        model=model,
        args=training_args,
        train_generate_batch_size=forget_args.train_generate_batch_size,
        train_dataset=retain_train_dataset,
        callbacks=[EvalCallback(model, tokenizer, start_id, eval_entries, retain_eval_dataset)],
        run_dir=run_dir,
        forget_lambda=forget_args.forget_lambda,
        data_collator=data_collator,
        forget_sample_file=forget_args.forget_sample_file,
        forget_debug_samples=forget_args.debug_samples_forget,
    )

    if is_rank_0():
        train_start_time = time.time()

    trainer.train()

    if is_rank_0():
        elapsed = time.time() - train_start_time
        wandb.log({"train/total_training_time_sec": elapsed})
        print(f"Training took {elapsed:.2f} seconds.")
        with open(os.path.join(run_dir, "training_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{elapsed:.2f} seconds\n")

        final_path = os.path.join(run_dir, "final_checkpoint")
        os.makedirs(final_path, exist_ok=True)
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        wandb.finish()


if __name__ == "__main__":
    main()