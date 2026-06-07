import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

# Make the shared helpers importable regardless of the current working directory.
import sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from eu_common.chat_format import build_chat_messages, tokenizer_supports_system_role


@dataclass
class SampleArguments:
    model_name_or_path: str = field(metadata={"help": "Model path or name"})
    output_file: str = field(default="sampled_texts.jsonl", metadata={"help": "Output path for sampled texts"})
    num_train_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=8)

    # ====== assistant length is FIXED ======
    assistant_len: int = field(
        default=256,
        metadata={"help": "FINAL assistant length saved after truncation (fixed length)."},
    )
    assistant_generate_len: int = field(
        default=-1,
        metadata={
            "help": "Max NEW assistant tokens for every sample (max only). "
                    "If -1, uses assistant_len."
        },
    )

    # ====== user length is VARIABLE ======
    user_prompt_mode: str = field(
        default="generate",
        metadata={"help": "User prompt mode: 'generate' or 'empty'."},
    )
    user_seq_len: int = field(
        default=256,
        metadata={"help": "Fallback final USER length if --user_length_choices is not set"},
    )
    user_length_choices: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated USER lengths. These are the FINAL user lengths saved after truncation."},
    )
    user_generate_len: int = field(
        default=-1,
        metadata={
            "help": "Max NEW USER tokens for every sample (max only). "
                    "Then truncate to each length in --user_length_choices. "
                    "If -1, uses max(user_length_choices)."
        },
    )

    # ====== system prompts (NO base prompt) ======
    # If prompt is None and file is None -> "" (empty system message)
    user_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt used ONLY for self-generating USER prompts. If None and no file, uses empty string."},
    )
    user_system_prompt_file: Optional[str] = field(
        default=None,
        metadata={"help": "If set, loads user_system_prompt from this file (UTF-8). Overrides --user_system_prompt."},
    )

    assistant_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt used ONLY for generating ASSISTANT responses. If None and no file, uses empty string."},
    )
    assistant_system_prompt_file: Optional[str] = field(
        default=None,
        metadata={"help": "If set, loads assistant_system_prompt from this file (UTF-8). Overrides --assistant_system_prompt."},
    )

    use_chat_template: bool = field(
        default=True,
        metadata={"help": "If True, uses tokenizer.apply_chat_template. Required for user_prompt_mode=generate."},
    )

    temperature: float = field(default=1.0)
    top_k: Optional[int] = field(default=100, metadata={"help": "Top-k sampling (None disables)"})
    seed: int = field(default=42)
    device: str = field(default="cuda", metadata={"help": "cuda or cpu"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to cache the model"})

    # ===== debug =====
    debug_dump_prefix: bool = field(
        default=False,
        metadata={"help": "If True, dump prefix_text and reconstructed full_text to verify boundary."},
    )


def _parse_int_choices(csv: Optional[str], fallback: int, arg_name: str) -> List[int]:
    if csv is None:
        choices = [int(fallback)]
    else:
        choices = [int(x.strip()) for x in csv.split(",") if x.strip()]
        if not choices:
            raise ValueError(f"{arg_name} is empty. Example: {arg_name} '32,64,128'")
    if any(L <= 0 for L in choices):
        raise ValueError(f"{arg_name} must be positive ints, got: {choices}")
    seen = set()
    uniq = []
    for L in choices:
        if L not in seen:
            seen.add(L)
            uniq.append(L)
    return uniq


def _allocate_counts_uniform(total_samples: int, lengths: List[int]) -> Dict[int, int]:
    k = len(lengths)
    base = total_samples // k
    rem = total_samples % k
    counts = {L: base for L in lengths}
    for i in range(rem):
        counts[lengths[i]] += 1
    return counts


def _build_batch_tasks(counts: Dict[int, int], batch_size: int, seed: int) -> List[Tuple[int, int]]:
    tasks: List[Tuple[int, int]] = []
    for L, n in counts.items():
        if n <= 0:
            continue
        full = n // batch_size
        rem = n % batch_size
        tasks.extend([(L, batch_size)] * full)
        if rem:
            tasks.append((L, rem))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks


def _load_prompt_from_file(path: Optional[str], default: Optional[str]) -> Optional[str]:
    if path is None:
        return default
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip("\n")


def _resolve_system_prompts(args: SampleArguments) -> Tuple[str, str]:
    """
    NO base system.
    Priority: file > string > empty string
    """
    user_system = _load_prompt_from_file(args.user_system_prompt_file, args.user_system_prompt)
    if user_system is None:
        user_system = ""

    assistant_system = _load_prompt_from_file(args.assistant_system_prompt_file, args.assistant_system_prompt)
    if assistant_system is None:
        assistant_system = ""

    return user_system, assistant_system


def _apply_chat_template_strict(
    tokenizer,
    messages,
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    continue_final_message: bool = False,
    return_tensors: Optional[str] = None,
):
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "This tokenizer has no apply_chat_template(). "
            "For user_prompt_mode=generate, chat_template is REQUIRED."
        )

    kwargs: Dict[str, Any] = dict(
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if continue_final_message:
        kwargs["continue_final_message"] = True

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as e:
        if continue_final_message and ("continue_final_message" in str(e) or "unexpected keyword argument" in str(e)):
            raise RuntimeError(
                "Tokenizer.apply_chat_template() does not support continue_final_message=True in this environment.\n"
                "Fix: upgrade transformers to a version that supports continue_final_message for this model."
            ) from e
        raise


def _generate_up_to_new_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    pad_token_id: int,
    eos_token_id: Optional[List[int]] = None,
) -> torch.Tensor:
    from transformers import LogitsProcessorList
    from transformers.generation import InfNanRemoveLogitsProcessor

    gen_kwargs = dict(
        do_sample=True,
        temperature=temperature,
        pad_token_id=pad_token_id,
        # bf16 + temperature=2.0 can produce inf/nan logits; clamp them before sampling.
        logits_processor=LogitsProcessorList([InfNanRemoveLogitsProcessor()]),
    )
    if top_k is not None:
        gen_kwargs["top_k"] = int(top_k)
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    import inspect
    try:
        sig = inspect.signature(model.generate)
        if "enable_thinking" in sig.parameters:
            gen_kwargs["enable_thinking"] = False
    except (TypeError, ValueError):
        pass

    try:
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
    except TypeError:
        prompt_len = input_ids.size(1)
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=prompt_len + max_new_tokens,
            **gen_kwargs,
        )


def _encode_prompts_safely(
    tokenizer,
    prompt_texts: List[str],
    model_device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    if input_ids.size(1) == 0:
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(prompt_texts), 1), bos_id, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

    return input_ids.to(model_device), attention_mask.to(model_device)


def main():
    parser = HfArgumentParser(SampleArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    user_system_prompt, assistant_system_prompt = _resolve_system_prompts(args)

    mode = str(args.user_prompt_mode).strip().lower()
    if mode not in ("generate", "empty"):
        raise ValueError("--user_prompt_mode must be either 'generate' or 'empty'")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.init_kwargs["padding_side"] = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Probe once whether this model's chat template accepts a system role
    # (Gemma-2 does not). Reused at both self-generation sites below.
    supports_system = tokenizer_supports_system_role(tokenizer)

    eos_ids: Optional[List[int]] = None
    if tokenizer.eos_token_id is not None:
        eos_ids = [int(tokenizer.eos_token_id)]

    if args.device == "cuda":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=args.cache_dir,
    )
    model.eval()
    model_device = next(model.parameters()).device

    batch_size = int(args.per_device_train_batch_size)
    if batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be > 0")

    total_samples = int(args.num_train_steps) * batch_size
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0 (num_train_steps * batch_size)")

    user_lengths = _parse_int_choices(args.user_length_choices, args.user_seq_len, "--user_length_choices")

    asst_final_L = int(args.assistant_len)
    if asst_final_L <= 0:
        raise ValueError("--assistant_len must be a positive int")

    if int(args.assistant_generate_len) == -1:
        asst_gen_L = asst_final_L
    else:
        asst_gen_L = int(args.assistant_generate_len)
        if asst_gen_L <= 0:
            raise ValueError("--assistant_generate_len must be a positive int or -1")

    if mode == "generate":
        if not args.use_chat_template:
            raise RuntimeError("user_prompt_mode=generate requires use_chat_template=True")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("user_prompt_mode=generate requires tokenizer.apply_chat_template()")

        if int(args.user_generate_len) == -1:
            user_max_new = max(user_lengths)
        else:
            user_max_new = int(args.user_generate_len)
            if user_max_new <= 0:
                raise ValueError("--user_generate_len must be a positive int or -1 when user_prompt_mode=generate")
    else:
        user_max_new = 0

    counts = _allocate_counts_uniform(total_samples, user_lengths)
    tasks = _build_batch_tasks(counts, batch_size=batch_size, seed=args.seed)

    print(f"Total samples               : {total_samples}")
    print(f"Batch size                  : {batch_size}")
    print(f"USER lengths (FINAL)        : {user_lengths} (after truncation)")
    print(f"USER max_new_tokens         : {user_max_new} (max only; 0 if empty)")
    print(f"ASSISTANT length (FINAL)    : {asst_final_L} (fixed)")
    print(f"ASSISTANT max_new_tokens    : {asst_gen_L} (max only)")
    print(f"User prompt mode            : {mode}")
    print(f"User system prompt          : {repr(user_system_prompt)}")
    print(f"Asst system prompt          : {repr(assistant_system_prompt)}")
    print(f"Use chat template           : {args.use_chat_template}")
    print("Allocated USER counts       : " + ", ".join([f"{L}:{counts[L]}" for L in user_lengths]))
    print(f"Num batch tasks             : {len(tasks)} (shuffled with seed={args.seed})")
    print(f"Output                      : {args.output_file}")
    if args.device == "cuda":
        print(f"Model dtype                 : {dtype} (bf16_supported={torch.cuda.is_bf16_supported()})")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # ===== prepare prefix for user self-generation (short-code style) =====
    if mode == "generate":
        user_msgs = build_chat_messages(
            user_text="",
            assistant_text=None,
            system_text=user_system_prompt,
            supports_system=supports_system,
        )
        user_prefix_ids = _apply_chat_template_strict(
            tokenizer,
            user_msgs,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,  # prefix ends right before user content
            return_tensors="pt",
        ).to(model_device)
        user_prefix_attn = torch.ones_like(user_prefix_ids, device=model_device)
        user_prefix_len = int(user_prefix_ids.shape[1])
        user_prefix_text = tokenizer.decode(user_prefix_ids[0], skip_special_tokens=False)
        user_prefix_mode = "chat_template_continue_final_user(strict)"
    else:
        user_prefix_ids = None
        user_prefix_attn = None
        user_prefix_len = 0
        user_prefix_text = ""
        user_prefix_mode = "unused(empty_mode)"

    written = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for t_idx, (user_L, bsz) in enumerate(tasks):

            # (1) USER self-generate or empty
            if mode == "generate":
                up_ids = user_prefix_ids.repeat(bsz, 1)
                up_attn = user_prefix_attn.repeat(bsz, 1)

                with torch.no_grad():
                    out_user = _generate_up_to_new_tokens(
                        model=model,
                        input_ids=up_ids,
                        attention_mask=up_attn,
                        max_new_tokens=int(user_max_new),
                        temperature=args.temperature,
                        top_k=args.top_k,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids,
                    )

                user_texts: List[str] = []
                user_raw_lens: List[int] = []
                user_used_ids: List[List[int]] = []

                for i in range(bsz):
                    seq = out_user[i]
                    gen_part = seq[user_prefix_len:]          # user content generated part
                    used = gen_part[: int(user_L)]            # FINAL user length varies here

                    user_raw_lens.append(int(gen_part.numel()))
                    user_used_ids.append(used.tolist())

                    ut = tokenizer.decode(used, skip_special_tokens=True)
                    user_texts.append(ut)
            else:
                user_texts = [""] * bsz
                user_raw_lens = [0] * bsz
                user_used_ids = [[] for _ in range(bsz)]

            # (2) system + user_text -> assistant prompt, then generate assistant
            if not (args.use_chat_template and hasattr(tokenizer, "apply_chat_template")):
                raise RuntimeError("This script expects chat_template for assistant generation.")

            prompt_texts = []
            for ut in user_texts:
                msgs = build_chat_messages(
                    user_text=ut,
                    assistant_text=None,
                    system_text=assistant_system_prompt,
                    supports_system=supports_system,
                )
                prompt_texts.append(
                    tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

            prompt_mode_used = "chat_template"
            input_ids, attention_mask = _encode_prompts_safely(tokenizer, prompt_texts, model_device)
            asst_input_len = int(input_ids.shape[1])

            with torch.no_grad():
                out_asst = _generate_up_to_new_tokens(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(asst_gen_L),
                    temperature=args.temperature,
                    top_k=args.top_k,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )

            # (3) save
            for i in range(bsz):
                gen_part = out_asst[i, asst_input_len:]       # assistant generated part
                asst_ids = gen_part[: int(asst_final_L)]      # FINAL assistant length fixed
                assistant_text = tokenizer.decode(asst_ids, skip_special_tokens=True)

                record: Dict[str, Any] = {
                    "user_system": user_system_prompt,
                    "assistant_system": assistant_system_prompt,

                    "user_text": user_texts[i],
                    "assistant_text": assistant_text,

                    "user_prompt_mode": mode,
                    "user_prefix_mode": user_prefix_mode,
                    "prompt_mode": prompt_mode_used,

                    "user_final_len": int(user_L),
                    "user_lengths_all": user_lengths,
                    "user_max_new_tokens": int(user_max_new),
                    "user_generated_len_raw": int(user_raw_lens[i]),

                    "assistant_len": int(asst_final_L),
                    "assistant_max_new_tokens": int(asst_gen_L),
                    "assistant_generated_len_raw": int(gen_part.numel()),
                }

                # optional boundary debug
                if args.debug_dump_prefix and mode == "generate":
                    record["user_prefix_len"] = int(user_prefix_len)
                    record["user_prefix_text"] = user_prefix_text
                    record["user_used_text_raw"] = tokenizer.decode(
                        torch.tensor(user_used_ids[i], dtype=torch.long),
                        skip_special_tokens=False,
                    )
                    full_ids = torch.cat(
                        [user_prefix_ids[0].to("cpu"), torch.tensor(user_used_ids[i], dtype=torch.long)],
                        dim=0,
                    )
                    record["user_full_text_raw"] = tokenizer.decode(full_ids, skip_special_tokens=False)

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            written += bsz
            if (t_idx % 50) == 0:
                print(f"[{t_idx:6d}/{len(tasks)}] written {written}/{total_samples} (last_user_L={user_L}, last_bsz={bsz})")

    print(f"Done. Wrote {written} samples to {args.output_file}")
    if written != total_samples:
        raise RuntimeError(f"Mismatch: wrote {written} but expected {total_samples}")


if __name__ == "__main__":
    main()
