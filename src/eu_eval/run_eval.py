"""EU evaluation orchestrator.

Runs the full Phase-1 evaluation pipeline on a trained checkpoint:
  1. Δ(x) detector threshold sweep (Table-8 reproduction)
  2. ASR on Harm-1/2 and JB-1/2 (with and without detector)
  3. Retention scoring on StemQA in-domain and OOD subjects
  4. Optionally: general-ability metrics (lm-eval-harness tasks)

Outputs a JSON report per run.

Usage
-----
python -m eu_eval.run_eval \\
    --checkpoint out/eu/lam0.4_alpacare_gemma-2-9b-it/final_checkpoint \\
    --safeunlearning_dir data/safeunlearning \\
    --stemqa_dir data/stemqa \\
    --indomain_subject biology \\
    --ood_subjects chemistry math physics \\
    --output_file out/eval/report_lam0.4.json \\
    --shieldlm_model thu-coai/ShieldLM-14B-qwen
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Make eu_common importable
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from eu_common.chat_format import build_chat_messages, tokenizer_supports_system_role


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EU evaluation orchestrator")
    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint directory")
    p.add_argument("--base_model", default=None,
                   help="Base model path for comparison (optional). Defaults to checkpoint.")
    p.add_argument("--safeunlearning_dir", default="data/safeunlearning",
                   help="Root of SafeUnlearning clone")
    p.add_argument("--stemqa_dir", default="data/stemqa",
                   help="Directory with StemQAMixture JSONL files")
    p.add_argument("--indomain_subject", default="biology",
                   help="StemQA in-domain subject")
    p.add_argument("--ood_subjects", nargs="+", default=["chemistry", "math", "physics"],
                   help="StemQA OOD subjects")
    p.add_argument("--output_file", default="out/eval/report.json")
    p.add_argument("--shieldlm_model", default="thu-coai/ShieldLM-14B-qwen")
    p.add_argument("--judge_model", default=None,
                   help="OpenAI model for ground_truth_similarity judge (e.g. gpt-4o)")
    p.add_argument("--detector_threshold", type=float, default=5.0)
    p.add_argument("--detector_gen_tokens", type=int, default=256)
    p.add_argument("--detector_sample_positions", type=int, default=32)
    p.add_argument("--harm_set", type=int, default=1, choices=[1, 2],
                   help="1=GPTFuzzer(100), 2=WildAttack(217)")
    p.add_argument("--max_jailbreak_prompts", type=int, default=None,
                   help="Cap number of JB prompts (for debugging)")
    p.add_argument("--max_retention_samples", type=int, default=200)
    p.add_argument("--skip_shieldlm", action="store_true")
    p.add_argument("--skip_retention", action="store_true")
    p.add_argument("--skip_detector_sweep", action="store_true")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model_and_tokenizer(path: str, device_map: str, torch_dtype_str: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, torch_dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def _generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)


def run_jailbreak_eval(
    model,
    tokenizer,
    jb_prompts: List[Dict],
    *,
    detector_threshold: float,
    gen_tokens: int,
    num_sample_positions: int,
    shieldlm_model_path: str,
    shieldlm_batch_size: int = 8,
    skip_shieldlm: bool = False,
) -> Dict:
    from eu_eval.detector import generate_with_detector

    responses_with_detector: List[str] = []
    responses_without_detector: List[str] = []
    deltas: List[float] = []
    forgotten_count = 0

    print(f"Running jailbreak eval on {len(jb_prompts)} prompts …")
    for i, item in enumerate(jb_prompts):
        prompt = item["prompt"]
        result = generate_with_detector(
            model, tokenizer, prompt,
            gen_tokens=gen_tokens,
            num_sample_positions=num_sample_positions,
            threshold=detector_threshold,
        )
        responses_with_detector.append(result["text"])
        deltas.append(result["delta"])
        if result["forgotten"]:
            forgotten_count += 1

        # Without detector: raw generation
        raw = _generate_response(model, tokenizer, prompt, max_new_tokens=gen_tokens)
        responses_without_detector.append(raw)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jb_prompts)}  forgotten_so_far={forgotten_count}")

    questions = [item["question"] for item in jb_prompts]

    result_dict = {
        "n_prompts": len(jb_prompts),
        "forgotten_count": forgotten_count,
        "forgotten_rate": forgotten_count / len(jb_prompts),
        "mean_delta": sum(deltas) / len(deltas),
    }

    if not skip_shieldlm:
        from eu_eval.shieldlm import ShieldLMJudge
        judge = ShieldLMJudge(shieldlm_model_path, batch_size=shieldlm_batch_size)
        asr_with = judge.compute_asr(questions, responses_with_detector)
        asr_without = judge.compute_asr(questions, responses_without_detector)
        result_dict["asr_with_detector"] = asr_with
        result_dict["asr_without_detector"] = asr_without

    return result_dict


def run_detector_threshold_sweep(
    model,
    tokenizer,
    jb_prompts: List[Dict],
    *,
    thresholds: Optional[List[float]] = None,
    gen_tokens: int = 256,
    num_sample_positions: int = 32,
) -> List[Dict]:
    """Table-8 reproduction: ASR vs threshold sweep."""
    from eu_eval.detector import compute_delta

    if thresholds is None:
        thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    print("Computing Δ(x) for all prompts …")
    deltas: List[float] = []
    for item in jb_prompts:
        device = next(model.parameters()).device
        enc = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        d = compute_delta(
            model, enc["input_ids"], enc["attention_mask"],
            gen_tokens=gen_tokens,
            num_sample_positions=num_sample_positions,
        )
        deltas.append(d)

    sweep_results = []
    for thr in thresholds:
        forgotten = [d < thr for d in deltas]
        sweep_results.append({
            "threshold": thr,
            "forgotten_count": sum(forgotten),
            "forgotten_rate": sum(forgotten) / len(forgotten),
        })
    return sweep_results


def run_retention_eval(
    model,
    tokenizer,
    records: List[Dict],
    judge_model: Optional[str],
    *,
    max_samples: int = 200,
    seed: int = 42,
    tag: str = "indomain",
) -> Dict:
    import random
    rng = random.Random(seed)
    if len(records) > max_samples:
        records = rng.sample(records, max_samples)

    questions = [r.get("question", r.get("instruction", "")) for r in records]
    golden_answers = [r.get("answer", r.get("output", r.get("response", ""))) for r in records]
    supports_system = tokenizer_supports_system_role(tokenizer)

    model_answers: List[str] = []
    for q in questions:
        msgs = build_chat_messages(q, system_text="", supports_system=supports_system)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        model_answers.append(_generate_response(model, tokenizer, prompt))

    result_dict: Dict = {"tag": tag, "n_samples": len(questions)}

    if judge_model is not None:
        from eu_eval.shieldlm import GroundTruthSimilarityJudge
        judge = GroundTruthSimilarityJudge(judge_model=judge_model)
        scored = judge.compute_retention_score(questions, golden_answers, model_answers)
        result_dict["similarity_score"] = scored
    else:
        result_dict["similarity_score"] = None
        print(f"  [retention/{tag}] No judge_model specified — skipping LLM judge.")

    return result_dict


def main() -> None:
    args = parse_args()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint:", args.checkpoint)
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, args.device_map, args.torch_dtype
    )

    report: Dict = {
        "checkpoint": args.checkpoint,
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── Jailbreak / ASR evaluation ────────────────────────────────────────────
    from eu_eval.data_loaders import load_jb_set
    print(f"Loading Harm-{args.harm_set} + JB templates …")
    jb_prompts = load_jb_set(args.safeunlearning_dir, args.harm_set)
    if args.max_jailbreak_prompts:
        import random
        rng = random.Random(args.seed)
        jb_prompts = rng.sample(jb_prompts, min(args.max_jailbreak_prompts, len(jb_prompts)))
    print(f"  {len(jb_prompts)} JB prompts loaded.")

    report["jailbreak_eval"] = run_jailbreak_eval(
        model, tokenizer, jb_prompts,
        detector_threshold=args.detector_threshold,
        gen_tokens=args.detector_gen_tokens,
        num_sample_positions=args.detector_sample_positions,
        shieldlm_model_path=args.shieldlm_model,
        skip_shieldlm=args.skip_shieldlm,
    )

    # ── Detector threshold sweep (Table 8) ────────────────────────────────────
    if not args.skip_detector_sweep:
        report["detector_sweep"] = run_detector_threshold_sweep(
            model, tokenizer, jb_prompts[:100],  # cap for speed
            gen_tokens=args.detector_gen_tokens,
            num_sample_positions=args.detector_sample_positions,
        )

    # ── Retention evaluation ──────────────────────────────────────────────────
    if not args.skip_retention:
        from eu_eval.data_loaders import load_stemqa
        print("Loading StemQAMixture …")
        splits = load_stemqa(
            data_dir=Path(args.stemqa_dir).parent,
            indomain_subject=args.indomain_subject,
            ood_subjects=args.ood_subjects,
            seed=args.seed,
        )
        report["retention"] = {
            "indomain": run_retention_eval(
                model, tokenizer, splits["test_indomain"], args.judge_model,
                max_samples=args.max_retention_samples, seed=args.seed, tag="indomain"
            ),
            "ood": run_retention_eval(
                model, tokenizer, splits["test_ood"], args.judge_model,
                max_samples=args.max_retention_samples, seed=args.seed, tag="ood"
            ),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
