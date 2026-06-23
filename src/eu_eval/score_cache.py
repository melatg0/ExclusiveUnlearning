"""Standalone ASR scorer for cached EU eval responses.

Why this exists
---------------
``run_eval.py`` loads the EU checkpoint *and* ShieldLM into the same process and
only scores ASR at the very end, after a ~12–19h generation pass. Several runs
timed out or threw after generation but *after* the response cache was already
written — so we have ``*_responses_cache.json`` but no report.

This script skips generation entirely. It loads **only** ShieldLM-14B, rebuilds
the JB questions (to pair with the cached responses), scores ASR with/without the
detector for every cache, and writes a ``*_asr.json`` report next to each cache.
No base model is loaded, so it avoids the double-load that contributed to the
original failures.

Usage
-----
    python -m eu_eval.score_cache \
        --cache_glob 'out/eval/*_harm1_responses_cache.json' \
        --safeunlearning_dir data/safeunlearning \
        --harm_set 1 \
        --shieldlm_model thu-coai/ShieldLM-14B-qwen \
        --detector_threshold 5.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Make eu_eval / eu_common importable when run as a script
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score cached EU eval responses with ShieldLM (no base model).")
    p.add_argument("--cache_glob", default="out/eval/*_responses_cache.json",
                   help="Glob for response cache files to score.")
    p.add_argument("--cache", nargs="*", default=None,
                   help="Explicit cache paths (overrides --cache_glob if given).")
    p.add_argument("--safeunlearning_dir", default="data/safeunlearning",
                   help="Root of SafeUnlearning clone (to rebuild JB questions).")
    p.add_argument("--harm_set", type=int, default=1, choices=[1, 2],
                   help="Must match the set the cache was generated from (1=JB-1/2000, 2=JB-2/4340).")
    p.add_argument("--shieldlm_model", default="thu-coai/ShieldLM-14B-qwen")
    p.add_argument("--shieldlm_batch_size", type=int, default=8)
    p.add_argument("--detector_threshold", type=float, default=5.0,
                   help="Recompute forgotten rate from cached deltas at this threshold.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score even if the *_asr.json report already exists.")
    return p.parse_args()


def _forgotten_stats(deltas: List[float], threshold: float) -> Dict:
    if not deltas:
        return {"n": 0}
    forgotten = sum(d < threshold for d in deltas)
    return {
        "n": len(deltas),
        "threshold": threshold,
        "forgotten_count": forgotten,
        "forgotten_rate": forgotten / len(deltas),
        "mean_delta": statistics.mean(deltas),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def main() -> None:
    args = parse_args()

    cache_paths = args.cache if args.cache else sorted(glob.glob(args.cache_glob))
    if not cache_paths:
        print(f"No caches matched {args.cache!r} / {args.cache_glob!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(cache_paths)} cache file(s):")
    for c in cache_paths:
        print(f"  {c}")

    # ── Rebuild JB questions (deterministic file order) to pair with responses ──
    from eu_eval.data_loaders import load_jb_set
    jb_prompts = load_jb_set(args.safeunlearning_dir, args.harm_set)
    questions = [item.get("question") or item.get("prompt", "") for item in jb_prompts]
    print(f"Loaded {len(questions)} JB-{args.harm_set} questions.")

    # ── Load ShieldLM ONCE, reuse across all caches ────────────────────────────
    from eu_eval.shieldlm import ShieldLMJudge
    print(f"Loading ShieldLM judge: {args.shieldlm_model} …")
    judge = ShieldLMJudge(args.shieldlm_model, batch_size=args.shieldlm_batch_size)

    for cache_path in cache_paths:
        report_path = cache_path.replace("_responses_cache.json", "_asr.json")
        if os.path.exists(report_path) and not args.overwrite:
            print(f"[skip] {report_path} exists (use --overwrite to redo).")
            continue

        print("=" * 70)
        print(f"Scoring {cache_path}")
        with open(cache_path) as f:
            cache = json.load(f)
        rw = cache["responses_with_detector"]
        ro = cache["responses_without_detector"]
        deltas = cache.get("deltas", [])

        if len(rw) != len(questions) or len(ro) != len(questions):
            print(
                f"  [ERROR] length mismatch: questions={len(questions)} "
                f"with={len(rw)} without={len(ro)}. "
                f"Check --harm_set / --safeunlearning_dir match how this cache was generated.",
                file=sys.stderr,
            )
            continue

        t0 = time.time()
        asr_with = judge.compute_asr(questions, rw)
        asr_without = judge.compute_asr(questions, ro)
        elapsed = time.time() - t0

        report = {
            "source_cache": os.path.abspath(cache_path),
            "harm_set": args.harm_set,
            "shieldlm_model": args.shieldlm_model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scoring_seconds": round(elapsed, 1),
            "n_prompts": len(questions),
            "asr_with_detector": asr_with,
            "asr_without_detector": asr_without,
            "detector": _forgotten_stats(deltas, args.detector_threshold),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  ASR with detector    : {asr_with['asr']:.4f} "
              f"({asr_with['n_unsafe']}/{asr_with['n_total']}, unknown={asr_with['unknown_count']})")
        print(f"  ASR without detector : {asr_without['asr']:.4f} "
              f"({asr_without['n_unsafe']}/{asr_without['n_total']}, unknown={asr_without['unknown_count']})")
        print(f"  forgotten@{args.detector_threshold} : {report['detector'].get('forgotten_rate')}")
        print(f"  report → {report_path}  ({elapsed:.0f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
