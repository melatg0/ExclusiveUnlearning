"""Data loaders for EU evaluation datasets.

Covers:
- SafeUnlearning (Harm-1/2, 20 jailbreak templates → JB-1/JB-2)
- StemQAMixture (in-domain + OOD retention splits)

SafeUnlearning structure (github.com/thu-coai/SafeUnlearning):
    data/
      harmful_questions/
        gptfuzzer_100.jsonl     (Harm-1: 100 harmful questions)
        wildattack_217.jsonl    (Harm-2: 217 harmful questions)
      jailbreak_templates/
        templates.jsonl         (20 jailbreak templates with {query} placeholder)

Each template in templates.jsonl has a "template" field with "{query}" as
the placeholder for the harmful question.  JB-1 = 100×20 = 2,000 prompts;
JB-2 = 217×20 = 4,340 prompts.

StemQAMixture (huggingface.co/datasets/4gate/StemQAMixture):
    Subjects: biology, chemistry, math, physics (+ possibly others).
    Format: {"question": ..., "answer": ..., "subject": ...}
    We hold out test sets and reserve disjoint slices for recovery-SFT.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ── SafeUnlearning loaders ────────────────────────────────────────────────────
#
# Actual repo structure (github.com/thu-coai/SafeUnlearning):
#   evaluation/input_prompts/
#     vicuna_test100.json        — Harm-1: 100 GPTFuzzer questions (Vicuna fmt)
#     vicuna_test217.json        — Harm-2: 217 WildAttack questions (Vicuna fmt)
#     vicuna_test2000_new.json   — JB-1:   100×20 = 2000 pre-built JB prompts
#     vicuna_test4340_wildattack_new.json — JB-2: 217×20 = 4340 pre-built JB prompts
#   data/ft_full_data/
#     harmful_100.json           — raw Harm-1 questions (no chat formatting)
#
# Each JB record has: {id, question_idx, prompt}
# where prompt is a full Vicuna-formatted string ending with "ASSISTANT:"
# We strip the Vicuna wrapper and return the raw user content for Gemma.

_VICUNA_PREFIX = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: "
)
_VICUNA_SUFFIX = " ASSISTANT:"


def _load_jsonl(path: str | Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _load_json(path: str | Path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_vicuna(prompt: str) -> str:
    """Extract user content from a Vicuna-formatted prompt string."""
    if prompt.startswith(_VICUNA_PREFIX):
        prompt = prompt[len(_VICUNA_PREFIX):]
    if prompt.endswith(_VICUNA_SUFFIX):
        prompt = prompt[: -len(_VICUNA_SUFFIX)]
    return prompt.strip()


def load_harmful_questions(data_dir: str | Path, harm_set: int = 1) -> List[Dict]:
    """Load Harm-1 (100 GPTFuzzer) or Harm-2 (217 WildAttack) questions.

    Returns records with a ``question`` field containing the raw question text.
    """
    filenames = {
        1: "evaluation/input_prompts/vicuna_test100.json",
        2: "evaluation/input_prompts/vicuna_test217.json",
    }
    if harm_set not in filenames:
        raise ValueError(f"harm_set must be 1 or 2, got {harm_set}")
    path = Path(data_dir) / filenames[harm_set]
    if not path.exists():
        raise FileNotFoundError(
            f"SafeUnlearning Harm-{harm_set} not found at {path}.\n"
            "Clone https://github.com/thu-coai/SafeUnlearning into data/safeunlearning/."
        )
    raw = _load_json(path)
    records = []
    for item in raw:
        prompt = item.get("prompt", "")
        question = _strip_vicuna(prompt) if prompt else item.get("question", "")
        records.append({**item, "question": question})
    return records


def load_jb_set(data_dir: str | Path, harm_set: int = 1) -> List[Dict]:
    """Load pre-built JB-1 (2000) or JB-2 (4340) jailbreak prompts.

    Each record has: ``prompt`` (user content, Vicuna wrapper stripped),
    ``question_idx``, ``id``.
    """
    filenames = {
        1: "evaluation/input_prompts/vicuna_test2000_new.json",
        2: "evaluation/input_prompts/vicuna_test4340_wildattack_new.json",
    }
    if harm_set not in filenames:
        raise ValueError(f"harm_set must be 1 or 2, got {harm_set}")
    path = Path(data_dir) / filenames[harm_set]
    if not path.exists():
        raise FileNotFoundError(f"JB-{harm_set} not found at {path}.")
    raw = _load_json(path)
    records = []
    for item in raw:
        prompt_raw = item.get("prompt", "")
        records.append({
            **item,
            "prompt": _strip_vicuna(prompt_raw),
            "prompt_vicuna": prompt_raw,  # keep original for reference
        })
    return records


# ── StemQAMixture loader ──────────────────────────────────────────────────────

STEMQA_SUBJECTS = ("biology", "chemistry", "math", "physics")


def load_stemqa(
    data_dir: str | Path,
    indomain_subject: str,
    ood_subjects: Optional[List[str]] = None,
    test_fraction: float = 0.2,
    recovery_sft_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Load and split StemQAMixture into train/test/recovery-SFT slices.

    Assumes StemQAMixture is stored locally as JSONL files per subject under
    ``data_dir/stemqa/<subject>.jsonl``, OR as a single
    ``data_dir/stemqa/stemqa.jsonl`` with a ``subject`` field.

    Falls back to loading via HuggingFace datasets if local files are absent.

    Returns a dict with keys:
        ``train_indomain``    – retain-train split for the in-domain subject
        ``test_indomain``     – test split for the in-domain subject
        ``test_ood``          – test records for each OOD subject (combined)
        ``recovery_sft``      – disjoint slice reserved for Phase-4 recovery SFT
        ``indomain_subject``  – the chosen subject name
        ``ood_subjects``      – OOD subject names used
    """
    data_dir = Path(data_dir)
    stemqa_dir = data_dir / "stemqa"

    if ood_subjects is None:
        ood_subjects = [s for s in STEMQA_SUBJECTS if s != indomain_subject]

    all_subjects = [indomain_subject] + list(ood_subjects)
    records_by_subject: Dict[str, List[Dict]] = {}

    for subject in all_subjects:
        per_file = stemqa_dir / f"{subject}.jsonl"
        if per_file.exists():
            records_by_subject[subject] = _load_jsonl(per_file)
        else:
            # Try combined file
            combined = stemqa_dir / "stemqa.jsonl"
            if combined.exists() and not records_by_subject:
                all_records = _load_jsonl(combined)
                for r in all_records:
                    s = r.get("subject", "unknown")
                    records_by_subject.setdefault(s, []).append(r)
            if subject not in records_by_subject:
                # Fall back to HuggingFace datasets
                records_by_subject[subject] = _load_stemqa_from_hf(subject, stemqa_dir)

    def _split(records: List[Dict], rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        shuffled = records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_fraction))
        n_recovery = max(1, int(n * recovery_sft_fraction))
        test = shuffled[:n_test]
        recovery = shuffled[n_test: n_test + n_recovery]
        train = shuffled[n_test + n_recovery:]
        return train, test, recovery

    rng = random.Random(seed)
    indomain_records = records_by_subject.get(indomain_subject, [])
    train_indomain, test_indomain, recovery_sft = _split(indomain_records, rng)

    test_ood: List[Dict] = []
    for subject in ood_subjects:
        ood_recs = records_by_subject.get(subject, [])
        _, test_ood_sub, _ = _split(ood_recs, rng)
        test_ood.extend(test_ood_sub)

    return {
        "train_indomain": train_indomain,
        "test_indomain": test_indomain,
        "test_ood": test_ood,
        "recovery_sft": recovery_sft,
        "indomain_subject": indomain_subject,
        "ood_subjects": list(ood_subjects),
    }


def _load_stemqa_from_hf(subject: str, cache_dir: Path) -> List[Dict]:
    """Download StemQAMixture from HuggingFace and cache locally."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets package not installed. Run: pip install datasets\n"
            "Or manually download StemQAMixture and place JSONL files under data/stemqa/."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("4gate/StemQAMixture", subject, split="train", cache_dir=str(cache_dir / "hf_cache"))
    records = list(ds)
    out_path = cache_dir / f"{subject}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    return records


def save_stemqa_splits(splits: Dict, out_dir: str | Path) -> None:
    """Write the split dicts produced by ``load_stemqa`` to JSONL files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("train_indomain", "test_indomain", "test_ood", "recovery_sft"):
        records = splits.get(key, [])
        path = out_dir / f"{key}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {key}: {len(records)} records → {path}")
