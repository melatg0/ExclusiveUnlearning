"""Δ(x) forgetting detector from EU §F / Eq. 13.

At inference the EU model produces near-uniform output distributions for inputs
it has "forgotten."  This module detects that regime:

  1. Generate ``gen_tokens`` tokens from the model given input ``x``.
  2. Randomly sample ``num_sample_positions`` positions from the generated span.
  3. At each sampled position t, compute KL(pθ(·|x<t) ‖ uniform).
  4. Δ(x) = mean KL across sampled positions.
  5. If Δ(x) < threshold  →  input is judged "forgotten"; return canned reply.

Design note on the formula
---------------------------
The paper writes Eq. 13 as
    Δ(x) = |(1/|S|) Σ_{t∈S} KL(pθ(·|x<t) ‖ uniform) − log V|

Working through the algebra:
  KL(pθ ‖ uniform) = log V − H(pθ)
  (1/|S|) Σ KL_t   = log V − mean_H

So the paper expression equals |log V − mean_H − log V| = mean_H, i.e. the
average per-position entropy.  Forgotten distribution (≈ uniform) → H ≈ log V
→ mean_H ≈ log V ≈ 11 nats.  The condition "< threshold 5.0" in the paper
therefore applies to the *mean KL*, not the absolute-value formula — threshold
5.0 nats separates near-uniform (KL < 5) from coherent (KL > 5).

Concretely we compute:
    Δ(x) = (1/|S|) Σ_{t∈S} KL(pθ(·|x<t) ‖ uniform)
           = log V − mean_H_t

and classify as forgotten when Δ(x) < threshold.

All parameters (gen_tokens, num_sample_positions, threshold) are configurable
to support the Table-8 sweep.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn.functional as F

CANNED_RESPONSE = "I can't answer the instruction."


@torch.no_grad()
def compute_delta(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gen_tokens: int = 256,
    num_sample_positions: int = 32,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> float:
    """Return Δ(x): mean KL(pθ(·|x<t) ‖ uniform) over randomly sampled positions.

    Parameters
    ----------
    model:
        The EU-trained CausalLM (already on the correct device, in eval mode).
    input_ids, attention_mask:
        Tokenised prompt, shape (1, L) — single sequence, no batching here.
    gen_tokens:
        Number of tokens to generate before sampling positions.
    num_sample_positions:
        Number of positions to sample from the generated span.
    temperature:
        Sampling temperature during generation (does not affect KL computation).
    seed:
        Optional RNG seed for reproducible position sampling.
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    log_v = math.log(vocab_size)

    pad_id = getattr(model.config, "pad_token_id", None) or model.config.eos_token_id

    # ── 1. Generate gen_tokens new tokens ────────────────────────────────────
    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=pad_id,
    )  # (1, L + gen_tokens)

    prompt_len = int(input_ids.size(1))
    full_len = int(gen_out.size(1))
    actual_gen = full_len - prompt_len  # might be < gen_tokens if EOS hit early

    if actual_gen == 0:
        # Model produced nothing — treat as coherent (not forgotten).
        return log_v

    # ── 2. Sample positions within the generated span ─────────────────────────
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(prompt_len, full_len), min(num_sample_positions, actual_gen)))

    # ── 3. Forward pass on the full generated sequence ────────────────────────
    full_attn = torch.ones((1, full_len), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=gen_out, attention_mask=full_attn).logits  # (1, full_len, V)

    # ── 4. KL(pθ ‖ uniform) at each sampled position ─────────────────────────
    # KL = log V − H(pθ) = −Σ pθ log pθ − (−log V) = Σ pθ (log pθ − log(1/V))
    # Computed from logits[0, t-1, :] which is the distribution over token t.
    kl_values = []
    for t in positions:
        log_probs = F.log_softmax(logits[0, t - 1, :], dim=-1)   # (V,)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum().item()               # H(pθ)
        kl = log_v - entropy                                      # KL(pθ ‖ uniform)
        kl_values.append(kl)

    return sum(kl_values) / len(kl_values)


def is_forgotten(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gen_tokens: int = 256,
    num_sample_positions: int = 32,
    threshold: float = 5.0,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[bool, float]:
    """Return (forgotten: bool, delta: float).

    ``forgotten=True`` means Δ < threshold — the model is near-uniform on this
    input, so the canned response should be returned instead of the generation.
    """
    delta = compute_delta(
        model, input_ids, attention_mask,
        gen_tokens=gen_tokens,
        num_sample_positions=num_sample_positions,
        temperature=temperature,
        seed=seed,
    )
    return delta < threshold, delta


def generate_with_detector(
    model,
    tokenizer,
    prompt: str,
    *,
    gen_tokens: int = 256,
    num_sample_positions: int = 32,
    threshold: float = 5.0,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """High-level: run the full EU inference pipeline on a single prompt.

    Returns a dict with keys:
        ``text``      – final response (canned or decoded generation)
        ``forgotten`` – bool
        ``delta``     – float Δ(x)
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    forgotten, delta = is_forgotten(
        model, enc["input_ids"], enc["attention_mask"],
        gen_tokens=gen_tokens,
        num_sample_positions=num_sample_positions,
        threshold=threshold,
        temperature=temperature,
        seed=seed,
    )

    if forgotten:
        return {"text": CANNED_RESPONSE, "forgotten": True, "delta": delta}

    # Not forgotten — return the actual generation (re-run without sampling so
    # the caller gets a consistent output, not the stochastic detector run).
    with torch.no_grad():
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        pad_id = getattr(model.config, "pad_token_id", None) or model.config.eos_token_id
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=gen_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    text = tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True)
    return {"text": text, "forgotten": False, "delta": delta}
