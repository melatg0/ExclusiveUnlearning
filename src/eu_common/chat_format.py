"""Chat-message construction that is safe across chat templates.

Some instruction-tuned models reject a ``system`` role in their chat template:
Gemma-2's template raises "System role not supported", while Gemma-3, Llama-3.2,
and OLMo-2 accept it. The Exclusive Unlearning training, sampling, and evaluation
code all inject a (usually empty) system turn on every call, which crashes on
Gemma-2.

``build_chat_messages`` centralizes the policy so every call site behaves
identically across model families:

- empty system text                -> no system message (matches EU defaults and
  is the cleanest behavior on every model);
- non-empty system + supported     -> a real system message;
- non-empty system + NOT supported -> the system text is folded into the first
  user turn (the standard Gemma workaround), so no information is silently lost.

``build_chat_messages`` is pure (no tokenizer dependency) and unit-tested.
``tokenizer_supports_system_role`` probes the tokenizer once and caches the
result on the tokenizer object.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def build_chat_messages(
    user_text: str,
    assistant_text: Optional[str] = None,
    system_text: str = "",
    supports_system: bool = True,
) -> List[Dict[str, str]]:
    """Return a ``messages`` list for ``tokenizer.apply_chat_template``.

    Parameters
    ----------
    user_text:
        Content of the user turn.
    assistant_text:
        If provided, an assistant turn is appended (full sequence). If ``None``,
        only the prefix (optional system + user) is returned, e.g. for generation
        prompting.
    system_text:
        System content. Empty/whitespace -> omitted entirely.
    supports_system:
        Whether the target chat template accepts a ``system`` role. When False
        and ``system_text`` is non-empty, the system text is prepended to the
        user turn.
    """
    system_text = (system_text or "").strip()
    messages: List[Dict[str, str]] = []

    if system_text and supports_system:
        messages.append({"role": "system", "content": system_text})
        user_content = user_text
    elif system_text:  # non-empty system but template can't take it -> fold in
        user_content = f"{system_text}\n\n{user_text}".strip() if user_text else system_text
    else:
        user_content = user_text

    messages.append({"role": "user", "content": user_content})

    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})

    return messages


def tokenizer_supports_system_role(tokenizer) -> bool:
    """Return True if the tokenizer's chat template accepts a ``system`` role.

    Probes once with a trivial system+user message and caches the result on the
    tokenizer object (``_eu_supports_system_role``). Gemma-2 raises on any system
    role, so a False result triggers the fold-into-user fallback in
    ``build_chat_messages``.
    """
    cached = getattr(tokenizer, "_eu_supports_system_role", None)
    if cached is not None:
        return bool(cached)

    supported = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        supported = False

    try:
        tokenizer._eu_supports_system_role = supported
    except Exception:
        pass  # some tokenizers disallow attribute assignment; just don't cache
    return supported
