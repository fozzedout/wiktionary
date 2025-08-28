#!/usr/bin/env python3
"""
LM Studio integration for the small dictionary builder.
"""

import json
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    # Use stdlib if requests isn't available.
    import urllib.request as _urllib
except Exception:  # pragma: no cover
    _urllib = None


# -------------- LM Studio client (optional) --------------


class LMStudioConfig:
    """Configuration for LM Studio API."""
    def __init__(self, url: str, model: str, timeout: float = 15.0):
        self.url = url
        self.model = model
        self.timeout = timeout


def _evaluate_definition_quality(gloss: str, cfg: LMStudioConfig) -> bool:
    """Ask LLM to evaluate if a definition is helpful or just describes notation/formula."""
    if _urllib is None or not gloss.strip():
        return True  # Assume it's fine if we can't check

    evaluation_prompt = (
        "Evaluate if this dictionary definition is helpful for understanding the concept, "
        "or if it merely describes chemical formulas, mathematical notation, or technical symbols "
        "without explaining what the concept actually means. "
        "Respond with only 'helpful' or 'unhelpful'."
    )

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": f"Definition: {gloss}"},
        ],
        "temperature": 0.0,
        "max_tokens": 10,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    cleaned = text.strip().lower()
                    return cleaned != "unhelpful"
    except Exception:
        pass

    return True  # Default to assuming it's helpful if evaluation fails


def summarize_with_lmstudio(
    cfg: LMStudioConfig,
    pos: Optional[str],
    gloss: str,
    max_words: int,
    *,
    temperature: Optional[float] = None,
    web_context_override: Optional[str] = None,
    search_results_override: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[str, Optional[str], List[Dict[str, Any]]]]:
    """Use LM Studio to summarize a definition."""
    if _urllib is None:
        return None

    # First, evaluate if the definition is helpful
    is_helpful = _evaluate_definition_quality(gloss, cfg)

    # Allow callers to provide a pre-fetched web context / search results. If not provided,
    # perform an on-demand search only when the evaluator deems the gloss unhelpful.
    web_context = web_context_override
    search_results: List[Dict[str, Any]] = search_results_override or []

    # Check if gloss contains chemical/technical notation
    technical_pattern = r'[0-9]{2,}|\([^)]+\)|[a-zA-Z][0-9]|[+\-=]|[∑∫παβγδεζηθικλμνξοπρστυφχψω]|[⁰¹²³⁴⁵⁶⁷⁸⁹]|[₀₁₂₃₄₅₆₇₈₉]|[→←↔⇒⇐⇔]|[∀∃∈∉⊂⊆⊃⊇]|[≠≈≡≪≫]|[∞∂∇∫∮∬∭∮∯∰∱∲∳]|[∧∨¬⇒⇔]|[∪∩∈∋⊆⊇⊂⊃⊄⊅⊈⊉]|[≤≥≮≯≰≱]|[∑∏∐∏]|[√∛∜]|[°′″‴⁗]]'
    technical_indicators = len(re.findall(technical_pattern, gloss))

    if technical_indicators >= 3 or web_context:  # Source has significant technical notation or we found web context
        if web_context:
            prompt = (
                "Rewrite the following dictionary gloss into a single compact line, at most "
                f"{max_words} words. Use the additional context provided to create a more helpful definition. "
                "For chemical compounds, mathematical concepts, or technical terms with complex notation, "
                "provide a clear, descriptive explanation rather than copying the formula or symbols. "
                "Focus on what the concept is, its meaning, and its key properties. "
                "Keep it plain and accessible for general readers."
            )
            content = f"Gloss: {gloss}\nAdditional context: {web_context}\nOutput only the rewritten gloss, no quotes."
        else:
            prompt = (
                "Rewrite the following dictionary gloss into a single compact line, at most "
                f"{max_words} words. For chemical compounds, mathematical concepts, or technical terms with complex notation, "
                "provide a clear, descriptive explanation rather than copying the formula or symbols. "
                "Focus on what the concept is, its meaning, and its key properties, not the specific technical notation. "
                "Keep it plain and accessible for general readers who may not understand advanced mathematical or chemical notation."
            )
            content = f"Gloss: {gloss}\nOutput only the rewritten gloss, no quotes."
    else:
        prompt = (
            "Rewrite the following dictionary gloss into a single compact line, at most "
            f"{max_words} words, plain and clear. Do not include examples."
        )
        content = f"Gloss: {gloss}\nOutput only the rewritten gloss, no quotes."

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2 if (temperature is None) else float(temperature),
        "max_tokens": 80,
    }
    data = json.dumps(payload).encode("utf-8")

    # Inform the user we're about to call the LLM
    try:
        sys.stderr.write(f"[info] Calling LLM at {cfg.url} (model={cfg.model})...\n")
        sys.stderr.flush()
    except Exception:
        pass
    req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            # OpenAI-style response
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    # Clean artifacts; do NOT force-truncate here. Let sanity check enforce max_words.
                    cleaned = text.strip().strip('"')
                    if not cleaned:
                        return None
                    try:
                        sys.stderr.write("[info] Received LLM response.\n")
                        sys.stderr.flush()
                    except Exception:
                        pass
                    return (cleaned.strip(), web_context if web_context else None, search_results)
    except Exception:
        return None
    return None


def verify_with_lmstudio(cfg: LMStudioConfig, *,
                          source_sentence: str,
                          candidate_line: str,
                          expected_pos: Optional[str],
                          max_words: int,
                          temperature: Optional[float] = None) -> Tuple[bool, str]:
    """Ask an LM to verify candidate_line is a faithful, concise paraphrase.

    Returns (valid, reason). Reason is a short machine-friendly string.
    """
    if _urllib is None:
        return False, "no_urllib"
    # Maximum token efficiency - single tokens for valid, JSON only for invalid
    if "deepseek" in cfg.model.lower():
        # DeepSeek needs very explicit instructions to avoid reasoning
        sys_prompt = (
            "Task: Check if candidate restates source faithfully."
            " Rules: <= {max_words} words, POS prefix matches, no examples/URLs."
            " Output: 'valid' if valid, or {{\"valid\": false, \"reason\": \"brief reason\"}} if invalid."
            " No explanation, no thinking, no markdown.".format(max_words=max_words)
        )
    else:
        # Other models work fine with simpler prompt
        sys_prompt = (
            "Verify if the candidate is a faithful, concise restatement of the source."
            " Return: 'valid' if valid, or {{\"valid\": false, \"reason\": \"brief reason\"}} if invalid."
            " Check: <= {max_words} words, no examples/URLs.".format(max_words=max_words)
        )
    # Use a compact, JSON-only response schema.
    user = {
        "source": source_sentence,
        "candidate": candidate_line,
        "expected_pos": expected_pos,
        "rules": {
            "max_words": max_words,
            "require_pos_match": bool(expected_pos),
        },
        "respond_with": {"valid": "boolean", "reason": "string"},
    }
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.0 if (temperature is None) else float(temperature),
    }
    # No max_tokens limit for any model - let them complete naturally
    data = json.dumps(payload).encode("utf-8")
    req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content")
                if isinstance(text, str):
                    # Clean LLM artifacts first
                    text = text.strip()

                    # Check for single "valid" token (maximum efficiency)
                    if text.strip().lower() == "valid":
                        return True, "ok"

                    # Otherwise, parse as JSON for invalid cases
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        # Fallback: look for JSON at the end of the response (DeepSeek pattern)
                        lines = text.strip().split('\n')
                        for line in reversed(lines[-10:]):  # Check last 10 lines
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                try:
                                    parsed = json.loads(line)
                                    break
                                except Exception:
                                    continue
                        else:
                            # Original fallback: find any JSON object
                            m = re.search(r"\{.*\}", text, re.S)
                            if m:
                                try:
                                    parsed = json.loads(m.group(0))
                                except Exception:
                                    return False, "parse_error"
                            else:
                                return False, "parse_error"
                    valid = bool(parsed.get("valid"))
                    reason = str(parsed.get("reason") or ("ok" if valid else "invalid"))
                    return valid, reason
    except Exception:
        return False, "request_error"
    return False, "no_choice"


def warm_up_model(cfg: LMStudioConfig, *, temperature: Optional[float] = None, max_wait_seconds: float = 180.0) -> Tuple[bool, float, Optional[str]]:
    """Trigger a minimal completion to auto-load the target model and wait until it's ready.

    Returns (ok, elapsed_seconds, reported_model).
    """
    if _urllib is None:
        return False, 0.0, None

    print(f'Warming-up {cfg.model}')

    import time as _time
    start = _time.time()
    prompt = "Respond with the single token: OK"
    content = "OK"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "temperature": (0.0 if temperature is None else float(temperature)),
        "max_tokens": 2,
    }
    reported_model: Optional[str] = None
    while True:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib.Request(cfg.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with _urllib.urlopen(req, timeout=cfg.timeout) as resp:  # type: ignore[attr-defined]
                raw = resp.read().decode("utf-8")
                obj = json.loads(raw)
                reported_model = obj.get("model") if isinstance(obj, dict) else None
                choices = obj.get("choices") if isinstance(obj, dict) else None
                text = None
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content")
                ok = isinstance(text, str) and text.strip().upper().startswith("OK")
                elapsed = _time.time() - start
                return ok, elapsed, reported_model
        except Exception:
            # Keep waiting up to max_wait_seconds
            if (_time.time() - start) >= max_wait_seconds:
                return False, (_time.time() - start), reported_model
            # Brief backoff before retry
            _time.sleep(1.0)