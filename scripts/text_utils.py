#!/usr/bin/env python3
"""
Text processing utilities for the small dictionary builder.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------- Parsing helpers --------------

WORD_KEYS = (
    "word",
    "title",
    "headword",
    "lemma",
    "entry",
)

POS_KEYS = (
    "pos",
    "partOfSpeech",
    "part_of_speech",
    "part-of-speech",
)

GLOSS_KEYS = (
    "gloss",
    "definition",
    "def",
    "sense",
    "meaning",
    "desc",
    "text",
)

SENSES_KEYS = (
    "senses",
    "definitions",
    "meanings",
    "glosses",
    "entries",
)

LANG_KEYS = (
    "lang",
    "language",
    "lang_name",
)

LANG_CODE_KEYS = (
    "lang_code",
    "langCode",
    "language_code",
)


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Get the first present value from a dictionary using a list of possible keys."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def extract_word(obj: Dict[str, Any]) -> Optional[str]:
    """Extract the word from a dictionary entry object."""
    w = _first_present(obj, WORD_KEYS)
    if isinstance(w, str):
        w = w.strip()
        # Ignore entries that look like phrases if desired? Keep as-is for now.
        return w if w else None
    return None


def extract_language(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract language information from a dictionary entry object."""
    lang = _first_present(obj, LANG_KEYS)
    code = _first_present(obj, LANG_CODE_KEYS)
    if isinstance(lang, str):
        lang = lang.strip()
    if isinstance(code, str):
        code = code.strip()
    return (lang if isinstance(lang, str) and lang else None,
            code if isinstance(code, str) and code else None)


def normalize_pos(pos: Optional[str]) -> Optional[str]:
    """Normalize part-of-speech tags to standard forms."""
    if not pos:
        return None
    p = str(pos).strip().lower()
    # Normalize common variants
    mapping = {
        "n": "noun",
        "v": "verb",
        "adj": "adjective",
        "adv": "adverb",
        "prep": "preposition",
        "pron": "pronoun",
        "det": "determiner",
        "conj": "conjunction",
        "interj": "interjection",
        "part": "particle",
        "num": "numeral",
    }
    return mapping.get(p, p)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Very light sentence splitter.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def _word_count(text: str) -> int:
    """Count the number of words in text."""
    return len(re.findall(r"\b\W+\b", text))


def _limit_words(text: str, max_words: int) -> str:
    """Limit text to a maximum number of words."""
    words = re.findall(r"\b\w+\b", text)
    if len(words) <= max_words:
        return text.strip()
    # Rebuild truncated string preserving original non-word chars roughly
    # Simple approach: join by space and add period if missing.
    truncated = " ".join(words[:max_words]).strip()
    if not re.search(r"[.!?]$", truncated):
        truncated += "."
    return truncated


def strip_llm_artifacts(text: str) -> str:
    """Remove LLM meta tags and any stray HTML/XML tags from text."""
    s = text or ""
    # Remove hidden reasoning blocks entirely (content and tags)
    s = re.sub(r"(?is)<\s*(think|reasoning|reflection|chain[-_ ]?of[-_ ]?thought)[^>]*>.*?<\s*/\s*\1\s*>", " ", s)
    # Remove markdown code blocks (```json ... ```)
    s = re.sub(r"(?s)```\s*json\s*(.*?)\s*```", r"\1", s)
    s = re.sub(r"(?s)```\s*(.*?)\s*```", r"\1", s)
    # Strip common inline tags (keep inner text already remaining)
    s = re.sub(r"(?is)</?\s*(br|b|i|u|em|strong|p|span|div|code|pre|blockquote|ul|ol|li|sup|sub)\b[^>]*>", " ", s)
    # Fallback: remove any remaining tags
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_gloss(gloss: str) -> str:
    """Clean a glossary definition by removing wikilinks and extra whitespace."""
    # Remove example brackets or wikilinks like [[word]]
    g = re.sub(r"\[\[(.*?)\]\]", r"\1", gloss)
    # Collapse whitespace
    g = re.sub(r"\s+", " ", g).strip()
    return g


def _stem(tok: str) -> str:
    """Very light stemmer to improve token overlap on paraphrases.

    Handles simple English variants: plurals, -ing/-ed verbs, -ly/-ally adverbs.
    Avoids heavy dependencies and keeps behavior conservative.
    """
    t = tok
    if len(t) > 4 and t.endswith("ally"):
        # pneumatically -> pneumatic
        return t[:-4]
    if len(t) > 4 and t.endswith("ly"):
        # quickly -> quick
        return t[:-2]
    if len(t) > 5 and t.endswith("ing"):
        # carving -> carv(e)
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        # pressurized -> pressuriz(e)
        return t[:-2]
    if len(t) > 3 and t.endswith("ies"):
        # candies -> candy
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es"):
        # boxes -> box; cases -> cas(e)
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        # rocks -> rock
        return t[:-1]
    return t


def _tokens(text: str) -> List[str]:
    """Tokenize text for comparison purposes."""
    # Normalize by removing punctuation and hyphens so source/candidate tokenization aligns.
    s = re.sub(r"[^\w]+", " ", text.lower())
    raw = re.findall(r"\w+", s)
    return [_stem(t) for t in raw]


def sanity_check_line_with_reason(source_sentence: str, candidate_line: str, max_words: int, expected_pos: Optional[str]) -> Tuple[bool, str]:
    """Check if a candidate line is a valid definition with reason."""
    # Reject if any residual tags remain
    if re.search(r"<[^>]+>", candidate_line or "", re.S):
        return False, "tags"
    # Check word count directly on the candidate line
    # Allow lines that are only 2-3 words over the limit
    if _word_count(candidate_line) > max_words + 3:
        return False, "too_long"
    # Remove POS prefix checking since we no longer mandate it
    cand_tokens = set(_tokens(candidate_line))
    if not cand_tokens:
        return False, "empty"
    if 'http://' in candidate_line or 'https://' in candidate_line:
        return False, "url"
    # Allow quotes in dictionary definitions - they're commonly used for proper names, examples, etc.
    # if '"' in candidate_line:
    #     return False, "quotes"
    return True, "ok"


def sanity_check_line(source_sentence: str, candidate_line: str, max_words: int, expected_pos: Optional[str]) -> bool:
    """Check if a candidate line is a valid definition."""
    ok, _ = sanity_check_line_with_reason(source_sentence, candidate_line, max_words, expected_pos)
    return ok


def format_def_line(pos: Optional[str], gloss: str, max_words: int) -> Optional[str]:
    """Format a definition line with optional POS prefix."""
    g = clean_gloss(gloss)
    if not g:
        return None
    # Prefer first sentence; fall back to whole gloss
    first = _split_sentences(g)[0] if _split_sentences(g) else g
    first = _limit_words(first, max_words)
    # Remove POS prefix requirement - just return the formatted definition
    return first


def first_sentence_word_count(gloss: str) -> int:
    """Count words in the first sentence of a gloss."""
    g = clean_gloss(gloss)
    parts = _split_sentences(g)
    s = parts[0] if parts else g
    return _word_count(s)


def extract_definitions(obj: Dict[str, Any]) -> List[Tuple[Optional[str], str]]:
    """Return list of (pos, gloss) pairs from a heterogenous entry object."""
    out: List[Tuple[Optional[str], str]] = []

    # Shape: entries: { pos: [ {definition: ...}, ... ] }
    entries = obj.get("entries")
    if isinstance(entries, dict):
        for pos_key, items in entries.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict):
                    g = _first_present(it, GLOSS_KEYS)
                    if isinstance(g, str):
                        out.append((pos_key, g))
                elif isinstance(it, str):
                    out.append((pos_key, it))

    # If there is a flat 'definitions' list of strings
    flat = _first_present(obj, ("definitions", "glosses"))
    if isinstance(flat, list) and flat and all(isinstance(x, str) for x in flat):
        pos = _first_present(obj, POS_KEYS)
        for g in flat:
            out.append((pos, str(g)))

    # Common shape: senses: [{pos, gloss}, ...] or senses: [{gloss}, ...] with pos on parent
    senses = _first_present(obj, SENSES_KEYS)
    if isinstance(senses, list):
        parent_pos = _first_present(obj, POS_KEYS)
        for s in senses:
            if isinstance(s, dict):
                s_pos = _first_present(s, POS_KEYS) or parent_pos
                g = _first_present(s, GLOSS_KEYS)
                if isinstance(g, str):
                    out.append((s_pos, g))
                # Sometimes nested structures e.g., {"glosses": ["..."]}
                g_list = _first_present(s, ("glosses", "definitions"))
                if isinstance(g_list, list):
                    for gg in g_list:
                        if isinstance(gg, str):
                            out.append((s_pos, gg))

    # Fallback: try top-level gloss-like fields
    for k in GLOSS_KEYS:
        g = obj.get(k)
        if isinstance(g, str):
            out.append((_first_present(obj, POS_KEYS), g))

    # Deduplicate identical (pos, gloss)
    dedup = list(dict.fromkeys(out))
    return dedup