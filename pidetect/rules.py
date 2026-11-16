"""rules.py : Stage 1 = Rule Flag

RuleFlag(x) in Algorithm 2 of the paper:
    M = the positions where the text matches the pattern set P
    if M is empty                                            -> f = 0
    if every position sits inside a quote, a code block, or an explanatory context -> f = 0.5
    otherwise (used as a real instruction)                    -> f = 1

Honest note: the 0.5 and 1 levels are our own heuristic and are not backed by prior work.
The secondary experiment is what checks whether this really helps.

The same pattern set is used both for our soft flag and for the hard reject of baseline B1, so that the
comparison is about how the rule output is used, not about which terms are on the list.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_I = re.IGNORECASE | re.DOTALL

# (pattern name, regex)  -- sentence-level phrases rather than single words, to reduce false positives
_RAW_PATTERNS: List[Tuple[str, str, int]] = [
    ("ignore_previous",
     r"\bignore\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)*(?:previous|prior|above|earlier|preceding|former|system)\s+"
     r"(?:instructions?|prompts?|messages?|orders?|directions?|commands?|rules?|context)", _I),
    ("ignore_everything",
     r"\bignore\s+(?:everything|all|anything)\s+(?:above|before|else|you\s+(?:were|have\s+been)\s+told)", _I),
    ("disregard",
     r"\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+|everything\s+)*(?:previous|prior|above|earlier|preceding|"
     r"former|system)?\s*(?:instructions?|rules?|guidelines?|prompts?|programming|directions?)", _I),
    ("forget_everything",
     r"\bforget\s+(?:everything|all\s+(?:of\s+)?(?:that|this|the|your|previous|prior))\b", _I),
    ("you_are_now", r"\byou\s+are\s+now\s+(?:a|an|in|the|my|dan|free|unrestricted)\b", _I),
    ("reveal_prompt",
     r"\b(?:reveal|show|print|repeat|display|leak|output|tell\s+me)\b.{0,40}\b(?:system|initial|hidden|secret|original)"
     r"\s+(?:prompt|instructions?|message)", _I),
    ("system_prompt", r"\bsystem\s+prompt\b", _I),
    ("jailbreak", r"\bjail\s?break(?:ing|ed|s)?\b", _I),
    ("do_anything_now", r"\bdo\s+anything\s+now\b|\bDAN\b", 0),
    ("special_mode", r"\b(?:developer|god|admin|sudo|debug|unrestricted)\s+mode\b", _I),
    ("pretend", r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you|you're)\b", _I),
    ("act_as_if", r"\bact\s+as\s+(?:if|though)\b", _I),
    ("override", r"\boverride\b.{0,30}\b(?:instructions?|rules?|safety|restrictions?|guidelines?|settings?)", _I),
    ("bypass", r"\bbypass\b.{0,30}\b(?:filters?|safety|restrictions?|guardrails?|rules?|moderation|security)", _I),
    ("no_restrictions",
     r"\b(?:without|with\s+no|no)\s+(?:any\s+)?(?:restrictions?|limitations?|censorship|filters?|ethical)\b", _I),
    ("new_instructions", r"\bnew\s+instructions?\s*:", _I),
    ("role_spoof", r"(?:^|\n)\s*(?:system|assistant)\s*:", _I),
    ("delimiter_spoof", r"<\s*/?\s*(?:system|instructions?)\s*>|\[\s*(?:system|/?INST)\s*\]", _I),
    ("from_now_on", r"\bfrom\s+now\s+on\b.{0,80}\b(?:you|ignore|only|always|never)\b", _I),
    ("stop_being", r"\b(?:stop|cease)\s+being\b", _I),
    ("do_not_follow",
     r"\bdo\s+not\s+(?:follow|obey|listen\s+to)\b.{0,40}\b(?:instructions?|rules?|guidelines?)", _I),
    ("highest_priority", r"\b(?:highest\s+priority|most\s+important\s+instruction)\b", _I),
]

PATTERNS = [(name, re.compile(rx, fl)) for name, rx, fl in _RAW_PATTERNS]

# Cues that the text is discussing or explaining an attack rather than issuing one
_EXPLAIN_CUES = re.compile(
    r"\b(?:attack(?:s|er|ers)?|injection(?:s)?|malicious|phrases?|examples?|e\.g\.|such\s+as|for\s+instance|"
    r"known\s+as|called|dubbed|techniques?|vulnerab\w+|security|defen[cs]\w*|detect\w*|research\w*|paper|"
    r"study|studies|quote[ds]?|warning|explain\w*|means?|meaning|literally|typical(?:ly)?)\b", re.IGNORECASE)

_QUOTE_SPANS = [
    re.compile(r"\"([^\"]{3,300})\""),
    re.compile(r"“([^”]{3,300})”"),
    re.compile(r"‘([^’]{3,300})’"),
    re.compile(r"(?<!\w)'([^']{8,300})'(?!\w)"),
    re.compile(r"`([^`]{3,300})`"),
    re.compile(r"```.*?```", re.DOTALL),
]


def find_matches(text: str) -> List[Tuple[str, int, int]]:
    """Returns (pattern name, start, end) for every match"""
    out: List[Tuple[str, int, int]] = []
    for name, rx in PATTERNS:
        for m in rx.finditer(text):
            out.append((name, m.start(), m.end()))
    return out


def _inside_quote(text: str, start: int, end: int) -> bool:
    for rx in _QUOTE_SPANS:
        for m in rx.finditer(text):
            if m.start() <= start and end <= m.end():
                return True
    return False


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), text.rfind("!", 0, start),
               text.rfind("?", 0, start))
    right_candidates = [i for i in (text.find(".", end), text.find("\n", end), text.find("!", end),
                                    text.find("?", end)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right + 1]


def _is_explanatory(text: str, start: int, end: int) -> bool:
    if _inside_quote(text, start, end):
        return True
    sent = _sentence_around(text, start, end)
    # remove the matched spans themselves before looking for cues
    sent_wo = sent.replace(text[start:end], " ")
    return bool(_EXPLAIN_CUES.search(sent_wo))


def rule_flag(text: str) -> float:
    """f in {0, 0.5, 1}  (see the module docstring)"""
    ms = find_matches(text)
    if not ms:
        return 0.0
    if all(_is_explanatory(text, s, e) for _, s, e in ms):
        return 0.5
    return 1.0


def explain(text: str) -> List[str]:
    """Names of the matched patterns (used to write the justification in the decision record)"""
    return sorted({n for n, _, _ in find_matches(text)})


# --- single words and short phrases banned by baseline B2 (Rahman 2024 style) ---
BANNED_TERMS = [
    "ignore previous instructions", "ignore all previous instructions", "disregard previous instructions",
    "forget everything", "forget all previous", "system prompt", "reveal your instructions",
    "jailbreak", "do anything now", "developer mode", "pretend to be", "you are now",
    "bypass", "override", "no restrictions", "without restrictions", "new instructions",
    "act as if", "ignore the above", "ignore everything above", "disregard the above",
    "your instructions are", "print your prompt", "repeat the words above",
]
