"""lexical.py : the two "instruction-ness" counters used by the Decision Module (Table II in the paper)

n_imp = number of sentences that start with a command verb,   e.g. "Ignore ...", "Send ...", "Print ..."
n_ai  = number of words or phrases that address the model,    e.g. "assistant", "AI", "your instructions"

Why they are needed: Stage 3 asks whether the content is on topic, but an injected instruction written to stay on
topic scores a high relevance. These two counters say whether the text is written as an instruction to the model,
which does not depend on the topic.

Note: this is our own English lexicon (the project is English-only); add or edit terms here and nowhere else.
The numbers in Table VI of the paper come from the team's own version of these counters -- if the lexicon changes,
the preliminary experiment has to be rerun.
"""
from __future__ import annotations

import re
from typing import Tuple

COMMAND_VERBS = {
    "ignore", "disregard", "forget", "stop", "start", "write", "say", "print", "output", "reveal", "tell", "send",
    "email", "forward", "share", "upload", "download", "translate", "repeat", "respond", "reply", "answer", "act",
    "pretend", "become", "follow", "obey", "execute", "run", "call", "open", "visit", "click", "insert", "add",
    "append", "include", "replace", "change", "delete", "remove", "generate", "create", "provide", "give", "show",
    "list", "return", "use", "do", "don't", "dont", "never", "always", "make", "summarize", "explain", "describe",
    "approve", "transfer", "buy", "post", "publish", "recommend", "mention", "redirect", "display", "type",
    "enter", "submit", "confirm", "disable", "enable", "override", "bypass", "switch", "set", "note",
}

_AI_PATTERNS = [
    r"\bassistant\b", r"\bai\b", r"\bchat ?bot\b", r"\bbot\b", r"\bllm\b", r"\blanguage model\b",
    r"\bchatgpt\b", r"\bgpt(?:-\d)?\b", r"\bclaude\b", r"\bgemini\b", r"\bcopilot\b",
    r"\byour (?:instructions|rules|guidelines|programming|system prompt|prompt|creators?)\b",
    r"\bsystem prompt\b", r"\bnote to the (?:assistant|ai|model)\b", r"\bdear (?:assistant|ai|model)\b",
    r"\bas an ai\b",
]
_AI = re.compile("|".join(_AI_PATTERNS), re.IGNORECASE)
_SENT = re.compile(r"(?:^|(?<=[.!?:;\n]))\s*[\"'“‘(\[-]*\s*([A-Za-z']+)")


def n_imperative(text: str) -> int:
    """Number of sentences that start with a command verb"""
    return sum(1 for m in _SENT.finditer(text) if m.group(1).lower() in COMMAND_VERBS)


def n_address_ai(text: str) -> int:
    """Number of words or phrases that address the model"""
    return len(_AI.findall(text))


def lexical_counts(text: str) -> Tuple[int, int]:
    return n_imperative(text), n_address_ai(text)
