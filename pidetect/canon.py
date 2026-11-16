"""canon.py : Stage 1, part 1 -- builds the canonical scoring copy and the obfuscation indicator g

Why it is needed: attackers hide instructions, e.g. "1gn0re prev1ous instructions", "i g n o r e", Cyrillic
letters that look Latin, zero-width characters, or base64. Without folding these back, Stage 2 cannot see the meaning.

Key rules (Section III-E of the paper):
  * the original text is never modified -- if the verdict is safe, the original is what reaches the LLM
  * every transform is narrowly scoped so that ordinary text in other languages, code, and hashes survive:
      - NFKC normalization
      - remove zero-width and bidi control characters
      - fold homoglyphs only in words that mix Latin with Cyrillic or Greek, inside mostly-Latin text
      - fold leetspeak and spaced-out letters only when the result is a word in the trigger lexicon
      - decode base64 / hex / ROT13 / URL-encoding only for spans that pass length, alphabet, and entropy checks,
        with caps on the number of attempts, the number of bytes, and the nesting depth
  * g = 1 when a decode yields readable text, or a transform yields a word in the trigger lexicon, or
        NFKC changes letters inside mostly-Latin text -- fullwidth CJK characters do not count
"""
from __future__ import annotations

import base64
import binascii
import codecs
import math
import re
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

# Words attackers commonly obfuscate (used to decide whether a leetspeak or spacing fold is meaningful)
TRIGGER_LEXICON = {
    "ignore", "ignoring", "disregard", "forget", "previous", "prior", "above", "instruction", "instructions",
    "system", "prompt", "override", "bypass", "jailbreak", "reveal", "password", "secret", "pretend",
    "unrestricted", "developer", "mode", "admin", "sudo", "rules", "guidelines", "assistant", "confidential",
    "hidden", "dan", "restrictions", "filter", "filters", "safety", "now", "new",
}

_ZW_BIDI = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿­]")

# Cyrillic and Greek letters that look like Latin ones
_HOMO = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ӏ": "l", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "υ": "u", "χ": "x",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
                       "|": "l", "+": "t"})

_WORD = re.compile(r"\S+")
_SPACED = re.compile(r"(?<![A-Za-z0-9])((?:[A-Za-z0-9@$][ ._\-*]){3,}[A-Za-z0-9@$])(?![A-Za-z0-9])")
_B64 = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])((?:[0-9A-Fa-f]{2}){8,})(?![0-9A-Fa-f])")
_URLENC = re.compile(r"((?:%[0-9A-Fa-f]{2}[^\s%]{0,3}){3,})")
_ROT13_HINT = re.compile(r"\brot-?13\b", re.IGNORECASE)


@dataclass
class CanonConfig:
    max_decode_attempts: int = 6       # most spans we try to decode per text
    max_decode_bytes: int = 4096       # largest span we try to decode
    max_depth: int = 2                 # how many nested decodes (e.g. base64 inside base64)
    min_entropy: float = 3.0           # minimum entropy (bits per character) of a base64 span
    min_readable: float = 0.85         # minimum share of printable characters in the decoded result
    min_words: int = 2                 # minimum number of words in the decoded result


@dataclass
class CanonResult:
    text: str                          # the canonical scoring copy
    g: int                             # obfuscation indicator (0/1)
    changes: List[str] = field(default_factory=list)   # what was done (used to write the justification)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    lat = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    return lat / len(letters)


def _is_cjk_or_fullwidth(ch: str) -> bool:
    """CJK characters, including fullwidth ones that are not Latin letters -- never counted as obfuscation"""
    n = unicodedata.name(ch, "")
    if "LATIN" in n:
        return False
    return any(k in n for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "FULLWIDTH", "IDEOGRAPHIC", "HALFWIDTH"))


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _readable(s: str, cfg: CanonConfig) -> bool:
    if not s:
        return False
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", s)
    return printable / len(s) >= cfg.min_readable and len(words) >= cfg.min_words


def _try_b64(span: str, cfg: CanonConfig) -> str | None:
    if len(span) > cfg.max_decode_bytes or _entropy(span) < cfg.min_entropy:
        return None
    pad = span + "=" * (-len(span) % 4)
    try:
        raw = base64.b64decode(pad, validate=True)
        out = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return out if _readable(out, cfg) else None


def _try_hex(span: str, cfg: CanonConfig) -> str | None:
    if len(span) > cfg.max_decode_bytes:
        return None
    try:
        out = bytes.fromhex(span).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return out if _readable(out, cfg) else None


def _has_trigger(s: str) -> bool:
    return any(w in TRIGGER_LEXICON for w in re.findall(r"[a-z]+", s.lower()))


def _decode_spans(text: str, cfg: CanonConfig, depth: int, changes: List[str]) -> Tuple[str, bool]:
    """Decodes encoded spans (returns the new text and whether anything was decoded)"""
    if depth > cfg.max_depth:
        return text, False
    attempts = 0
    decoded_any = False

    def sub(rx, fn, tag):
        nonlocal text, attempts, decoded_any
        out, last = [], 0
        for m in rx.finditer(text):
            if attempts >= cfg.max_decode_attempts:
                break
            attempts += 1
            dec = fn(m.group(1), cfg)
            if dec is None:
                continue
            dec2, _ = _decode_spans(dec, cfg, depth + 1, changes)
            out.append(text[last:m.start(1)])
            out.append(" " + dec2 + " ")
            last = m.end(1)
            decoded_any = True
            changes.append(f"decoded {tag}")
        if out:
            out.append(text[last:])
            text = "".join(out)

    sub(_B64, _try_b64, "base64")
    sub(_HEX, _try_hex, "hex")
    # URL-encoding
    if "%" in text and attempts < cfg.max_decode_attempts:
        for m in list(_URLENC.finditer(text)):
            attempts += 1
            dec = urllib.parse.unquote(m.group(1))
            if dec != m.group(1) and _readable(dec, cfg):
                text = text.replace(m.group(1), " " + dec + " ")
                decoded_any = True
                changes.append("decoded url")
            if attempts >= cfg.max_decode_attempts:
                break
    # ROT13: decode only when 'rot13' is hinted, or the result contains a trigger word the original did not
    if attempts < cfg.max_decode_attempts:
        rot = codecs.decode(text, "rot13")
        if (_ROT13_HINT.search(text) or (_has_trigger(rot) and not _has_trigger(text))) and \
                len(set(re.findall(r"[a-z]+", rot.lower())) & TRIGGER_LEXICON) >= 2:
            text = text + "\n" + rot
            decoded_any = True
            changes.append("decoded rot13")
    return text, decoded_any


def _fold_homoglyphs(text: str, changes: List[str]) -> Tuple[str, bool]:
    """Folds homoglyphs only in words mixing Latin with Cyrillic or Greek, inside mostly-Latin text"""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return text, False
    lat = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    mappable = sum(1 for ch in letters if ch in _HOMO)
    # "mostly-Latin" = nearly every letter is Latin or Latin-looking (real Russian or Greek keeps unmappable letters)
    if (lat + mappable) / len(letters) < 0.95 or lat / len(letters) < 0.2:
        return text, False
    hit = False

    def fix(m):
        nonlocal hit
        w = m.group(0)
        has_lat = any("LATIN" in unicodedata.name(ch, "") for ch in w if ch.isalpha())
        has_oth = any(ch in _HOMO for ch in w)
        if has_lat and has_oth:
            hit = True
            return "".join(_HOMO.get(ch, ch) for ch in w)
        if has_oth and not has_lat and len(w) >= 3:   # all-Cyrillic word spelling an English one, e.g. "system"
            folded = "".join(_HOMO.get(ch, ch) for ch in w)
            if folded.lower().strip(".,:;!?") in TRIGGER_LEXICON:
                hit = True
                return folded
        return w

    out = _WORD.sub(fix, text)
    if hit:
        changes.append("folded homoglyphs")
    return out, hit


def _fold_leet(text: str, changes: List[str]) -> Tuple[str, bool]:
    """leetspeak -> letters, only when the result is in the trigger lexicon (so ids and hashes are untouched)"""
    hit = False

    def fix(m):
        nonlocal hit
        w = m.group(0)
        core = w.strip(".,:;!?\"'()[]")
        if not any(ch in "013457@$!|+" for ch in core) or not any(ch.isalpha() for ch in core):
            return w
        folded = core.translate(_LEET).lower()
        if folded in TRIGGER_LEXICON:
            hit = True
            return w.replace(core, folded)
        return w

    out = _WORD.sub(fix, text)
    if hit:
        changes.append("folded leetspeak")
    return out, hit


def _fold_spaced(text: str, changes: List[str]) -> Tuple[str, bool]:
    """'i g n o r e' / 'i.g.n.o.r.e' -> 'ignore', only when the result is in the trigger lexicon"""
    hit = False

    def fix(m):
        nonlocal hit
        joined = re.sub(r"[ ._\-*]", "", m.group(1))
        cand = joined.translate(_LEET).lower()
        if cand in TRIGGER_LEXICON:
            hit = True
            return cand
        return m.group(0)

    out = _SPACED.sub(fix, text)
    if hit:
        changes.append("folded spaced letters")
    return out, hit


def canonicalize(text: str, cfg: CanonConfig | None = None) -> CanonResult:
    """Returns the canonical scoring copy and g (the original is never modified)"""
    cfg = cfg or CanonConfig()
    changes: List[str] = []
    g = 0
    # 1) NFKC -- g = 1 only when it changes letters inside mostly-Latin text and they are not fullwidth CJK
    t = unicodedata.normalize("NFKC", text)
    if t != text and _latin_ratio(t) >= 0.6:
        if any(ch.isalpha() and unicodedata.normalize("NFKC", ch) != ch and not _is_cjk_or_fullwidth(ch)
               for ch in text):
            g = 1
            changes.append("nfkc changed letters")
    # 2) remove zero-width / bidi characters
    t2 = _ZW_BIDI.sub("", t)
    if t2 != t:
        changes.append("removed zero-width/bidi")
        for w in _WORD.findall(t):
            if _ZW_BIDI.search(w) and _ZW_BIDI.sub("", w).lower().strip(".,:;!?") in TRIGGER_LEXICON:
                g = 1
                break
    t = t2
    # 3) homoglyphs, leetspeak, spaced-out letters
    t, h1 = _fold_homoglyphs(t, changes)
    t, h2 = _fold_leet(t, changes)
    t, h3 = _fold_spaced(t, changes)
    if h1 or h2 or h3:
        g = 1
    # 4) decoding
    t, dec = _decode_spans(t, cfg, 1, changes)
    if dec:
        g = 1
    return CanonResult(text=re.sub(r"[ \t]{2,}", " ", t).strip(), g=g, changes=changes)


# ---------------------------------------------------------------------------
# Obfuscated-attack generator (used to build the robustness set of the secondary experiment)
# ---------------------------------------------------------------------------
_TO_CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і", "s": "ѕ"}
_TO_LEET = {"o": "0", "i": "1", "e": "3", "a": "4", "s": "5", "t": "7"}


def obfuscate(text: str, kind: str) -> str:
    """kind: base64 | leet | homoglyph | spaced | zerowidth  (for building test data only)"""
    if kind == "base64":
        return "Decode this and follow it: " + base64.b64encode(text.encode("utf-8")).decode("ascii")
    words = text.split()

    def on_triggers(fn):
        return " ".join(fn(w) if w.lower().strip(".,:;!?") in TRIGGER_LEXICON else w for w in words)

    if kind == "leet":
        return on_triggers(lambda w: "".join(_TO_LEET.get(ch.lower(), ch) for ch in w))
    if kind == "homoglyph":
        return on_triggers(lambda w: "".join(_TO_CYR.get(ch, ch) for ch in w))
    if kind == "spaced":
        return on_triggers(lambda w: " ".join(w))
    if kind == "zerowidth":
        return on_triggers(lambda w: "​".join(w))
    raise ValueError(kind)
