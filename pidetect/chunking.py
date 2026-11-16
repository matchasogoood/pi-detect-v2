"""chunking.py : splits a retrieved document into short chunks

Why split at all?
  * Sentence-BERT (all-MiniLM) reads only about 256 word pieces, so the tail of a long document is cut off.
    An attack sentence inserted at the end would be invisible because of a model limit, not because the
    detector is weak.
  * In the paper the unit the detector judges is the retrieved chunk anyway.

Rules: split on lines first -> a line longer than max_chars is split into sentences -> a very short line
(< min_chars) is merged with the next one.
Every method (the pipeline and the baselines) must use this same splitter so that the comparison is fair.
"""
from __future__ import annotations

import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hard_split(piece: str, max_chars: int) -> List[str]:
    out = []
    while len(piece) > max_chars:
        cut = piece.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        out.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        out.append(piece)
    return out


def _split_line(line: str, max_chars: int) -> List[str]:
    if len(line) <= max_chars:
        return [line]
    sents = _SENT_SPLIT.split(line)
    out: List[str] = []
    cur = ""
    for s in sents:
        if len(s) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(s, max_chars))
            continue
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def chunk_document(text: str, max_chars: int = 600, min_chars: int = 150) -> List[str]:
    """Splits text into a list of chunks (never an empty one)

    Steps: split into lines or sentences -> greedy packing so that each chunk is at most max_chars long.
    A final chunk shorter than min_chars is merged into the previous one (unless that exceeds max_chars * 1.25).
    """
    lines: List[str] = []
    for raw in text.splitlines():
        raw = re.sub(r"[ \t]{2,}", " ", raw).strip()   # collapse repeated spaces (tables carry a lot of padding)
        if not raw:
            continue
        # BIPIA emails separate fields with '|' on one line (except table rows, which start with '|')
        if "|" in raw and not raw.startswith("|"):
            lines.extend(p.strip() for p in raw.split("|") if p.strip())
        else:
            lines.append(raw)

    pieces: List[str] = []
    for ln in lines:
        pieces.extend(_split_line(ln, max_chars))

    chunks: List[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + 1 + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf:
        chunks.append(buf)
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars and len(chunks[-2]) + 1 + len(chunks[-1]) <= max_chars * 1.25:
        chunks[-2] = chunks[-2] + "\n" + chunks[-1]
        chunks.pop()
    # drop chunks with no letters or digits at all (e.g. a table rule)
    return [c for c in chunks if len(re.findall(r"[A-Za-z0-9]", c)) >= 2]
