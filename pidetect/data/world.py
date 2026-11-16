"""world.py : collects every raw data source in one place (World), which sessions.py then splits into sessions"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from . import loaders
from .lang import is_english as _is_english
from .synthetic import make_synthetic_externals, make_synthetic_kb


@dataclass
class World:
    source: str                                   # "real" or "synthetic"
    bipia_contexts: Dict[str, List[dict]]
    bipia_attacks: Dict[str, Dict[str, List[str]]]
    deepset: Dict[str, List[dict]]
    gandalf: Dict[str, List[str]]
    notinject: List[dict]
    kb: List[dict]
    security: Dict
    notes: List[str] = field(default_factory=list)
    tasktracker: Dict[str, List[dict]] = field(default_factory=dict)
    language_filter: Dict[str, Dict[str, int]] = field(default_factory=dict)   # source -> {before, removed, kept}

    def summary(self) -> str:
        n_att = sum(1 for r in self.deepset.get("train", []) + self.deepset.get("test", []) if r["label"] == 1)
        return (f"World[{self.source}] BIPIA ctx train/test={len(self.bipia_contexts['train'])}/"
                f"{len(self.bipia_contexts['test'])}, BIPIA attack cats train/test={len(self.bipia_attacks['train'])}/"
                f"{len(self.bipia_attacks['test'])}, deepset={sum(len(v) for v in self.deepset.values())} "
                f"(attacks {n_att}), gandalf={sum(len(v) for v in self.gandalf.values())}, "
                f"notinject={len(self.notinject)}, KB clusters={len(self.kb)}, "
                f"tasktracker={sum(len(v) for v in self.tasktracker.values())}" +
                (f" | {'; '.join(self.notes)}" if self.notes else ""))


def load_world(raw_dir: str, local_dir: str, source: str = "real", need: Sequence[str] = ("bipia", "deepset",
               "gandalf", "notinject"), tasks: Sequence[str] = ("email", "table"), seed: int = 0,
               tasktracker_max_rows: int = 4000, english_only: bool = True) -> World:
    """need = the datasets this experiment really uses (a missing one raises a clear error); the rest stay empty"""
    local = loaders.load_local(local_dir)
    notes: List[str] = []
    if source == "synthetic":
        ext = make_synthetic_externals(seed)
        notes.append("synthetic data: for testing the code only")
        w = World("synthetic", ext["bipia_contexts"], ext["bipia_attacks"], ext["deepset"], ext["gandalf"],
                  ext["notinject"], make_synthetic_kb(seed), local["security"], notes,
                  ext["tasktracker"] if "tasktracker" in need else {})
        return keep_english(w) if english_only else w

    empty_b = dict(contexts={"train": [], "test": []}, attacks={"train": {}, "test": {}}, tasks=[])
    bipia = loaders.load_bipia(raw_dir, tasks) if "bipia" in need else empty_b
    deepset = loaders.load_deepset(raw_dir) if "deepset" in need else {"train": [], "test": []}
    gandalf = loaders.load_gandalf(raw_dir) if "gandalf" in need else {"train": [], "validation": [], "test": []}
    notinject = loaders.load_notinject(raw_dir) if "notinject" in need else []
    tasktracker = loaders.load_tasktracker(raw_dir, tasktracker_max_rows, seed) if "tasktracker" in need else {}
    if bipia["tasks"]:
        notes.append(f"BIPIA tasks used: {bipia['tasks']}")
    tasktracker = drop_bipia_test_overlap(tasktracker, bipia["attacks"].get("test", {}), notes)
    w = World("real", bipia["contexts"], bipia["attacks"], deepset, gandalf, notinject,
              local["kb"], local["security"], notes, tasktracker)
    return keep_english(w) if english_only else w


def drop_bipia_test_overlap(tt: Dict[str, List[dict]], bipia_test: Dict[str, List[str]], notes: List[str]):
    """Leakage guard: the TaskTracker poisoned-validation notebook uses BIPIA attacks from both train and test,
    but we use BIPIA test as our test set -> drop the non-test TaskTracker rows whose paragraph contains a BIPIA
    test attack. (scripts/05_build_tasktracker.py already excludes them at build time; this is a second guard for
    files converted from the original notebooks.)"""
    attacks = [a for v in bipia_test.values() for a in v if a.strip()]
    if not tt or not attacks:
        return tt
    out, dropped = {}, 0
    for split, rows in tt.items():
        if split == "test":
            out[split] = rows
            continue
        keep = [r for r in rows if not (r["label"] == 1 and any(a in r["text"] for a in attacks))]
        dropped += len(rows) - len(keep)
        out[split] = keep
    if dropped:
        notes.append(f"TaskTracker: dropped {dropped} rows containing a BIPIA test attack (leakage guard)")
    return out


def keep_english(w: World) -> World:
    """Drops non-English text from every source (before the dev/val/test split) and records the counts in
    w.language_filter (which goes into run_info.json) and in w.notes

    Expected counts (verified with scripts/04_language_report.py):
      NotInject  85 of 339 dropped (all 84 of the Multilingual category: Chinese 81, Russian 2, Spanish 1, plus one Chinese Common Query)
      BIPIA      nothing dropped (the questions and emails are all English)
      deepset    308 of 662 dropped (German or German mixed with English 294, other languages 14), leaving 140 injections / 214 benign
      Gandalf    7 of 1,000 dropped (German 4, Spanish mixed with English 2, Russian 1)
      TaskTracker nothing dropped (English by construction, see below); counts are in the report of 04_language_report.py

    BIPIA tables (the table task) are judged by their question alone: the table is English but its cells hold foreign
    proper names (athletes, teams, venues), which mislead the identifier into dropping English tables.
    TaskTracker is never filtered: scripts/05_build_tasktracker.py builds it only from English sources (SQuAD, English
    HotPotQA, English prompt lists, no translated variants). Its Wikipedia paragraphs quote foreign names and glosses
    ("(German: ...)"), and judging them as language samples dropped 475 of 16,000 English rows (and 15 of 480 even when
    only the question was judged), so it is counted for confirmation only, like the mock knowledge base.
    """
    stats: Dict[str, Dict[str, int]] = {}
    real = w.source == "real"                      # synthetic data is made-up words -> layer 1 only (see lang.py)

    def is_english(t: str) -> bool:
        return _is_english(t, use_lingua=real)

    def note(name: str, before: int, kept: int) -> None:
        d = stats.setdefault(name, dict(before=0, kept=0, removed=0))
        d["before"] += before
        d["kept"] += kept
        d["removed"] += before - kept

    def flt(name, rows, key=None):
        keep = [r for r in rows if is_english(r[key] if key else r)]
        note(name, len(rows), len(keep))
        return keep

    def ctx_ok(c: dict) -> bool:
        return is_english(c["question"]) and (c.get("task") == "table" or is_english(c["context"]))

    w.deepset = {k: flt("deepset", v, "text") for k, v in w.deepset.items()}
    w.gandalf = {k: flt("gandalf", v) for k, v in w.gandalf.items()}
    w.notinject = flt("notinject", w.notinject, "prompt")
    new_ctx = {}
    for k, v in w.bipia_contexts.items():
        new_ctx[k] = [c for c in v if ctx_ok(c)]
        note("bipia_contexts", len(v), len(new_ctx[k]))
    w.bipia_contexts = new_ctx
    w.bipia_attacks = {sp: {c: flt("bipia_attacks", a) for c, a in cats.items()} for sp, cats in w.bipia_attacks.items()}
    w.bipia_attacks = {sp: {c: a for c, a in cats.items() if a} for sp, cats in w.bipia_attacks.items()}
    new_tt = {}
    for k, v in w.tasktracker.items():
        new_tt[k] = list(v)                                # English by construction (see docstring) -> counted, never dropped
        note("tasktracker (counted only)", len(v), sum(is_english(r["user_prompt"]) for r in v))
    w.tasktracker = new_tt
    for c in w.kb:                                     # our own data is all English -> counted for confirmation, never dropped
        texts = [t for key in ("instructions", "docs", "bridge", "attacks") for t in c.get(key, [])]
        note("mock_kb (counted only)", len(texts), sum(map(is_english, texts)))
    w.language_filter = {k: v for k, v in stats.items() if v["before"]}
    removed = {k: v["removed"] for k, v in w.language_filter.items() if v["removed"]}
    w.notes.append("English only: dropped " + (", ".join(f"{k} {v}" for k, v in removed.items()) or "nothing"))
    return w


def default_dirs(root: str | None = None) -> tuple[str, str]:
    """(raw_dir, local_dir) relative to the project root"""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data_raw"), os.path.join(root, "data_local")