#!/usr/bin/env python3
"""04_language_report.py : counts which languages each dataset contains, how many texts of each, and how many the English filter drops

Why: the paper (Section IV-A, the "Language composition" table) reports these counts from the real data, and the
experiments are English-only (config.yaml: data.english_only), so we need to know exactly what is dropped.

Language identifier: lingua (the most accurate on short texts); otherwise langdetect; otherwise the rules in pidetect/data/lang.py
  install:  pip install lingua-language-detector     (already in requirements.txt)
Texts with no letters at all (only digits or symbols) are counted as "und" (undetermined).

How to use (after running scripts/01_download_data.py)
  python scripts/04_language_report.py                  # every dataset present on disk
  python scripts/04_language_report.py --raw-dir path/to/data_raw

Output (in results/final/language_report/)
  language_counts.csv      a dataset x language table (number of texts)
  filter_check.csv         what the code's filter (is_english) kept and dropped, against the identifier -- shows whether it is right
  disagreements.csv        texts where the filter and the identifier disagree (mostly short English texts the identifier misreads)
  non_english_examples.txt sample non-English texts (to read by eye)
  language_table.tex       the table rows to paste into the paper (the Language composition table)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pidetect.data import loaders  # noqa: E402
from pidetect.data.lang import is_english  # noqa: E402

try:
    from pidetect.attackbank import HAND_ATTACKS  # noqa: E402
except Exception:  # noqa: BLE001
    HAND_ATTACKS = []


# ------------------------------------------------------------------ language identifier
def make_detector():
    try:
        from lingua import LanguageDetectorBuilder
        det = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

        def f(t: str) -> str:
            lang = det.detect_language_of(t)
            return lang.iso_code_639_1.name.lower() if lang is not None else "und"
        return f, "lingua"
    except ImportError:
        pass
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0

        def f(t: str) -> str:
            try:
                return detect(t)
            except Exception:  # noqa: BLE001
                return "und"
        return f, "langdetect"
    except ImportError:
        pass
    return (lambda t: "en" if is_english(t) else "non-en"), "heuristic (pidetect/data/lang.py)"


LANG_NAMES = dict(en="English", de="German", fr="French", es="Spanish", it="Italian", pt="Portuguese",
                  zh="Chinese", ja="Japanese", ko="Korean", ru="Russian", ar="Arabic", nl="Dutch", th="Thai",
                  und="undetermined")


# ------------------------------------------------------------------ the text units the experiments actually use
def units(raw: str, local: str):
    """yield (dataset name, subset, text) -- matching what pidetect/data/world.py filters"""
    have = []
    try:
        b = loaders.load_bipia(raw, ("email", "table"))
        for sp, rows in b["contexts"].items():
            for r in rows:      # tables: judged by the question (cells hold foreign proper names) -- as in world.keep_english
                text = r["question"] if r["task"] == "table" else r["context"] + "\n" + r["question"]
                yield "BIPIA contexts", f"{r['task']}/{sp}", text
        for sp, cats in b["attacks"].items():
            for c, a in cats.items():
                for t in a:
                    yield "BIPIA attacks", sp, t
        have.append("bipia")
    except FileNotFoundError:
        print("  (skipping BIPIA: files not found)")
    try:
        for sp, rows in loaders.load_deepset(raw).items():
            for r in rows:
                yield "deepset", f"{sp}/label{int(r['label'])}", str(r["text"])
        have.append("deepset")
    except FileNotFoundError:
        print("  (skipping deepset: files not found)")
    try:
        g = loaders.load_gandalf(raw)
        for sp, rows in g.items():
            for t in rows:
                yield "Gandalf", sp, t
        have.append("gandalf")
    except FileNotFoundError:
        print("  (skipping Gandalf: files not found)")
    try:
        for r in loaders.load_notinject(raw):
            yield "NotInject", r["category"] or "all", r["prompt"]
        have.append("notinject")
    except FileNotFoundError:
        print("  (skipping NotInject: files not found)")
    try:
        for sp, rows in loaders.load_tasktracker(raw, max_rows=10 ** 9).items():
            for r in rows:
                yield "TaskTracker", f"{sp}/label{r['label']}", r["user_prompt"]   # the question only (what the filter would see)
        have.append("tasktracker")
    except FileNotFoundError:
        print("  (skipping TaskTracker: files not found -- build them with scripts/05_build_tasktracker.py)")
    try:
        loc = loaders.load_local(local)
        for cl in loc["kb"]:
            for key in ("instructions", "docs", "bridge", "attacks"):
                for t in cl.get(key, []):
                    yield "Mock KB", key, t
        for key, v in loc["security"].items():
            for t in (v if isinstance(v, list) else [v]):
                yield "Security texts", key, t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
    except FileNotFoundError:
        print("  (skipping our own data: data_local/ not found)")
    for t in HAND_ATTACKS:
        yield "Hand-written attacks", "bank", t


def has_letters(t: str) -> bool:
    return any(ch.isalpha() for ch in t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data_raw"))
    ap.add_argument("--local-dir", default=os.path.join(ROOT, "data_local"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "final", "language_report"))
    ap.add_argument("--max-examples", type=int, default=15, help="sample non-English texts to show per dataset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    detect, name = make_detector()
    print(f"language identifier: {name}")
    counts = collections.defaultdict(collections.Counter)          # dataset -> lang -> n
    sub_counts = collections.defaultdict(collections.Counter)      # (dataset, part) -> lang -> n
    filt = collections.defaultdict(collections.Counter)            # dataset -> (detected_en?, kept?) -> n
    removed_langs = collections.defaultdict(collections.Counter)   # dataset -> languages of the texts the filter dropped
    disagree = []                                                  # filter and identifier disagree (to read by eye)
    examples = collections.defaultdict(list)
    for ds, part, text in units(args.raw_dir, args.local_dir):
        lang = detect(text) if has_letters(text) else "und"
        counts[ds][lang] += 1
        sub_counts[(ds, part)][lang] += 1
        kept = True if ds == "TaskTracker" else is_english(text)   # TaskTracker is English by construction: never dropped (world.keep_english)
        filt[ds][(lang in ("en", "und"), kept)] += 1
        if not kept:
            removed_langs[ds][lang] += 1
        if kept != (lang in ("en", "und")):
            disagree.append((ds, part, lang, "KEPT" if kept else "REMOVED", text.replace("\n", " ")[:200]))
        if lang not in ("en", "und") and len(examples[ds]) < args.max_examples:
            examples[ds].append((lang, kept, text.replace("\n", " ")[:160]))
    if not counts:
        print("no data found -> run python scripts/01_download_data.py first")
        return 1

    langs = sorted({lg for c in counts.values() for lg in c}, key=lambda lg: (lg != "en", lg == "und", lg))
    with open(os.path.join(args.out, "language_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "part", "total"] + langs)
        for ds, c in counts.items():
            w.writerow([ds, "ALL", sum(c.values())] + [c.get(lg, 0) for lg in langs])
            for (d2, part), c2 in sorted(sub_counts.items()):
                if d2 == ds:
                    w.writerow([ds, part, sum(c2.values())] + [c2.get(lg, 0) for lg in langs])
    with open(os.path.join(args.out, "filter_check.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "detector_en_kept", "detector_en_REMOVED", "detector_nonEN_REMOVED",
                    "detector_nonEN_KEPT"])
        for ds, c in filt.items():
            w.writerow([ds, c[(True, True)], c[(True, False)], c[(False, False)], c[(False, True)]])
    with open(os.path.join(args.out, "non_english_examples.txt"), "w", encoding="utf-8") as f:
        for ds, ex in examples.items():
            f.write(f"== {ds}\n")
            for lang, kept, t in ex:
                f.write(f"  [{lang}{' | KEPT by filter' if kept else ''}] {t}\n")

    with open(os.path.join(args.out, "disagreements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "part", "detector_lang", "filter", "text"])
        w.writerows(disagree)

    # Rows for the paper (the Language composition table): dataset & total & English (kept) & removed (languages)
    # "kept" = passed the code's filter. The languages of the dropped texts come from the identifier (reliable on long
    # sentences); check disagreements.csv for short texts it misreads before putting the numbers in the paper.
    tex = []
    for ds, c in counts.items():
        tot = sum(c.values())
        kept = filt[ds][(True, True)] + filt[ds][(False, True)]
        rl = removed_langs[ds]
        langs_txt = ", ".join(f"{LANG_NAMES.get(lg, lg)} {n}" for lg, n in rl.most_common())
        tex.append(f"{ds} & {tot} & {kept} & {tot - kept}" + (f" ({langs_txt})" if langs_txt else "") + " \\\\")
    with open(os.path.join(args.out, "language_table.tex"), "w", encoding="utf-8") as f:
        f.write(f"% dataset & total & English (kept) & removed (languages by {name})\n")
        f.write("\n".join(tex) + "\n")

    print("\nLanguages found (number of texts)")
    for ds, c in counts.items():
        tot = sum(c.values())
        print(f"  {ds:22s} total {tot:6d} | " + ", ".join(f"{LANG_NAMES.get(lg, lg)} {n}" for lg, n in c.most_common()))
    print("\nis_english() against the language identifier")
    for ds, c in filt.items():
        bad = c[(True, False)] + c[(False, True)]
        print(f"  {ds:22s} kept {c[(True, True)] + c[(False, True)]:6d} dropped {c[(True, False)] + c[(False, False)]:5d}"
              + (f"  ({bad} disagree with the identifier -> see filter_check.csv)" if bad else ""))
    print(f"\nresults written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())