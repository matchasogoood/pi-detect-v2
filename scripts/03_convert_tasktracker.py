#!/usr/bin/env python3
"""03_convert_tasktracker.py : converts files produced by the TaskTracker notebooks (github.com/microsoft/TaskTracker)
into the format this code reads -> data_raw/tasktracker/<split>.jsonl  (columns user_prompt, text, label)

TaskTracker builds its data with its own notebooks (the full set has 500K+ rows) and the column names vary by version,
so this script guesses the common ones; if it cannot, pass --prompt-key / --text-key / --label-key yourself.

Recommended: use scripts/05_build_tasktracker.py instead (it builds everything in one command). This script is for
people who already ran the notebooks. Those files (dataset_out_*.json) have no label column, so the label is taken
from secondary_task_prompt (empty = clean, non-empty = poisoned).

Examples
  python scripts/03_convert_tasktracker.py dataset_out_clean_v2.json dataset_out_poisoned_v2.json --split test
  python scripts/03_convert_tasktracker.py dataset_out_clean.json dataset_out_poisoned_v1.json --split val
  python scripts/03_convert_tasktracker.py data.jsonl --split test --prompt-key primary_task --text-key final_text
  python scripts/03_convert_tasktracker.py clean.json --split val --force-label 0     # a file that is entirely clean
Security: read with json only (never eval, never pickle)
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_KEYS = ["user_prompt", "primary_task_prompt", "primary_task", "instruction", "question", "prompt"]
TEXT_KEYS = ["text", "final_text_paragraph", "final_text", "paragraph", "context", "data"]
LABEL_KEYS = ["label", "poisoned", "is_poisoned", "injected"]


def read_any(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def pick(row, keys, forced):
    if forced:
        return row.get(forced)
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--prompt-key")
    ap.add_argument("--text-key")
    ap.add_argument("--label-key")
    ap.add_argument("--force-label", type=int, choices=[0, 1], default=None, help="use this label for every row in the file")
    ap.add_argument("--max-rows", type=int, default=20000, help="keep at most this many rows (sampled)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows, skipped = [], 0
    for p in a.files:
        for r in read_any(p):
            up = pick(r, PROMPT_KEYS, a.prompt_key)
            tx = pick(r, TEXT_KEYS, a.text_key)
            lb = a.force_label if a.force_label is not None else pick(r, LABEL_KEYS, a.label_key)
            if lb is None and "secondary_task_prompt" in r:       # files from the TaskTracker notebooks have no label column
                lb = int(bool(str(r["secondary_task_prompt"] or "").strip()))
            if up is None or tx is None or lb is None:
                skipped += 1
                continue
            rows.append(dict(user_prompt=str(up), text=str(tx), label=int(bool(int(lb)))))
    if not rows:
        print(f"no usable rows ({skipped} skipped) -> name the columns with --prompt-key / --text-key / --label-key")
        return 1
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.max_rows]
    out_dir = os.path.join(ROOT, "data_raw", "tasktracker")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{a.split}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n1 = sum(r["label"] for r in rows)
        print(f"wrote {len(rows)} rows ({n1} poisoned, {len(rows) - n1} clean) -> {out}  ({skipped} incomplete rows skipped)")
    if n1 == 0 or n1 == len(rows):
        print("warning: only one label present -> merge the clean and poisoned files of the same split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
