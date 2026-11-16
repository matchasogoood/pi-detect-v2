"""loaders.py : reads the datasets from data_raw/ (downloaded by scripts/01_download_data.py)

Security note: the downloaded files are untrusted data, so they are read with json only.
Nothing is ever exec'd, eval'd or unpickled, and trust_remote_code is never used for a dataset.

Expected layout of data_raw/
  bipia/benchmark/email/{train,test}.jsonl        (keys: context, question, ideal)
  bipia/benchmark/table/{train,test}.jsonl
  bipia/benchmark/text_attack_{train,test}.json   ({category: [5 attack sentences]})
  deepset_prompt_injections/{train,test}.jsonl    (text, label)   label 1 = injection
  gandalf_ignore_instructions/{train,validation,test}.jsonl   (text, similarity)
  notinject/NotInject_{one,two,three}.jsonl       (prompt, word_list, category)   (every *.jsonl in the folder is read)
  tasktracker/*.jsonl                            (user_prompt, text, label)   label 1 = a paragraph with an injected instruction
                                                 built by scripts/05_build_tasktracker.py (see the README)
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Sequence


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _need(path: str, how: str = "python scripts/01_download_data.py") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}\nRun this first:  {how}")
    return path


def load_bipia(raw_dir: str, tasks: Sequence[str] = ("email", "table")) -> Dict:
    """BIPIA : contexts[split] = [{id, task, context, question, ideal}], attacks[split] = {category: [str]}

    The 'train' / 'test' splits are BIPIA's own. Their attack categories do not overlap (except 'Language
    Translation'), so we use train as validation and test as test.
    """
    base = os.path.join(raw_dir, "bipia", "benchmark")
    contexts: Dict[str, List[dict]] = {"train": [], "test": []}
    found = []
    for task in tasks:
        ok = False
        for split in ("train", "test"):
            p = os.path.join(base, task, f"{split}.jsonl")
            if not os.path.exists(p):
                continue
            ok = True
            for i, row in enumerate(read_jsonl(p)):
                if "context" not in row or "question" not in row:
                    continue          # e.g. the 'code' task has no question -> skip
                q = re.sub(r"^\s*Q:\s*", "", str(row["question"])).strip()
                ctx = row["context"]
                if isinstance(ctx, list):
                    ctx = "\n".join(map(str, ctx))
                contexts[split].append(dict(id=f"{task}-{split}-{i}", task=task, context=str(ctx),
                                            question=q, ideal=str(row.get("ideal", ""))))
        if ok:
            found.append(task)
    if not found:
        _need(os.path.join(base, "email", "train.jsonl"))
    attacks = {}
    for split in ("train", "test"):
        p = _need(os.path.join(base, f"text_attack_{split}.json"))
        with open(p, "r", encoding="utf-8") as f:
            attacks[split] = {k: [str(x) for x in v] for k, v in json.load(f).items()}
    return dict(contexts=contexts, attacks=attacks, tasks=found)


def load_deepset(raw_dir: str) -> Dict[str, List[dict]]:
    d = os.path.join(raw_dir, "deepset_prompt_injections")
    return {s: read_jsonl(_need(os.path.join(d, f"{s}.jsonl"))) for s in ("train", "test")}


def load_gandalf(raw_dir: str) -> Dict[str, List[str]]:
    d = os.path.join(raw_dir, "gandalf_ignore_instructions")
    out = {}
    for s in ("train", "validation", "test"):
        p = os.path.join(d, f"{s}.jsonl")
        out[s] = [str(r["text"]) for r in read_jsonl(_need(p))] if os.path.exists(p) else []
    return out


def load_notinject(raw_dir: str) -> List[dict]:
    """NotInject : benign prompts containing 1/2/3 trigger words (reads every *.jsonl in notinject/)
    n_words = 1/2/3 from the file name (one/two/three); if the name does not say, it is counted from word_list"""
    d = os.path.join(raw_dir, "notinject")
    words = {"one": 1, "two": 2, "three": 3}
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        base = os.path.basename(p).lower()
        n_file = next((v for k, v in words.items() if k in base), None)
        for r in read_jsonl(p):
            if "prompt" not in r:
                continue
            wl = r.get("word_list")
            n = n_file or (len(wl) if isinstance(wl, list) else 0)
            rows.append(dict(prompt=str(r["prompt"]), category=str(r.get("category", "")), n_words=n))
    if not rows:
        _need(os.path.join(d, "NotInject_one.jsonl"))
    return rows


def load_local(local_dir: str) -> Dict:
    """Our own data: the mock KB (mock_kb*.json) and security_texts.json"""
    kb: List[dict] = []
    for p in sorted(glob.glob(os.path.join(local_dir, "mock_kb*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            kb.extend(json.load(f))
    with open(_need(os.path.join(local_dir, "security_texts.json")), "r", encoding="utf-8") as f:
        sec = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return dict(kb=kb, security=sec)


def load_tasktracker(raw_dir: str, max_rows: int = 4000, seed: int = 0) -> Dict[str, List[dict]]:
    """TaskTracker (Abdelnabi et al.) : each row = a user task (user_prompt) + a paragraph (text) + a label

    Files named train / val / test keep that split; otherwise everything is merged into "all" (sessions.py splits it)
    Only max_rows rows per split are read (sampled with a fixed seed), because the full set has 500K+ rows
    """
    import random
    d = os.path.join(raw_dir, "tasktracker")
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    if not files:
        raise FileNotFoundError(
            f"no *.jsonl files in {d}\nBuild TaskTracker with:  python scripts/05_build_tasktracker.py\n"
            "(or convert files from the original notebooks with scripts/03_convert_tasktracker.py -- see the README)\n"
            "or turn it off in config.yaml (data.use_tasktracker: false)")
    out: Dict[str, List[dict]] = {}
    for p in files:
        base = os.path.basename(p).lower()
        split = next((k for k in ("train", "val", "test") if k in base), "all")
        rows = []
        for r in read_jsonl(p):
            up = r.get("user_prompt") or r.get("primary_task") or r.get("instruction")
            tx = r.get("text") or r.get("paragraph") or r.get("final_text")
            if up is None or tx is None or "label" not in r:
                continue
            rows.append(dict(user_prompt=str(up), text=str(tx), label=int(r["label"])))
        out.setdefault(split, []).extend(rows)
    rng = random.Random(seed)
    for k, v in out.items():
        if len(v) > max_rows:
            out[k] = rng.sample(v, max_rows)
    return out
