#!/usr/bin/env python3
"""01_download_data.py : downloads the datasets used by the experiments into data_raw/

Dataset                          Source                                        Used by
  BIPIA (email, table)            GitHub microsoft/BIPIA (raw files)            H1, H2, H3
  deepset/prompt-injections       Hugging Face datasets                         every experiment (direct attacks + benign text)
  Lakera/gandalf_ignore_instructions  Hugging Face datasets                      H1, H3, Secondary (direct attack)
  leolee99/NotInject              Hugging Face datasets                         secondary (benign text with trigger words)
  (TaskTracker is built separately -- see scripts/05_build_tasktracker.py)

Security: the downloaded files are untrusted data -> this script writes them out as plain JSON/JSONL and never
uses trust_remote_code for any dataset (the experiment code reads them with json only)

How to use
  python scripts/01_download_data.py                     # download everything
  python scripts/01_download_data.py --only bipia        # only some of them (bipia deepset gandalf notinject)
  python scripts/01_download_data.py --force             # download again, overwriting
  python scripts/01_download_data.py --with-models       # also fetch the Sentence-BERT model up front (~90 MB)
  python scripts/01_download_data.py --models-only       # only the model, no datasets
  python scripts/01_download_data.py --with-promptguard  # try to fetch Prompt-Guard (needs access + an HF login)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")              # override with --raw-dir (keep data.raw_dir in config.yaml in sync)

BIPIA_BASE = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/"
BIPIA_FILES = [                          # (path on GitHub, destination under data_raw/bipia/benchmark/)
    "text_attack_train.json",
    "text_attack_test.json",
    "email/train.jsonl",
    "email/test.jsonl",
    "table/train.jsonl",
    "table/test.jsonl",
]
HF_DATASETS = {                          # short name -> (repo id, destination folder, expected splits, minimum total rows)
    "deepset": ("deepset/prompt-injections", "deepset_prompt_injections", ["train", "test"], 600),
    "gandalf": ("Lakera/gandalf_ignore_instructions", "gandalf_ignore_instructions", ["train", "validation", "test"], 900),
    "notinject": ("leolee99/NotInject", "notinject", ["NotInject_one", "NotInject_two", "NotInject_three"], 300),
}
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ------------------------------------------------------------------ BIPIA
def download_bipia(force: bool, info: dict) -> bool:
    say("\n[BIPIA] fetching the raw files from GitHub (the bipia package is not needed)")
    ok = True
    base = os.path.join(RAW, "bipia", "benchmark")
    for rel in BIPIA_FILES:
        dst = os.path.join(base, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 100 and not force:
            say(f"  already there  {rel}  ({os.path.getsize(dst):,} bytes)  -> skipped (use --force to re-download)")
        else:
            url = BIPIA_BASE + rel
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pi-detect-experiments"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 100:
                    raise RuntimeError(f"file is suspiciously small ({len(data)} bytes)")
                with open(dst, "wb") as f:
                    f.write(data)
                say(f"  downloaded {rel}  ({len(data):,} bytes)")
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as e:
                say(f"  !! could not download {rel}: {e}\n     open {url} in a browser and save it as {dst}")
                ok = False
                continue
        # check the contents
        try:
            if rel.endswith(".jsonl"):
                n = count_lines(dst)
                with open(dst, "r", encoding="utf-8") as f:
                    first = json.loads(f.readline())
                missing = [k for k in ("context", "question") if k not in first]
                if missing:
                    say(f"  !! {rel}: missing key(s) {missing} (the BIPIA file format may have changed)")
                    ok = False
                info.setdefault("bipia", {})[rel] = dict(rows=n, sha256=sha256(dst))
            else:
                with open(dst, "r", encoding="utf-8") as f:
                    d = json.load(f)
                info.setdefault("bipia", {})[rel] = dict(categories=len(d), sha256=sha256(dst))
        except Exception as e:  # noqa: BLE001
            say(f"  !! could not read {rel}: {e}")
            ok = False
    return ok


# ------------------------------------------------------------------ Hugging Face datasets
def download_hf(key: str, force: bool, info: dict) -> bool:
    repo, folder, splits, min_total = HF_DATASETS[key]
    dst_dir = os.path.join(RAW, folder)
    os.makedirs(dst_dir, exist_ok=True)
    say(f"\n[{key}] {repo}")
    have = all(os.path.exists(os.path.join(dst_dir, s + ".jsonl")) for s in splits)
    if have and not force:
        n = {s: count_lines(os.path.join(dst_dir, s + ".jsonl")) for s in splits}
        say(f"  already there {n} -> skipped (use --force to re-download)")
        info[key] = dict(repo=repo, rows=n)
        return sum(n.values()) >= min_total
    try:
        from datasets import load_dataset
    except ImportError:
        say("  !! the datasets package is not installed  -> pip install -r requirements.txt")
        return False
    try:
        ds = load_dataset(repo)               # never with trust_remote_code
    except Exception as e:  # noqa: BLE001
        say(f"  !! download failed: {type(e).__name__}: {str(e)[:300]}\n"
            f"     check the internet, VPN or proxy and try again (see Troubleshooting in README.md)")
        return False
    counts = {}
    for split in ds.keys():
        path = os.path.join(dst_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
        counts[split] = len(ds[split])
    say(f"  files written: {counts}")
    info[key] = dict(repo=repo, rows=counts)
    ok = True
    missing = [s for s in splits if s not in counts]
    if missing:
        say(f"  !! expected split(s) not found: {missing} (the dataset layout on Hugging Face may have changed)")
        ok = False
    if sum(counts.values()) < min_total:
        say(f"  !! only {sum(counts.values())} rows in total, fewer than expected (>= {min_total}); please check before using")
        ok = False
    return ok


def check_columns(info: dict) -> bool:
    """Checks the columns the experiment code needs"""
    need = {"deepset_prompt_injections": ("train.jsonl", ["text", "label"]),
            "gandalf_ignore_instructions": ("train.jsonl", ["text"]),
            "notinject": ("NotInject_one.jsonl", ["prompt"])}
    ok = True
    for folder, (fn, cols) in need.items():
        p = os.path.join(RAW, folder, fn)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        miss = [c for c in cols if c not in first]
        if miss:
            say(f"  !! {folder}/{fn}: missing column(s) {miss} (present: {list(first)})")
            ok = False
    return ok


# ------------------------------------------------------------------ models
def download_sbert() -> bool:
    say(f"\n[model] {SBERT_ID} (the Sentence-BERT used in Stages 2-4)")
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(SBERT_ID)
        v = m.encode(["hello world"], normalize_embeddings=True)
        say(f"  ready: embedding dimension {v.shape[1]}")
        return v.shape[1] == 384
    except ImportError:
        say("  !! sentence-transformers is not installed  -> pip install -r requirements.txt")
    except Exception as e:  # noqa: BLE001
        say(f"  !! failed: {type(e).__name__}: {str(e)[:300]}")
    return False


def download_promptguard(model_id: str) -> bool:
    say(f"\n[model] {model_id} (baseline B3a; a gated model)")
    say("  first: (1) create a huggingface.co account  (2) open the model page and accept its license  (3) create a Read access token\n"
        "         then run  python -c \"from huggingface_hub import login; login()\"  or set HF_TOKEN=... before this script")
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(model_id)
        say(f"  downloaded to {p}")
        return True
    except ImportError:
        say("  !! huggingface_hub is not installed  -> pip install -r requirements.txt")
    except Exception as e:  # noqa: BLE001
        say(f"  !! failed ({type(e).__name__}): {str(e)[:300]}\n"
            "     a 401/403 means no access or not logged in  ->  the experiments skip baseline B3a automatically")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["bipia", "deepset", "gandalf", "notinject"], default=None,
                    help="download only some of them")
    ap.add_argument("--force", action="store_true", help="download again, overwriting")
    ap.add_argument("--with-models", action="store_true", help="also fetch Sentence-BERT up front")
    ap.add_argument("--models-only", action="store_true", help="fetch only the Sentence-BERT model, no datasets")
    ap.add_argument("--with-promptguard", action="store_true", help="try to fetch Prompt-Guard (gated)")
    ap.add_argument("--promptguard-id", default="meta-llama/Prompt-Guard-86M")
    ap.add_argument("--raw-dir", default=None, help="destination folder (default: data_raw/ in the project folder)")
    args = ap.parse_args()
    global RAW
    if args.raw_dir:
        RAW = os.path.abspath(args.raw_dir)

    todo = [] if args.models_only else (args.only or ["bipia", "deepset", "gandalf", "notinject"])
    if args.models_only:
        args.with_models = True
    os.makedirs(RAW, exist_ok=True)
    info_path = os.path.join(RAW, "DOWNLOAD_INFO.json")
    info = {}
    if os.path.exists(info_path):                     # continue from a previous run (a models-only run must not erase the dataset record)
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            info = {}
    info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results = {}
    t0 = time.time()
    if "bipia" in todo:
        results["bipia"] = download_bipia(args.force, info)
    for k in ("deepset", "gandalf", "notinject"):
        if k in todo:
            results[k] = download_hf(k, args.force, info)
    if not check_columns(info):
        results["columns"] = False
    if args.with_models:
        results["sbert"] = download_sbert()
    if args.with_promptguard:
        results["promptguard"] = download_promptguard(args.promptguard_id)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    say("\n================ summary ================")
    for k, v in results.items():
        say(f"  {'OK  ' if v else 'FAIL'}  {k}")
    say(f"took {time.time() - t0:.0f} s | data is in {RAW}")
    bad = [k for k, v in results.items() if not v and k != "promptguard"]
    if bad:
        say("\nsomething failed -> read the '!!' lines above and see Troubleshooting in README.md "
            "(re-running is safe: files already downloaded are skipped)")
        return 1
    say("\nTaskTracker is built separately:\n"
        "  python scripts/05_build_tasktracker.py   (see section 6 of README.md)\n"
        "  until then you can set data.use_tasktracker: false in config.yaml")
    say("\nNext:  python scripts/00_check_env.py   (checks whether everything is ready)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
