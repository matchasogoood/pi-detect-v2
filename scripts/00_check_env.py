#!/usr/bin/env python3
"""00_check_env.py : checks whether this machine is ready to run the experiments (safe to run first, changes nothing)

Six checks
  1) the Python version        2) the required packages      3) the machine (GPU/RAM/disk)
  4) internet access (GitHub / Hugging Face)                 5) the data files in data_raw/ and data_local/
  6) the Sentence-BERT model (if already downloaded) and Prompt-Guard access (optional)

Output: OK / WARN / FAIL for each item, with how to fix it; exit code 0 = ready, 1 = something still FAILs

How to use:   python scripts/00_check_env.py
              python scripts/00_check_env.py --offline     # skip the internet checks
"""
import argparse
import glob
import importlib
import json
import os
import platform
import shutil
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = [   # (import name, pip name, required?)
    ("numpy", "numpy", True), ("pandas", "pandas", True), ("scipy", "scipy", True),
    ("sklearn", "scikit-learn", True), ("matplotlib", "matplotlib", True), ("yaml", "pyyaml", True),
    ("tqdm", "tqdm", False),
    ("torch", "torch", True), ("transformers", "transformers", True),
    ("sentence_transformers", "sentence-transformers", True),
    ("datasets", "datasets", True), ("huggingface_hub", "huggingface_hub", True),
    ("pytest", "pytest", False),
    ("lingua", "lingua-language-detector", True),
]
RESULTS = []          # (level, message)


def rec(level: str, msg: str) -> None:
    RESULTS.append((level, msg))
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[level]
    print(f"{tag} {msg}", flush=True)


def check_python() -> None:
    v = sys.version_info
    s = f"Python {v.major}.{v.minor}.{v.micro} ({platform.python_implementation()}) on {platform.system()}"
    if v < (3, 9):
        rec("FAIL", f"{s} is too old -> Python 3.9 or newer is required (3.12 or 3.13 recommended)")
    elif v < (3, 12):
        rec("WARN", f"{s} -> lingua-language-detector 2.2.0 (the pinned language filter) needs Python 3.12+; "
                    "on 3.11 install lingua-language-detector==2.1.1 instead")
    elif v >= (3, 14):
        rec("WARN", f"{s} -> very new; some packages (torch and friends) may not support it yet")
    else:
        rec("OK", s)
    in_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix)) or bool(os.environ.get("VIRTUAL_ENV")) \
        or bool(os.environ.get("CONDA_PREFIX"))
    if in_venv:
        rec("OK", "running inside a virtual environment (the project's packages are kept separate)")
    else:
        rec("WARN", "not running inside a virtual environment -> see section 3 of README.md (optional)")


def check_packages() -> None:
    missing = []
    for mod, pipname, required in PACKAGES:
        try:
            m = importlib.import_module(mod)
            rec("OK", f"package {pipname} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            if required:
                missing.append(pipname)
                rec("FAIL", f"package {pipname} not found ({type(e).__name__})")
            else:
                rec("WARN", f"package {pipname} not found (optional, but needed to run the test suite)")
    if missing:
        rec("FAIL", "fix with:  pip install -r requirements.txt   (for torch, see section 3 of README.md)")


def _ram_gb():
    """Total RAM in GB on Windows, macOS and Linux (None if it cannot be read)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("p", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("v", ctypes.c_ulonglong), ("w", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]
            ms = MS(); ms.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.total / 1e9
        if sys.platform == "darwin":
            import subprocess
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1e9
        with open("/proc/meminfo") as f:
            return int(next(l for l in f if l.startswith("MemTotal")).split()[1]) / 1e6
    except Exception:  # noqa: BLE001
        return None


DEVICE_IN_USE = "cpu"


def check_machine() -> None:
    """Reports what the machine has and which device the project will use (automatically: CUDA > Apple GPU > CPU)."""
    global DEVICE_IN_USE
    sys.path.insert(0, ROOT)
    rec("OK", f"{platform.system()} {platform.release()} | {platform.machine()} | {os.cpu_count()} CPU threads")
    try:
        from pidetect.device import available_devices, best_device
        d = available_devices()
        DEVICE_IN_USE = best_device()
        o = d["os"]
        if o in ("windows", "linux"):
            if d["cuda"]:
                rec("OK", f"{o.capitalize()}: NVIDIA GPU found and usable: {d['cuda']}")
            elif d["hardware"]:
                rec("WARN", f"{o.capitalize()}: an NVIDIA GPU is installed ({d['hardware']}) but PyTorch cannot use it "
                            "(probably the CPU-only PyTorch build, or an old driver) -> install the CUDA build from "
                            "https://pytorch.org/get-started/locally/ (pick your OS and CUDA version), then run this check again")
            else:
                rec("WARN", f"{o.capitalize()}: no NVIDIA GPU found -> the CPU will be used (works, but slower; not measured on CPU)")
        elif o == "macos":
            if d["mps"]:
                rec("OK", f"macOS: Apple GPU (MPS) found and usable ({d['hardware'] or 'Apple silicon'})")
            elif d["hardware"]:
                rec("WARN", f"macOS: Apple silicon ({d['hardware']}) found but PyTorch cannot use MPS -> "
                            "update PyTorch (pip install -U torch) and macOS, then run this check again")
            else:
                rec("WARN", "macOS: no Apple-silicon GPU (Intel Mac) -> the CPU will be used (works, but slower; not measured on CPU)")
        else:
            rec("WARN", "unrecognised operating system -> the CPU will be used")
        forced = os.environ.get("PI_DEVICE")
        rec("OK", f"device the project will use: {DEVICE_IN_USE.upper()}  "
                  f"({'forced by PI_DEVICE' if forced else 'chosen automatically from the system above, nothing to configure'}; "
                  "to force another one set PI_DEVICE=cpu, cuda or mps)")
    except Exception as e:  # noqa: BLE001
        rec("WARN", f"could not inspect the compute device ({type(e).__name__}); the CPU will be used")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    rec("OK" if free_gb >= 5 else "WARN", f"free disk space {free_gb:.1f} GB (about 5 GB for torch + models + results; "
                                           "TaskTracker needs about 2 GB more)")
    gb = _ram_gb()
    if gb is not None:
        rec("OK" if gb >= 8 else "WARN", f"RAM {gb:.1f} GB (8 GB or more recommended)")


def check_network() -> None:
    for name, url, need in (("GitHub (BIPIA)", "https://raw.githubusercontent.com/microsoft/BIPIA/main/README.md", True),
                            ("Hugging Face (datasets/models)", "https://huggingface.co", True)):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pi-detect-experiments"})
            with urllib.request.urlopen(req, timeout=10) as r:
                rec("OK", f"{name} is reachable (HTTP {r.status})")
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code in (301, 302, 307, 308, 405):      # the server answered, so it is reachable
                rec("OK", f"{name} is reachable (HTTP {code})")
            else:
                rec("FAIL" if need else "WARN",
                    f"{name} is unreachable ({type(e).__name__}: {str(e)[:80]}) -> check the internet, VPN or proxy, "
                    "or download the files on another machine and copy them into data_raw/ (see Troubleshooting in README.md)")


def count_lines(p: str) -> int:
    with open(p, "r", encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def check_data() -> None:
    raw = os.path.join(ROOT, "data_raw")
    loc = os.path.join(ROOT, "data_local")
    spec = [("BIPIA email train", "bipia/benchmark/email/train.jsonl"), ("BIPIA email test", "bipia/benchmark/email/test.jsonl"),
            ("BIPIA table train", "bipia/benchmark/table/train.jsonl"), ("BIPIA table test", "bipia/benchmark/table/test.jsonl"),
            ("BIPIA attacks train", "bipia/benchmark/text_attack_train.json"),
            ("BIPIA attacks test", "bipia/benchmark/text_attack_test.json"),
            ("deepset train", "deepset_prompt_injections/train.jsonl"), ("deepset test", "deepset_prompt_injections/test.jsonl"),
            ("Gandalf train", "gandalf_ignore_instructions/train.jsonl")]
    missing = 0
    for label, rel in spec:
        p = os.path.join(raw, *rel.split("/"))
        if os.path.exists(p):
            n = count_lines(p) if p.endswith(".jsonl") else len(json.load(open(p, encoding="utf-8")))
            rec("OK", f"data_raw/{rel}  ({n:,} {'rows' if p.endswith('.jsonl') else 'categories'})")
        else:
            missing += 1
            rec("FAIL", f"data_raw/{rel} not found  ({label})")
    ni = glob.glob(os.path.join(raw, "notinject", "*.jsonl"))
    if ni:
        rec("OK", f"data_raw/notinject/ has {len(ni)} file(s) ({sum(count_lines(p) for p in ni):,} rows)")
    else:
        missing += 1
        rec("FAIL", "data_raw/notinject/*.jsonl not found")
    if missing:
        rec("FAIL", "fix with:  python scripts/01_download_data.py")
    kb = sorted(glob.glob(os.path.join(loc, "mock_kb*.json")))
    n_clusters = 0
    bad = []
    for p in kb:
        try:
            for c in json.load(open(p, encoding="utf-8")):
                n_clusters += 1
                for k in ("id", "instructions", "docs", "bridge", "attacks"):
                    if k not in c:
                        bad.append(f"{os.path.basename(p)}:{c.get('id', '?')} is missing '{k}'")
                if isinstance(c.get("docs"), list) and len(c["docs"]) < 6:
                    bad.append(f"{os.path.basename(p)}:{c.get('id', '?')} has only {len(c['docs'])} docs (at least 6 are needed)")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{os.path.basename(p)}: could not be read ({e})")
    if bad:
        rec("FAIL", "the mock KB has the wrong format: " + "; ".join(bad[:5]))
    elif n_clusters >= 5:
        rec("OK", f"data_local/mock_kb*.json has {n_clusters} clusters")
    else:
        rec("FAIL", f"the mock KB has only {n_clusters} clusters (at least 5 are needed to split val/test)")
    p = os.path.join(loc, "security_texts.json")
    rec("OK" if os.path.exists(p) else "FAIL", "data_local/security_texts.json " + ("found" if os.path.exists(p) else "missing"))
    tt = glob.glob(os.path.join(raw, "tasktracker", "*.jsonl"))
    if tt:
        rec("OK", f"data_raw/tasktracker/ has {len(tt)} file(s) ({sum(count_lines(p) for p in tt):,} rows)")
    else:
        rec("WARN", "TaskTracker not built yet (data_raw/tasktracker/*.jsonl) -> build it with python scripts/05_build_tasktracker.py "
                    "or set data.use_tasktracker: false in config.yaml (see section 6 of README.md)")


def check_models(offline: bool = False) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        try:
            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            rec("OK", "the Sentence-BERT model (all-MiniLM-L6-v2) is already on this machine")
        except Exception:  # noqa: BLE001
            rec("WARN", "the Sentence-BERT model is not downloaded yet -> it downloads automatically on the first run (~90 MB), "
                        "or fetch it now with:  python scripts/01_download_data.py --models-only")
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.path.insert(0, os.path.join(ROOT, "demo"))
        import io, contextlib, weights_pack
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = weights_pack.check()
        if rc == 0:
            rec("OK", "trained FT-C weights shipped in demo/weights are intact (the demo will not need to fine-tune)")
        else:
            rec("WARN", "the shipped FT-C weights are missing or damaged -> run  python demo/weights_pack.py check  for the fix "
                        "(without them the demo fine-tunes its own FT-C, about 70 s on a GPU)")
    except Exception:  # noqa: BLE001
        pass
    check_hf_token(offline)


def check_hf_token(offline: bool) -> None:
    """Token needed only for the gated Prompt-Guard baseline. Never prints the token."""
    tok = None
    try:
        from huggingface_hub import get_token      # env HF_TOKEN or the login file, on Windows, macOS and Linux
        tok = get_token()
    except Exception:  # noqa: BLE001
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    steps = ("get one in 4 steps (README, 'Hugging Face token'): 1) free account at huggingface.co, "
             "2) accept the licence at huggingface.co/meta-llama/Prompt-Guard-86M, "
             "3) create a Read token at huggingface.co/settings/tokens, 4) run  huggingface-cli login  (or  hf auth login)")
    if not tok:
        rec("WARN", "no Hugging Face token -> only the Prompt-Guard baseline (B3a) is skipped automatically; "
                    "the proposed system, the demo and the other baselines do not need one. To enable it: " + steps)
        return
    if offline:
        rec("OK", "a Hugging Face token is set (not tested because of --offline)")
        return
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=tok).get("name", "?")
        rec("OK", f"Hugging Face token is valid (account: {who})")
    except Exception as e:  # noqa: BLE001
        rec("WARN", f"a Hugging Face token is set but was rejected ({type(e).__name__}) -> create a new Read token. " + steps)
        return
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("meta-llama/Prompt-Guard-86M", "config.json", token=tok)
        rec("OK", "access to meta-llama/Prompt-Guard-86M is granted (baseline B3a will run)")
    except Exception as e:  # noqa: BLE001
        rec("WARN", f"no access to meta-llama/Prompt-Guard-86M yet ({type(e).__name__}) -> open "
                    "huggingface.co/meta-llama/Prompt-Guard-86M while logged in, accept the licence, wait for approval; "
                    "until then B3a is skipped automatically")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip the internet checks")
    args = ap.parse_args()
    print(f"project folder: {ROOT}\n")
    print("== 1) Python =="); check_python()
    print("\n== 2) packages =="); check_packages()
    print("\n== 3) machine =="); check_machine()
    if not args.offline:
        print("\n== 4) internet =="); check_network()
    print("\n== 5) data =="); check_data()
    print("\n== 6) models and Hugging Face token =="); check_models(args.offline)
    n_fail = sum(1 for lv, _ in RESULTS if lv == "FAIL")
    n_warn = sum(1 for lv, _ in RESULTS if lv == "WARN")
    print(f"\n================ summary: FAIL {n_fail} | WARN {n_warn} ================")
    if n_fail:
        print("not ready yet -> fix every [FAIL] using the hint on its line, then run this script again")
        print("(to check only that the code runs, without any data, try the test mode: "
              "python scripts/run_all.py --smoke --data synthetic --embedder hashing)")
        return 1
    print(f"ready! the project will run on: {DEVICE_IN_USE.upper()} (automatic)")
    print("next:  python scripts/run_all.py --smoke      (quick test)   |   python demo/server.py      (interactive demo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
