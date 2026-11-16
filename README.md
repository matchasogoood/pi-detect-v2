# Behavior-based detection of prompt injection in LLM applications

Code, experiments and an interactive demo for the CSS453 (Cyber Crimes and Digital Forensics) term project at SIIT.

A black-box detector that sits in front of an LLM and reads the **behavior of a whole session**, not one message at a
time. It targets injections that per-message detectors miss: an instruction split over several retrieved chunks, or a
slow drift that walks the session off the user's task.

> **Authority principle.** The user may give instructions; retrieved content may not. Each session keeps one reference
> embedding `R` of the user's latest task. Only a forwarded direct message that states a new task may change `R`;
> retrieved chunks are only measured against it.

## Pipeline

| Stage | What it computes | Channel |
|---|---|---|
| 1 Canonicalize | readable scoring copy (Unicode, homoglyphs, leetspeak, base64/hex/ROT13); rule flag `f`, obfuscation flag `g` | both |
| 2 Override check | content score `s` (in the proposed system: **FT-C**, a fine-tuned MiniLM), attack-bank fast path, content block path | both |
| 3 Relevance | `r` = similarity of the most instruction-like window to `R` | retrieved text |
| 4 Session behavior | CUSUM `a`, off-task share `o`, relevance decline `delta`, task resets `k`, blocked sources `u` | both |
| Decision | logistic regression over 13 features -> Forward / Forward-Untrusted / Block | both |

## Main result (5 fresh splits, seeds 20-24; 2,100 attack and 2,100 benign sessions)

| | Proposed (+FT-C+block) | PIGuard |
|---|---|---|
| Session detection | 0.650 [0.597, 0.703] | 0.446 [0.411, 0.481] |
| Session false-positive rate | 0.057 | 0.071 |
| Session AUC | 0.907 [0.894, 0.922] | 0.828 [0.813, 0.842] |
| Detection at 5% / 10% FPR | 0.671 / 0.740 | 0.448 / 0.490 |

Paired difference in detection: +0.204 [+0.180, +0.227]. On drift and split-chunk attacks, per-message detectors see
almost nothing (0.01-0.18) while the proposed system reaches 0.38-0.54.

**Read these numbers with the paper's caveats:** FT-C is trained in-domain while the guard models are used as
released; the choice of the proposed variant among several was made from the final run (the pre-stated rule would have
picked a log-odds variant whose AUC is higher by only 0.003, inside the interval), so the figures are descriptive, not
a confirmatory test; thresholds do not transfer exactly between splits; English only; no adaptive attacker.
Small result tables are in `results/final_ft/`; the raw event logs are too large for git.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_env.py            # checks Python, packages, GPU/CPU, disk, internet, data, token; says what to fix
python -m pytest -q                       # unit tests, no data needed
```

### Data (not in git)

```bash
python scripts/01_download_data.py --with-models     # BIPIA, deepset, Gandalf, NotInject, the embedding model
python scripts/05_build_tasktracker.py                # TaskTracker (about 2 GB free, slow first step)
```

Check each dataset's license before redistributing anything.

### Which device is used (nothing to configure)

The code picks the best device on its own, on Windows, macOS and Linux: **NVIDIA GPU (CUDA) > Apple GPU (MPS) > CPU**.
`python scripts/00_check_env.py` shows what it found and which one will be used. A GPU is not required, but the first
demo start and a full run are slower on CPU (we measured only on an NVIDIA GPU). To force a device, set `PI_DEVICE`
to `cpu`, `cuda` or `mps` before running.

### PIGuard runs code from its own model repo

The PIGuard baseline (`leolee99/PIGuard`) is loaded with `trust_remote_code=True`, i.e. it executes the small
`modeling_piguard.py` file that ships with the model on Hugging Face. We read that file before enabling it (it only
subclasses `DebertaV2ForSequenceClassification` and swaps the classifier head), but we do not redistribute it. Read it
yourself, or set `baselines.piguard.enabled: false` in `config.yaml` to skip this baseline.

### Hugging Face token (optional)

Only **one** model needs it: Meta's Prompt-Guard baseline (`B3a`), which is gated. Without a token that baseline is
skipped automatically; the proposed system, the demo and the other baselines work without it.

1. Create a free account at <https://huggingface.co/join>.
2. Open <https://huggingface.co/meta-llama/Prompt-Guard-86M> while logged in and accept the license (approval can take a while).
3. Create a token at <https://huggingface.co/settings/tokens>: *Create new token* -> type **Read** -> copy it once.
4. Give it to your own shell, not to a file. Either log in once (stored in your user folder, outside this repository):
   ```bash
   huggingface-cli login        # newer versions: hf auth login   (paste the token when asked)
   ```
   or set it for the current terminal only:
   ```powershell
   $env:HF_TOKEN = "hf_xxx"      # Windows PowerShell
   ```
   ```bash
   export HF_TOKEN=hf_xxx        # macOS / Linux
   ```
5. Run `python scripts/00_check_env.py` again: it tells you whether the token is valid and whether Prompt-Guard
   access is granted (it never prints the token). Then `python scripts/01_download_data.py --with-promptguard`.

Never write a token into a file or commit it. If one leaks, revoke it at the same settings page and create a new one.
Clear it when you are done: `Remove-Item Env:HF_TOKEN` (PowerShell) or `unset HF_TOKEN`.

### Reproduce the final run

```bash
python followup/run_ft.py --seeds 20 21 22 23 24 --ours-fpr-scale 0.4 --out results/final_ft/h1
```

Settings of this run and the rules written before it are in `followup/PREREG_ft.md`. The full experiment guide
(timings, troubleshooting, data-leakage rules) is `docs/EXPERIMENT_GUIDE.md`.

## Interactive demo

A three-panel local web page: **chat** (verdicts), **pipeline** (the diagram of the paper, the path of each event
lights up with its results), **numbers** (every computed value, with a glossary that explains each variable and where
it comes from). It runs the real proposed system on one seed, event by event; panels can be resized by dragging.

```bash
python demo/server.py            # ~5 min: the trained FT-C weights ship in demo/weights/ (see below); without them +70 s
# open http://127.0.0.1:8765 when it prints "demo ready"
python demo/server.py --stub     # fake numbers, only to test the page (shows a red banner)
```

**Trained weights.** The fine-tuned FT-C of seed 20 (the one the demo uses, 91 MB) is stored in `demo/weights/` as three
parts of about 30 MB plus a manifest with SHA-256 hashes, because GitHub refuses files over 100 MB. The demo joins and
verifies them on first start (`demo/weights_pack.py`); nothing to do by hand. They were trained on the development
split of seed 20 built from the data downloaded by `scripts/01_download_data.py` and `scripts/05_build_tasktracker.py`;
if your data differs, delete `demo/cache/ftc_seed20.pt` and the demo fine-tunes a new one (about 70 s on a GPU).
If a part is missing or damaged (a bad download), run `python demo/weights_pack.py check`: it lists every part as OK / MISSING / DAMAGED
and prints the repair commands (`git checkout -- demo/weights`, or re-download the folder).

Type a message or paste a retrieved document, draw a random example from the held-out test split, or play one of six
random scenarios (benign, direct attack, injection in a document, split chunks, drift, topic change). Details:
[`demo/README.md`](demo/README.md).

## Layout

```
pidetect/      the system: stages 1-4, decision module, baselines, metrics
experiments/   original experiment scripts (H1-H3, decision model, component checks)
followup/      final run with FT-C (run_ft.py, ft_scorer.py), ablation, decision-model checks, PREREG_ft.md
scripts/       data download / build, environment check
demo/          interactive three-panel demo
tests/         unit tests
results/       small result tables of the final run (no raw logs)
docs/          the long experiment guide
```

## Team

SIIT, CSS453, instructor Somchart Fugkeaw. Add the group members and their contributions here.

## License

Not chosen yet. Add a LICENSE file before making the repository public.
