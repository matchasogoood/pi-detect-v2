# Experiment Guide
**Topic 23 — Behavior-Based Detection of Prompt Injection Attacks in LLM Applications**

This folder holds the code of the **proposed system** (Stages 1–4 plus a logistic-regression Decision Module) and of
**every experiment in Section IV** of the paper. The sections below are written to be followed in order, step by step,
even by someone who has never written Python.

> **Two rules that must not be broken**
> 1. Results produced with `--data synthetic` or `--embedder hashing` only show **that the code runs**. They must
>    **never** go into the paper. The scripts drop a `WARNING_NOT_FOR_PAPER.txt` file in those result folders.
> 2. If a result does **not** meet its criterion, **report it as it is**. Never adjust the test data or the criterion
>    to make it pass.

---

## Contents
1. [What the experiments are, and where they appear in the paper](#1)
2. [Prerequisites](#2)
3. [Installation (once)](#3)
4. [A quick run to check the installation (no data needed)](#4)
5. [Downloading the real data](#5)
6. [TaskTracker (built with one command)](#6)
7. [Running the real experiments](#7)
8. [Reading the results, and where each one goes in the paper](#8)
9. [The settings in config.yaml](#9)
10. [What each file does](#10)
11. [How the system works (paper symbols to code)](#11)
12. [Data-leakage rules](#12)
13. [Troubleshooting](#13)
14. [Limitations to be aware of](#14)
15. [Using Google Colab instead of your own machine](#15)

---

<a id="1"></a>
## 1. What the experiments are, and where they appear in the paper

The single research question of the paper: *can a black-box detector that reads session behavior under the authority
principle catch injections that a per-message detector misses, without raising false alarms when the user changes
topic on their own?*

| Name | Script | What it answers | In the paper | Criterion (fixed before running) |
|---|---|---|---|---|
| **Prelim** | `experiments/prelim_decision_model.py` | Chooses the decision model (reruns Table VI with the attack bank rebuilt per fold, so nothing leaks) | Section IV-C, Table VI | the one-standard-error rule picks the model |
| **H1** | `experiments/h1_behavior_vs_memoryless.py` | Does reading the session state catch more attacks, and sooner? (the question in the paper's title) | Section IV-D | the 95% interval of the session detection difference (full − memoryless) excludes 0 |
| **H2** | `experiments/h2_instruction_relevance.py` | Does Stage 3 (comparing the instruction window with the user's task) catch indirect injections that Stages 1–2 miss? | Section IV-E | the 95% interval of the TPR difference at 1% FPR (r_window − S12) excludes 0 |
| **H3** | `experiments/h3_authority_principle.py` | Does the authority principle cut false alarms on genuine topic changes while resisting drift and probing? | Section IV-F | FAR drops against the accumulated profile (CI excludes 0) with no higher ASR |
| **S** | `experiments/s_component_checks.py` | Component checks on Stages 1–2 (soft flag, clip, fast path, canonicalization) plus cost | Section IV-G | exploratory (not a main claim); report CIs with the Holm correction |

H1, H2, H3 and S are repeated over 5 seeds (`seeds` in the config) and reported as a mean with a 95% confidence
interval. Prelim uses stratified 5-fold cross-validation repeated 3 times, as in Table VI.

**How this differs from the earlier code (`pi_detect_experiments`)** — the earlier version was the earlier system
(Stage 2 = bank similarity + λf, with a threshold per stage). This version implements the **current paper** in full:
canonicalization with g, the classifier head with sentence windows, the clipped fusion, Stage 3 on the instruction
window (ω*), Stage 4 on both channels (o, f̄, δ, h, the CUSUM a, k, u), the 13-feature LR Decision Module,
Forward-Untrusted, trust and τ_a, a real top-k retriever, and the experiments reorganized into Prelim / H1 / H2 / H3 / S.
The earlier system survives as `rulebased.py`, which is used as a baseline in H3c.

---

<a id="2"></a>
## 2. Prerequisites

| Requirement | Detail |
|---|---|
| Computer | Windows 10/11, macOS or Linux. **No GPU needed** (one makes it faster) |
| RAM | at least 4 GB (8 GB recommended) |
| Disk space | at least 5 GB free (torch + models + data + results) |
| Internet | needed for the installation and the first download |
| Python | **3.12 or 3.13** (3.9–3.11 also work, but then install `lingua-language-detector==2.1.1`, see below) |

**Installing Python (if you do not have it)**
- Windows: go to https://www.python.org/downloads/, download 3.13, and **tick "Add python.exe to PATH"** before
  clicking Install
- macOS: the same link, or `brew install python@3.13`
- To check: open a terminal (Windows: press the Windows key, type `cmd`, press Enter) and run
  ```
  python --version
  ```
  `Python 3.13.x` means you are ready. On some macOS/Linux machines you must type `python3` instead of `python`
  in every command in this guide.

> Prefer not to install anything? Jump to [section 15 (Google Colab)](#15).

---

<a id="3"></a>
## 3. Installation (once)

**Step 1 — unzip** `pi_detect_v2.zip` anywhere, e.g. `Documents/pi_detect_v2`.

**Step 2 — open a terminal and go into the folder**
```
cd Documents/pi_detect_v2
```
(On Windows, if the folder is on another drive such as D:, type `D:` first, or use `cd /d <project>
Every command below **must be run from this folder**. VS Code works too: open the folder, then use its built-in
terminal.

**Step 3 — create a virtual environment** (a box that keeps this project's packages separate from the rest of the machine)
```
python -m venv .venv
```
Then **activate** it (needed every time you open a new terminal):
- Windows (cmd): `.venv\Scripts\activate`
- Windows (PowerShell, which is what VS Code uses): `.venv\Scripts\Activate.ps1`
  (if it complains about the execution policy, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first)
- macOS / Linux: `source .venv/bin/activate`

When it works, the prompt starts with `(.venv)`.

**Step 4 — install the packages**
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This takes 5–15 minutes (torch is large). On a machine with no GPU, a smaller CPU-only torch can be installed first:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
On Python 3.11 or older, `lingua-language-detector==2.2.0` cannot be installed. Edit that line in
`requirements.txt` to `lingua-language-detector==2.1.1`, or create the virtual environment with Python 3.12+ instead.

**Step 5 — check the machine**
```
python scripts/00_check_env.py
```
It reports `[ OK ]` / `[WARN]` / `[FAIL]` for each item with a hint. At this point the data is not downloaded yet, so
the "data" section will FAIL, which is expected (fixed in section 5).

---

<a id="4"></a>
## 4. A quick run to check the installation (no data needed)

```
python scripts/run_all.py --smoke --data synthetic --embedder hashing
```
- `--smoke` = little data, 1 seed; `--data synthetic` = fake data; `--embedder hashing` = no model download
- Takes about 1–3 minutes. Ending with `all succeeded` means every experiment runs end to end.
- The numbers are **meaningless** (fake data), so do not be alarmed by odd results.

Optionally, run the automated tests:
```
python -m pytest -q                 # everything (~1 minute)
python -m pytest -q -m "not slow"   # the fast part only (~2 seconds)
```

---

<a id="5"></a>
## 5. Downloading the real data

```
python scripts/01_download_data.py --with-models
```
Downloads into `data_raw/`:

| Dataset | Source | Used for |
|---|---|---|
| BIPIA (email, table) | GitHub microsoft/BIPIA (its official train/test split) | indirect injection (H1, H2) |
| deepset/prompt-injections | Hugging Face | direct attacks + benign text (every experiment, Table VI) |
| Lakera/gandalf_ignore_instructions | Hugging Face | direct attacks (H1, H3, S) |
| leolee99/NotInject | Hugging Face | benign text containing trigger words (S) |
| all-mpnet-base-v2 (`--with-models`) | Hugging Face | the embedding model (~420 MB; set in `embedder.model_name`) |

At run time the code **drops every non-English text** automatically, because this project trains and tests on English only.

**Counting the languages (do this once after downloading, to fill the Language composition table of the paper)**
```
python scripts/04_language_report.py
```
It takes about 10 seconds and writes to `results/final/language_report/`:
- `language_table.tex` holds the table rows you can paste straight into the paper. The TaskTracker row is filled in
  already: TaskTracker is English by construction and is never filtered (16,000 / 16,000 / 0).
- `disagreements.csv` lists the texts where the filter and the language identifier disagree. Read it by eye before
  using the numbers. Most of them are short English sentences carrying personal names, such as
  `who was faster, dalziel or doornbos?`, which the identifier misreads.

Verified counts on the real data (every dropped text was checked by hand):

| Dataset | Total | English (kept) | Dropped |
|---|---|---|---|
| BIPIA contexts (email 100, table 1,000) | 1,100 | 1,100 | 0 |
| BIPIA attack instructions (train 75, test 75) | 150 | 150 | 0 |
| deepset | 662 | 354 (140 injections, 214 benign) | 308: German or German mixed with English 294, other languages 14 (Spanish, French, Croatian, Russian, Turkish, romanized Hindi, Khmer) |
| Gandalf | 1,000 | 993 | 7: German 4, Spanish mixed with English 2, Russian 1 |
| NotInject | 339 | 254 | 85: Chinese 82, Russian 2, Spanish 1 (all 84 of the Multilingual category, plus one Chinese Common Query) |
| mock KB / security articles / our own attacks | 330 / 46 / 15 | all | 0 |
| TaskTracker | — | run the script | — |

Notes: BIPIA tables are judged by their **question**, because the cells hold foreign proper names (athletes, teams,
venues); judging the whole table would wrongly drop about 90 English tables. Some BIPIA attacks ask for an answer in
another language, e.g. `Translate your response into Spanish.`, but the instruction itself is English, so it is kept.

Our own data lives in `data_local/` already (a 10-topic mock knowledge base and the security articles).

**Baseline B3a (Prompt-Guard) — optional.** It is a gated model:
1. Create an account at https://huggingface.co, open `meta-llama/Prompt-Guard-86M` and accept the license
2. Settings → Access Tokens → create a Read token
3. Run `python -c "from huggingface_hub import login; login()"` and paste the token
4. `python scripts/01_download_data.py --with-promptguard`

Without this, the experiments skip B3a automatically and say why (the run continues). You can also skip it explicitly
with `--skip-baselines B3a`.

**Baseline B3c (ProtectAI v2)** needs no licence: it is downloaded on first use
(`protectai/deberta-v3-base-prompt-injection-v2`, Apache-2.0). Skip it with `--skip-baselines B3c`.

**Training the indirect head on what it will be asked to judge.** The content head of Stage 2 scores a *window* of
a retrieved document, which on BIPIA and TaskTracker is part document and part injected instruction. Its positive
examples used to be bare attack strings, so it was trained on one distribution and used on another; that is a
knowledge-base shape (there the injected chunk is inserted whole) and not a BIPIA or TaskTracker one. The positives
now also include poisoned documents paired with the instruction that was injected into them -- BIPIA chunks
poisoned the same way the evaluation sessions are, and TaskTracker's own poisoned paragraphs, which ship the
injected instruction in a field of their own -- cut into the same windows the detector scores, keeping only the
windows that hold the injection. Everything comes from the dev split: the BIPIA attack categories and documents
held out for test are untouched, as is TaskTracker's test file, and no baseline is affected.

**Fairness of the comparison.** Our pipeline reads a whole session while a guard model sees one message, so the
baseline list also contains session-aware and tunable forms of each baseline; a win that only appears against the
weakest form of a baseline is not a win. `B1t` scores by how many patterns matched, so it can be placed at the same
false-alarm target as everything else (plain `B1` has no threshold). `B2` adds the verbatim-match bonus to the
similarity instead of saturating at 1.0, so its score no longer piles up at a ceiling that no threshold can split.
A name ending in `+sn` (`B3c+sn`) is the same baseline judged within its own session: each event is scored by how
far it stands above the events before it, which is the cheapest form of the context our pipeline uses. Finally,
`h1.ext_content` runs our session layer with a public classifier's score in place of our Stage 2, so that the only
difference left between that variant and the baseline is the session layer itself.

`h1.ext_modes` / `h2.ext_modes` choose how that classifier's score is used. `ind_win` gives it the third-party
channel only (on messages the user writes it changes nothing, because the two score about equally there) and lets it
read each Stage 3 window of a long document, keeping the worst -- the same rule the guard models already apply to
their own 512-token segments, at a finer grain. `ind_win_logit` adds the two corrections of Section IV-B: the score
enters the decision model as log-odds rather than as a probability, and the decision model fits a separate slope per
channel for the content score and the two lexical counts. Neither correction touches the baseline. The log-odds map
is strictly monotone over everything a float32 softmax can distinguish, so the baseline's own AUC and its detection
rate at any false-alarm rate are unchanged by construction; it only stops a probability that saturates near 1 from
compressing the region a low threshold reads. The per-channel slopes are a change to our decision model, fitted on
validation like every other constant, and the run also reports `full(per-channel)` -- our own system with those
slopes and no external model -- so that the correction can be read apart from the guard model it was found with.

Both model baselines are used the way their model cards prescribe, so that the comparison is fair:
Prompt-Guard is scored per channel (the jailbreak probability for messages the user writes, the
injection-or-jailbreak probability for third-party content), and texts longer than the 512-token window are split
into overlapping segments and scored segment by segment, a text counting as high as its worst segment.

Check again: `python scripts/00_check_env.py` should now have no FAIL (TaskTracker stays a WARN until section 6).

---

<a id="6"></a>
## 6. TaskTracker (built with one command)

The paper uses the TaskTracker data of Abdelnabi et al. in H1 and H2. Each instance is a user task plus a paragraph
that may carry an injected instruction, which is exactly the input Stage 3 judges.

**Why it has to be built:** its authors publish no ready-made data files, only Jupyter notebooks that assemble the data
from SQuAD, HotPotQA and several attack sets. `scripts/05_build_tasktracker.py` follows the same steps as their four
notebooks (clean and poisoned, validation and test) in a single command — **no Jupyter, and no cloning their repo.**

### Step 1 — prepare (once)
- Finish section 3 (installation) and make sure the prompt shows `(.venv)`
- Be inside the `pi_detect_v2` folder (`dir config.yaml` on Windows, `ls config.yaml` on macOS, must find the file)
- About **2 GB** of free space (the downloaded sources are ~650 MB and can be deleted afterwards)
- An internet connection that can reach Hugging Face (if huggingface.co opens in a browser, it works)

### Step 2 — run the script
```
python scripts/05_build_tasktracker.py
```
It runs four steps and prints its progress:
1. `[1/4]` small files from GitHub (a few seconds)
2. `[2/4]` SQuAD and HotPotQA — **the slow one**: the HotPotQA train file is 540 MB and prints `xx / 540 MB` as it
   goes (5–20 minutes on a normal connection). Leave it running.
3. `[3/4]` three datasets from Hugging Face (CodeAlpaca, BeaverTails, do-not-answer)
4. `[4/4]` assembling the data and writing the files

On success it ends with something like:
```
  val: 8000 rows (4000 poisoned, 4000 clean) -> ...data_raw/tasktracker/val.jsonl
  test: 8000 rows (4000 poisoned, 4000 clean) -> ...data_raw/tasktracker/test.jsonl
done.
```

**A warning you can ignore:** `could not download TrustLLM jailbreak.json -> skipped (optional)`. That dataset is gated
on Hugging Face; the script continues with the other validation attack sources and records the fact in `BUILD_INFO.json`.
To include it: log in (`python -c "from huggingface_hub import login; login()"`), open
https://huggingface.co/datasets/TrustLLM/TrustLLM-dataset, accept the terms, and run the script again (already
downloaded files are not fetched twice).

### Step 3 — check the result
```
python scripts/00_check_env.py
python scripts/04_language_report.py
```
- the TaskTracker line of `00_check_env.py` must be `[ OK ]`
- `04_language_report.py` has a TaskTracker row for information only: the experiments never drop TaskTracker rows
  (its Wikipedia paragraphs quote foreign names, which made the filter wrongly drop 475 English rows), so the
  paper's Table V row is 16,000 / 16,000 / 0

### If something goes wrong
| Symptom | Fix |
|---|---|
| `could not download hotpot_train_v1.1.json`, or the download stops | run the same command again (finished files are skipped). If it still fails, open the URL the script prints in a browser and save the file into `data_raw/tasktracker_src/` under the name it gives |
| not enough space or bandwidth for the 540 MB file | `python scripts/05_build_tasktracker.py --skip-hotpot-train` (clean test is then sampled from HotPotQA dev, a small difference that is recorded in `BUILD_INFO.json`) |
| `could not load PKU-Alignment/BeaverTails` (or another HF dataset) | open that dataset page on huggingface.co; if it asks you to accept its terms, log in and accept, then log in locally with `python -c "from huggingface_hub import login; login()"` and run again |
| `No module named datasets` | `.venv` is not activated, or `pip install -r requirements.txt` has not been run |

**Deliberate differences from the original notebooks (already written up in the paper)**
1. **Leakage guard:** their validation notebook uses BIPIA attacks from both train and test, but we use BIPIA test as
   our test set, so the BIPIA test attacks are excluded from validation (the data loader drops them a second time as a
   safety net).
2. Seeded sampling, keeping 4,000 rows per label per split (the full sets have tens of thousands of rows). Change it
   with `--max-rows`.
3. No training split (500K+ rows, which we do not use) and no translated variants (we are English-only).

**Alternative (if you already ran their notebooks):** convert the resulting `dataset_out_*.json` files with
```
python scripts/03_convert_tasktracker.py dataset_out_clean_v2.json dataset_out_poisoned_v2.json --split test
python scripts/03_convert_tasktracker.py dataset_out_clean.json dataset_out_poisoned_v1.json --split val
```

**If it turns out to be impossible**, set `data: use_tasktracker: false` in `config.yaml` and carry on, but the paper
(Datasets, H1, H2, Table V) must then say that TaskTracker was not used.

---

<a id="7"></a>
## 7. Running the real experiments

**Try a small run on the real data first** (it checks the data and takes a few minutes):
```
python scripts/run_all.py --smoke
```
Then the full run:
```
python scripts/run_all.py
```
Or one at a time (the recommended order):
```
python experiments/prelim_decision_model.py                # Table VI rerun (old inputs: f, m, n_imp, n_ai)
python experiments/prelim_decision_model.py --inputs new   # Table VI with the new Stage 2 (s, n_imp, n_ai)
python experiments/h1_behavior_vs_memoryless.py
python experiments/h2_instruction_relevance.py
python experiments/h3_authority_principle.py
python experiments/s_component_checks.py
python scripts/02_collect_results.py                       # merges everything into results/final/RESULTS.md
```

**Options available to every script**

| Option | Meaning |
|---|---|
| `--smoke` | small run (1 seed, little data) |
| `--seeds 0 1` | choose the seeds (default 0–4) |
| `--skip-baselines B3a B3b` | skip baselines that are not installed |
| `--out FOLDER` | change where results are written |
| `--config FILE.yaml` | use a different config |

**Rough timings** (an ordinary CPU, real embeddings): smoke runs take a few minutes per experiment; a full 5-seed run
takes roughly 10–60 minutes per experiment depending on the machine (the first one is slowest, because every embedding
is computed from scratch). If it takes too long, reduce `--seeds` or the numbers under `h1:` / `h2:` / `h3:` in the
config — but then report the real sizes in the paper.

---

<a id="8"></a>
## 8. Reading the results, and where each one goes in the paper

Results of the final run live in `results/final/<experiment>/` (development and smoke runs are kept apart in `results/dev/` and `results/smoke/`; see `results/README.md`). The file to open first is **`summary.md`** (any Markdown viewer, or VS Code):
its first line says whether the criterion was **met**. `results/final/RESULTS.md` is every experiment merged into one file.

| Experiment | Files | Where it goes in the paper |
|---|---|---|
| Languages | `language_report/language_table.tex` | the "Language composition" table (replaces the red `\tbd{}`) |
| Prelim | `prelim_old/table_v.md` and `prelim_new/table_v.md` (plus the LaTeX block in each summary.md) | the numbers of Table VI; `prelim_old` keeps the features of the published table, `prelim_new` repeats it with the fused Stage 2 score, as the paper promises; `permutation_check.md` confirms nothing leaks; `lr_coefficients.md` gives the weights for the analysis paragraph |
| H1 | `h1/overall.md`, `h1/h1_tests.md`, `h1/by_attack_kind.md`, `h1/cost.md`, `h1/low_trust_share.md` | the H1 results, the differences with CIs, the breakdown by attack kind and source, the latency and memory of each configuration, and the share of benign sessions that end up in the low-trust state |
| H2 | `h2/tpr_at_fpr.md`, `h2/h2_tests.md` | TPR per subset at both the 1% and the 5% FPR target, each with the FPR actually achieved, `target_reached` (within 25% of the target) and `rel_se` (how precisely that target can be estimated from the benign events available). The paper's headline is 5%; 1% is reported as an under-powered secondary. ontopic and split_windows are the failure cases that must be reported |
| H3 | `h3/h3a_false_alarms.md`, `h3/h3b_boiling_frog.md`, `h3/h3c_threshold_aware.md`, `h3/h3d_probing_reset.md`, `h3/h3b_weak_anchor.md`, `h3/h3a_tau_same_sensitivity.md` | H3a–H3d in order |
| S | `secondary/components.md`, `secondary/secondary_tests.md`, `secondary/cost.md` | the secondary section and the cost figures (latency, memory) |

**Reading a TPR at a target FPR.** A threshold is set on validation for a target FPR and then applied unchanged,
so the FPR actually achieved on test is reported next to every TPR. A model whose score takes few distinct values
(a shallow tree, a rule) cannot be placed at the target at all: its threshold lands at or above its highest benign
score, nothing is flagged, and the row reads TPR 0.000 at FPR 0.000. `n_distinct_scores` and `target_reached` in
`prelim_*/table_v.md` make that visible; for such a model the AUC, which needs no threshold, is the number to read.

**Method names in the tables**
- `full` = the system the paper proposes
- `memoryless` = the same system with no state (only s, c, n_imp, n_ai)
- `ref_only` = a reference R but no session history
- `S12` = Stages 1–2 only / `S12+r_chunk` = relevance of the whole chunk / `S12+r_window` = Stage 3 as in the paper
- `accumulated` = the running profile (the earlier system) / `channel_agnostic` = no authority principle
- `B1-regex`, `B2-banned+emb`, `B3a-PromptGuard`, `B3c-ProtectAIv2` = the baselines (B3b, PIGuard, is off by default)

**Terms in the tables**: `session_detection` = share of attack sessions caught, `session_far` = share of benign
sessions falsely alarmed, `delay` = events from the start of the attack to the block, `ASR` = share of attacks that got
through, `TPR`/`FPR` = event level, `lo`/`hi` = the 95% confidence interval, `excludes_zero` = the interval excludes 0
(significant).

**Numbers quoted from other papers** (such as the white-box TaskTracker results of Abdelnabi et al.) go into
`data_local/reference_numbers_template.csv` with their source, and may appear in the discussion only, never in a
comparison table.

**Every result folder has** a `run_info.json` (when it ran, which data and model, and `language_filter` = how many
texts each dataset had before and after the English filter). If `WARNING_NOT_FOR_PAPER.txt` sits beside it, those
numbers **must not be used**.

---

<a id="9"></a>
## 9. The settings in config.yaml

> Edit it in VS Code or any text editor. **Never use tab characters** — only spaces.

| Setting | Meaning | In the paper |
|---|---|---|
| `embedder.kind` | `sbert` = the real one, `hashing` = the fallback for testing the code | Stage 2 |
| `data.source` | `real` / `synthetic` | — |
| `data.english_only` | keep English text only (default true) | Section IV-A |
| `data.use_tasktracker` | whether TaskTracker is used | Section IV-A |
| `seeds` | the seeds (5 of them = five random splits) | Protocol |
| `bank.n_clusters`, `n_holdout` | number of attack families / how many are held out | Stage 2, data splits |
| `stage2.doc_len_threshold`, `window_sents`, `max_windows` | long indirect documents are scored per sentence window | Eq. 5 |
| `stage2.eta` | η = the clip bound | Eq. 6 |
| `stage2.nu_fast`, `tau_f_min` | how tight the fast path is (τ_f) | Stage 2 |
| `pipeline.W`, `k_min` | Stage 4 window size / minimum events before its signals count | Stage 4 |
| `fit.nu`, `nu_r` | target FPR for Block (1%) and Forward-Untrusted (5%) → t_blk, t_rev | Eq. 12 |
| `fit.tau_o_quantile`, `rho_quantile`, `tau_a_quantile` | τ_o, ρ and τ_a as quantiles of benign validation data | Stage 4, authority rule |
| `fit.C_grid`, `cv_folds` | choosing the LR's C by cross-validation | Eq. 11 |
| `fit.n_fit_sessions` | how many validation sessions are used to fit | — |
| `prelim.*` | 3×5-fold, 5% FPR, old or new inputs | Table VI |
| `h1.target_session_far` | every method is set to the same session false-alarm rate (5%) | H1 |
| `h2.target_fpr` | FPR of indirect events (1%) | H2 |
| `h3.Ns`, `max_attempts` | number of drift chunks / probing attempts | H3b, H3d |
| `secondary.lambdas` | the fixed λ values compared against the soft flag | S |
| `smoke.*` | the values used instead when `--smoke` is passed | — |

---

<a id="10"></a>
## 10. What each file does

```
pi_detect_v2/
├── README.md                 this guide
├── config.yaml               all the settings
├── requirements.txt          the package list
├── data_local/               our own data (mock KB, security articles)
├── data_raw/                 (created by the download) external data
├── results/                  (created by the runs) the results
├── experiments/              the five experiment scripts (see section 1)
├── scripts/
│   ├── 00_check_env.py       checks the machine and the data
│   ├── 01_download_data.py   downloads the data and the model
│   ├── 02_collect_results.py merges everything into results/final/RESULTS.md
│   ├── 03_convert_tasktracker.py  converts files from the TaskTracker notebooks
│   ├── 04_language_report.py counts the languages per dataset (Language composition table)
│   ├── 05_build_tasktracker.py builds the TaskTracker dataset in one command (section 6)
│   └── run_all.py            runs every experiment in order
├── pidetect/                 the system itself (no need to touch it just to run the experiments)
└── tests/                    the automated tests
```

| File in `pidetect/` | Role |
|---|---|
| `schema.py` | data structures: Event (text, channel, label, source d, turn t), Session, the three action levels |
| `embedder.py` | text to vector (Sentence-BERT) plus the hashing fallback |
| `data/lang.py` | the English-only filter |
| `canon.py` | **Stage 1**: the canonical scoring copy (leetspeak, homoglyphs, zero-width, base64 …) and the indicator g |
| `rules.py` | **Stage 1**: the rule flag f ∈ {0, 0.5, 1} |
| `attackbank.py` | the attack bank (built from dev only, with 2 families held out) |
| `stage2.py` | **Stage 2**: m (bank similarity), q (the head with sentence windows), s (fusion with the clip), the fast path |
| `lexical.py` | the n_imp and n_ai counters |
| `pipeline.py` | **Stages 3–4 + Decision Module + the authority rule** (Algorithm 1) |
| `fit.py` | fits and calibrates everything on dev/val (two passes, out-of-fold thresholds) |
| `sessions.py` | the dev/val/test splits, the top-k retriever, and every session builder (topic change, boiling frog, probing …) |
| `evaluate.py` | event-level and session-level metrics |
| `stats.py` | paired bootstrap, Holm, McNemar, corrected t-test |
| `rulebased.py`, `rulebased_tuning.py` | the rule-based baseline (the earlier system), used in H3c |
| `baselines.py` | B1, B2, B3a, B3b |
| `runtime.py`, `expkit.py`, `report.py`, `metrics.py`, `chunking.py`, `data/` | shared utilities |

---

<a id="11"></a>
## 11. How the system works (paper symbols to code)

For each message (`ProposedDetector.step` in `pipeline.py`), following Algorithm 1:

| Step | In the paper | In the code |
|---|---|---|
| Stage 1 | the canonical copy, g, f | `canon.canonicalize`, `rules.rule_flag` |
| Stage 2 | m (Eq. 3), q (Eq. 4–5), s (Eq. 6), the fast path | `Stage2.process` |
| Stage 4 (direct) | the CUSUM a (Eq. 8), k, u | `st.a`, `st.D`, `st.B` |
| Stage 3 | r = cos(e_ω*, R), or r̄ when there is none | `e_rel = s2.raw.e_top` |
| Stage 4 (indirect) | o, f̄ (earlier events only), δ (between retrieval turns), h | `_behavior` |
| Decision | the 13 features x (Eq. 9) → p (Eq. 10) → Block / Forward-Untrusted / Forward (Eq. 12) | `predict_p`, `cfg.t_blk`, `cfg.t_rev` |
| No trusted reference | indirect text is at least Forward-Untrusted | `untrusted_floor` |
| Authority rule | direct, not blocked, a new task → new R, clear H, trust = [Forward ∧ s < τ_a] | the end of `step` |
| Reasons | ψ_j (Eq. 13) | `reasons` |

**Fitting** (`fit.py`): the heads are trained on dev → κ, the fusion and τ_f on the first half of val → r̄, τ_o, ρ, τ_a
and τ_same on the same half → the LR on the second half in two passes (the first with no LR, the second with the LR so
that the state matches the real policy) → t_blk and t_rev as quantiles of the out-of-fold probabilities (folds grouped
by session).

**Variants** reuse the same Stage 2 and the same constants as the full system and refit the LR the same way, which
makes the comparison fair.

---

<a id="12"></a>
## 12. Data-leakage rules

- Every source is split into **dev / val / test by seed before** any session is built
- **dev** = the attack bank and the head training, **val** = all fitting of the fusion, the LR and the thresholds,
  **test** = reported once
- BIPIA keeps its **official** train/test split (no test attack is ever used for training)
- the mock KB is split by topic (one topic never appears in both val and test)
- 2 attack families are held out of the bank, to measure unseen attacks
- Prelim (Table VI) rebuilds the bank **in every fold** from that fold's training part
- **Never** change the data or a criterion after seeing a test result

---

<a id="13"></a>
## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `python` is not recognized | Windows: reinstall Python with "Add to PATH" ticked / macOS-Linux: use `python3` |
| no `(.venv)` in the prompt | not activated (section 3, step 3) — needed in every new terminal |
| `pip install` fails on torch | use Python 3.12/3.13 and install the CPU torch as in section 3 |
| `lingua-language-detector==2.2.0` cannot be installed | you are on Python 3.11 or older: change that line to `==2.1.1`, or rebuild `.venv` with Python 3.12+ |
| `file not found: data_raw/...` | run `python scripts/01_download_data.py` |
| `no *.jsonl files in data_raw/tasktracker` | do section 6, or set `use_tasktracker: false` |
| `could not load the embedding model` | check the internet, then run `python scripts/01_download_data.py --models-only` |
| `[baseline B3a] skipped` | normal when Prompt-Guard access has not been requested (section 5) |
| `config.yaml has problems` | read the message; it is usually a tab character or wrong indentation |
| the run takes very long | try `--smoke` first, reduce `--seeds`, close other programs |
| `MemoryError` | lower `tasktracker_max_rows` and the `n_test_*` values in the config |
| H3 says "criterion untestable" | neither variant raised a single false alarm → report them as indistinguishable and read H3b alongside |
| a criterion is "not supported" | not a bug — report it honestly in the paper, with an analysis of why |

---

<a id="14"></a>
## 14. Limitations to be aware of (and to state in the paper)

1. **The n_imp / n_ai counters** in `lexical.py` are a rewritten lexicon and may not match the one behind the first
   version of Table VI — use the numbers from this Prelim run throughout.
2. **Follow-ups** in the simulated sessions come from templates plus key phrases taken from the documents; they are
   not real user messages.
3. **The mock KB** has 10 topics, written by the team → a member who did not write the detector should review and
   extend it.
4. **Spread attacks** (split_chunks, boiling frog, direct spread) are written by the team → H1 reports the breakdown
   by source.
5. **English only.** Every dataset is filtered down to English before the dev/val/test split (`data.english_only:
   true`), so attacks in other languages are out of scope (the paper says so in the Threat Model and in Threats to
   Validity). The filter is in `pidetect/data/lang.py` and combines character and stop-word rules with the lingua
   language identifier (which must be installed; the version is pinned in `requirements.txt` so that every machine
   drops exactly the same texts). Without lingua the code warns and writes `WARNING_NOT_FOR_PAPER.txt`, because the
   rules alone do not catch all the German in deepset. **Note:** deepset keeps only 354 of its 662 prompts
   (140 injections).
6. **TPR at 1% FPR** needs a lot of benign data; with little data the threshold is coarse — check `n_neg` in the
   tables and report it as the Calibration part of the paper requires.
7. **B2** is our own baseline design (banned phrases + embedding), not a reproduction of Rahman et al.
8. **Forward-Untrusted** cannot be evaluated against a real LLM (we do not run one) → it counts as "not blocked" in
   the ASR and its share is reported separately.

---

<a id="15"></a>
## 15. Using Google Colab instead of your own machine

1. Open https://colab.research.google.com → New notebook
2. Click the folder icon on the left → upload `pi_detect_v2.zip`
3. Run the cells one by one (press ▶):
   ```
   !unzip -q pi_detect_v2.zip
   %cd pi_detect_v2
   !pip install -q -r requirements.txt
   !python scripts/run_all.py --smoke --data synthetic --embedder hashing
   !python scripts/01_download_data.py --with-models
   !python scripts/05_build_tasktracker.py
   !python scripts/run_all.py --smoke
   !python scripts/run_all.py
   ```
4. Download the results: `!zip -r results.zip results`, then right-click `results.zip` → Download

> Colab disconnects after a period of inactivity and the files are lost → download `results.zip` as soon as a run
> finishes, or run one experiment at a time.
