# Interactive demo

A local web page that runs the **real proposed system** (`full+B5(ext)` = +FT-C+block) on one seed, one event at a
time, and shows what every stage computes. No LLM is called: the page plays the role of the application in front of
the model and shows what the detector would do with each message or retrieved document.

## The three panels

| Panel | What it shows |
|---|---|
| **1. Chat** | what the application sees. Each message gets a verdict: Forwarded / Forwarded as untrusted data / Blocked, with the reason. Retrieved documents are drawn dashed, with the dataset they came from and their **true dataset label**, so you can compare the verdict with the truth |
| **2. Pipeline** | the diagram of the paper. The boxes on the event's path light up in order and show their result (for example `f = 0.00 \| g = 0`). A block in Stage 2 sends the red fast-path / block-path line straight to the decision; Stages 3-4 are then greyed out. Click a box to jump to its numbers |
| **3. Numbers** | every value computed for the selected event, grouped by stage: FT-C score against its block threshold, bank similarity, relevance, CUSUM, the decision probability against its thresholds, how much each feature moved the log-odds, and the session state afterwards. **What are these variables?** opens a glossary: what each symbol means and where it comes from (computed / learned / calibrated / design). Click a variable name in the numbers to jump to its entry |

Click any message in the chat to inspect its event again. Drag the bars between panels (and above the input box) to
resize; double-click a bar to reset. Sizes are not remembered between page loads.

## Run

```bash
python demo/server.py              # the real system; open http://127.0.0.1:8765 when it prints "demo ready"
python demo/server.py --stub       # fake numbers to test the page only (red banner); never use it to present
python demo/server.py --seed 21    # another seed (fine-tunes FT-C again for that seed)
python demo/server.py --raw-dir PATH --port 8800
```

Needs the datasets (`python scripts/01_download_data.py --with-models`, `python scripts/05_build_tasktracker.py`) and a
Hugging Face login only if you use a gated model. Clear any `HF_TOKEN` you do not need; never put a token in a file.

**Start-up time** (measured on the development laptop with a GPU): about 6.5 minutes the first time, about 5 minutes
afterwards. The first run fine-tunes FT-C (70 s) and caches the weights in `demo/cache/ftc_seed<seed>.pt` (git-ignored);
later runs load them. The rest is loading data, embedding about 12,800 texts, fitting the system twice and
calibrating thresholds, which is repeated every time. Start the server about ten minutes before presenting and leave it
running. Wait for `demo ready`.

## What you can send

- **Type** a message and choose *User message* (the user typed it) or *Retrieved document* (RAG content, web page,
  tool output). A retrieved document is judged against the user's latest task, so send a user message first.
- The **User message / Retrieved document** switch also decides what is offered above it: with *User message* the example
  buttons are Benign question, Direct attack and Legitimate topic change and the dataset list holds user-message sets only;
  with *Retrieved document* they are Injection inside a document, Split across chunks and Topic drift, with document sets only.
  "Any dataset" means any dataset of the selected channel.
- **Random sample** draws one example from the held-out **test** split of the seed: direct attacks (deepset, Gandalf),
  benign messages (deepset, NotInject, our own trigger-word messages), poisoned documents (TaskTracker, BIPIA with an
  attack inserted, mock knowledge base) or clean documents (TaskTracker, BIPIA, mock knowledge base, security
  articles that quote attacks). It picks the dataset first, so small datasets are not drowned by large ones. For a
  document it first sends the matching user task.
- **Click a pipeline box** to highlight its card in the Numbers panel. If you then press *What are these variables?*, the
  glossary opens at that box's stage with every variable of the stage marked (the interceptor, user-prompt and
  retrieved-content boxes point to the channel flag `c`). Click empty space to clear the selection.
- **Hide panels 2 and 3** with the – button in their headers; they shrink to a narrow strip and the + button brings them back.
- **Language toggle (EN / TH)** in the header. English is the default. Thai changes the buttons, headings, hints and the
  whole variable glossary; the pipeline diagram, the chat texts, the verdicts and the numbers stay in English so they match
  the paper. The choice is not remembered between visits. Thai texts live in `glossary.json` and `examples.json` (`*_th` fields).
- **Scenario buttons** (benign, direct attack, injection inside a document, injection split across chunks, topic drift,
  legitimate topic change) send **one message of that kind per click** into the current session, drawn at random from
  the test split. Only **New session** starts a new session. The split and drift buttons hand out the next chunk of one
  multi-chunk attack on each click (click repeatedly to walk through it; a new attack is drawn when it is used up).
  A retrieved chunk is judged against a user task, so if the session has none yet, the matching task is sent first.

Everything sampled comes from the test split, which the system never trained on. Results on single sessions are
anecdotes: the numbers in the paper are averages over 5 seeds and 2,100 sessions per class. Split and drift attacks
are detected in about 54% and 38% of cases, so some scenarios will pass through. That is the real behavior.

## Files

```
server.py        standard-library HTTP server (API: /api/step, /api/sample, /api/scenario, /api/config, ...)
engine.py        builds the real system for one seed (reuses followup/run_ft.py) and returns a trace per event
sampler.py       random examples and random sessions from the test split
stub_engine.py   fake numbers with the same shape, for testing the page
glossary.json    the variable glossary (symbol, meaning, provenance)
examples.json    titles and blurbs of the scenario buttons; the fixed texts are used by --stub only
static/index.html  the page (HTML, CSS, JavaScript in one file)
```

## Notes

- The proposed system builds its content score `s` from **FT-C**; "our own fused score" (`s_ours`) is shown for
  comparison and is not used by the pipeline.
- The session-level alarm of the paper (at least one block in the session) is not drawn; the page shows the verdict of
  each event.
- Latency shown at the top of the pipeline panel is the time to process one event on this machine.
- Do not run `--stub` for a presentation: its numbers are fake.

## Trained weights

`demo/weights/` holds the fine-tuned FT-C of seed 20 as three parts (~30 MB each) and `ftc_seed20.json` (SHA-256 of every part and of the whole file). On the first start the demo joins them into `demo/cache/ftc_seed20.pt`, verifies the hash and loads them instead of fine-tuning (saves about 70 s). A damaged or missing part is reported and the demo fine-tunes from scratch. To rebuild the parts after retraining: `python demo/weights_pack.py split demo/cache/ftc_seed20.pt`.

Check or repair the shipped weights: `python demo/weights_pack.py check` (OK / MISSING / DAMAGED per part, plus the commands to fix it, e.g. `git checkout -- demo/weights`). `python scripts/00_check_env.py` reports the same in one line.
