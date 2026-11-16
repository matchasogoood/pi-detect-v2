"""pidetect: experiment code for Topic 23, "Behavior-Based Detection of Prompt Injection Attacks in LLM Applications"

The proposed system (reading the files in this order is the easiest way to follow it)
  schema.py     data structures: Event (eps = x, c, t, d), Session, the three action levels
  embedder.py   text -> semantic vector (Sentence-BERT all-MiniLM-L6-v2)
  canon.py      Stage 1 : scoring copy + obfuscation indicator g
  rules.py      Stage 1 : rule flag f ∈ {0, 0.5, 1}
  attackbank.py attack bank {a_j} (built from dev only, with attack families held out for testing)
  stage2.py     Stage 2 : bank score m, classifier head q (sentence windows), fused score s with clip, fast path
  lexical.py    the counters n_imp and n_ai
  pipeline.py   Stage 3-4 + Decision Module (LR) + authority rule = Algorithm 1
  fit.py        fits and calibrates everything on dev/val (never touches test)
  sessions.py   splits data into dev/val/test and builds the simulated sessions of every experiment
  evaluate.py   event-level and session-level metrics
  stats.py      paired bootstrap, Holm, McNemar, corrected t-test
Supporting modules
  rulebased.py / rulebased_tuning.py  rule-based baseline (the earlier version of the system), used in H3c
  baselines.py  B1 regex, B2 banned phrases + embedding, B3 PromptGuard / PIGuard
  runtime.py, expkit.py, report.py, metrics.py, chunking.py, data/  shared utilities
"""

__version__ = "2.0.0"
