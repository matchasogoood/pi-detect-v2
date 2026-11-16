#!/usr/bin/env python3
"""H1 : behavior-based versus memoryless  (Section IV-D of the paper)

Question : does reading the session state catch more attacks, and catch them sooner, than a detector that judges
           each message on its own?
Compares : full (the whole system) vs ref_only (a reference but no history) vs memoryless (no state at all),
           plus baselines B1 / B2 / B3a / B3c, which are memoryless by design
Fairness : every method is thresholded so that at most h1.target_session_fpr of the benign val sessions raise an alarm
Criterion: the 95% paired-bootstrap interval of the session detection rate difference (full - memoryless) excludes 0

How to run (from the project root):
  python experiments/h1_behavior_vs_memoryless.py --smoke --data synthetic --embedder hashing   # check the code
  python experiments/h1_behavior_vs_memoryless.py                                               # the real run
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from pidetect.evaluate import (baseline_frame, baseline_threshold, records_frame,  # noqa: E402
                               session_ranking, session_table, summarize)
from pidetect.expkit import build_baselines, mean_ci_table, measure_cost, verdict, write_summary  # noqa: E402
from pidetect.fit import calibrate_session_threshold, refit_variant  # noqa: E402
from pidetect.metrics import target_reached  # noqa: E402
from pidetect.baselines import ExternalContentScore, ext_pipe_overrides  # noqa: E402
from pidetect.pipeline import CONTENT_FEATURES, FEATURES_NO_COUNTERS, REF_ONLY_FEATURES  # noqa: E402
from pidetect.progress import SeedProgress  # noqa: E402
from pidetect.report import save_df  # noqa: E402
from pidetect.runtime import add_common_args, build_full_system, fit_config, log, out_dir, setup, write_run_info  # noqa: E402
from pidetect.sessions import (benign_direct_session, bipia_session, boiling_frog_session,  # noqa: E402
                               direct_multiturn_session, kb_session, narrowing_followup_session,
                               security_article_session, tasktracker_session, topic_change_session)
from pidetect.stats import paired_bootstrap  # noqa: E402


def h1_sessions(sp, split: str, seed: int, n: int):
    """The H1 sessions: single-event and spread attacks, plus benign sessions of similar length"""
    rng = random.Random(seed * 31 + (1 if split == "val" else 2))
    kb = sp.kb[split]
    bip = sp.bipia[split]
    bip_atk = [a for v in bip["attacks"].values() for a in v]
    d_att = [d["text"] for d in sp.att[split]]
    d_ben = [d["text"] for d in sp.ben[split]] + list(sp.triggers[split])
    out = []
    for i in range(n):
        t = f"h1-{split}-{i}"
        # ---- single-event attacks ----
        if bip["contexts"] and bip_atk:
            out.append(bipia_session(rng.choice(bip["contexts"]), rng.choice(bip_atk), rng, t + "-bip"))
        if sp.tt.get(split):
            pos = [r for r in sp.tt[split] if r["label"] == 1]
            if pos:
                out.append(tasktracker_session(rng.choice(pos), t + "-tt"))
        if kb:
            c = rng.choice(kb)
            out.append(kb_session(c, rng, t + "-kb1", n_turns=rng.randint(1, 3),
                                  attack=rng.choice(rng.choice(kb)["attacks"]), attack_kind="single_chunk"))
        if d_ben and d_att:
            out.append(direct_multiturn_session(rng.sample(d_ben, min(3, len(d_ben))), rng.choice(d_att), rng,
                                                t + "-d1", kind="single"))
        # ---- spread attacks (these need the session history) ----
        if kb:
            c = rng.choice(kb)
            out.append(kb_session(c, rng, t + "-kbs", n_turns=rng.randint(1, 3),
                                  attack=rng.choice(rng.choice(kb)["attacks"]), attack_kind="split_chunks"))
            c = rng.choice(kb)
            out.append(boiling_frog_session(c, rng.choice([3, 5]), rng, t + "-bf", rng.choice(c["attacks"])))
        if d_ben and d_att:
            out.append(direct_multiturn_session(rng.sample(d_ben, min(3, len(d_ben))), rng.choice(d_att), rng,
                                                t + "-ds", kind="spread"))
        # ---- benign sessions (similar length) ----
        if kb:
            out.append(kb_session(rng.choice(kb), rng, t + "-kbb", n_turns=rng.randint(1, 3)))
            out.append(narrowing_followup_session(rng.choice(kb), rng, t + "-nar", n_followups=rng.randint(1, 3)))
            if len(kb) >= 2:
                a, b = rng.sample(kb, 2)
                out.append(topic_change_session(a, b, rng, t + "-tc"))
        if bip["contexts"]:
            out.append(bipia_session(rng.choice(bip["contexts"]), None, rng, t + "-bipb"))
        if sp.tt.get(split):
            neg = [r for r in sp.tt[split] if r["label"] == 0]
            if neg:
                out.append(tasktracker_session(rng.choice(neg), t + "-ttb"))
        if d_ben:
            out.append(benign_direct_session(rng.sample(d_ben, min(len(d_ben), rng.randint(2, 5))), t + "-db"))
        if sp.sec_instr[split] and sp.articles[split]:
            out.append(security_article_session(rng.choice(sp.sec_instr[split]),
                                                rng.sample(sp.articles[split], min(2, len(sp.articles[split]))),
                                                t + "-sec"))
    return out


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    cfg, world, emb = setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
    out = out_dir(args, "h1")
    h = cfg["h1"]
    fc = fit_config(cfg)
    frames, sess_frames, fit_rows, cost_rows = [], [], [], []
    # built once and reused across seeds: loading a guard model five times costs minutes, and its score cache is
    # what makes the external-content variants affordable
    bls = build_baselines(cfg, emb, h["baselines"], args.skip_baselines)
    ext_bls = {}
    for bname in h.get("ext_content", []):
        hit = [v for k, v in bls.items() if k.startswith(bname + "-") or k == bname]
        if hit:
            ext_bls[bname] = hit[0]
        else:
            got = build_baselines(cfg, emb, [bname], args.skip_baselines)
            if got:
                ext_bls[bname] = list(got.values())[0]
    prog = SeedProgress("H1", cfg["seeds"])
    for seed in cfg["seeds"]:
        prog.start(seed)
        det, sp, bank, va, vb, info = build_full_system(cfg, world, emb, seed)
        fit_rows.append(dict(seed=seed, **info))
        variants = [det,
                    refit_variant(det, "ref_only", vb, fc, seed, features=REF_ONLY_FEATURES),
                    refit_variant(det, "memoryless", vb, fc, seed, features=CONTENT_FEATURES)]
        # the simplified system: without u and k (H3d separated neither from noise) and without the stricter
        # anchor threshold tau_a (the weak-anchor case separated it from nothing either)
        if h.get("simplified", True):
            variants.append(refit_variant(det, "simplified", vb, fc, seed,
                                          features=FEATURES_NO_COUNTERS, use_trust=False))
        # the same correction applied to our own content score: a separate slope per channel for s and the two
        # lexical counts.  Reported so that the gain of the per-channel decision model can be read without the
        # public guard model, which the ext variants below also carry.
        if h.get("interactions", True):
            variants.append(refit_variant(det, "full(per-channel)", vb, fc, seed,
                                          interactions=("s", "n_imp", "n_ai")))
        # the session layer on top of a public classifier: the content score comes from the baseline's own model,
        # so whatever this variant adds over that baseline belongs to the session layer and not to our Stage 2
        for bname, bl in ext_bls.items():
            try:
                for tag in h.get("ext_modes_by", {}).get(bname, h.get("ext_modes", ["ext", "fused"])):
                    d2 = build_full_system(cfg, world, emb, seed, ext=ExternalContentScore.from_mode(bl, tag),
                                           **ext_pipe_overrides(tag))[0]
                    d2.name = f"full+{bname}({tag})"
                    variants.append(d2)
                    # additive ablation (followup/run_ablation.py only; absent from config.yaml): the same system with
                    # the session state removed, refitted the same way as the memoryless / ref_only variants above
                    for ab in h.get("ext_ablate", []):
                        feats = {"memoryless": CONTENT_FEATURES, "ref_only": REF_ONLY_FEATURES}[ab]
                        variants.append(refit_variant(d2, f"{d2.name}-{ab}", vb, fc, seed, features=feats))
            except Exception as exc:                                      # noqa: BLE001
                log(f"seed {seed}: external content score {bname} skipped ({type(exc).__name__}: {exc})")
        val_s = h1_sessions(sp, "val", seed, h["n_test_per_kind"])
        test_s = h1_sessions(sp, "test", seed, h["n_test_per_kind"])
        benign_val = [s for s in val_s if not s.is_attack]
        for d in variants:
            ci = calibrate_session_threshold(d, benign_val, h["target_session_fpr"])
            log(f"seed {seed}: {d.name}: t_blk={ci['t_blk']:.4f} val session FPR={ci['val_session_fpr']:.3f}"
                f"{'' if ci['feasible'] else '  (target not reachable: the fast path alone already exceeds it)'}")
            df = records_frame(d, test_s, seed)
            frames.append(df)
            # cost of this configuration, on the same test run (Section IV-B); never let it break the experiment
            try:
                cost_rows.append(dict(seed=seed, method=d.name,
                                      **measure_cost(d, test_s, n_events=40 if cfg.get("_smoke") else 150)))
            except Exception as exc:                                     # noqa: BLE001
                log(f"seed {seed}: cost measurement for {d.name} skipped ({type(exc).__name__}: {exc})")
        for name, bl in bls.items():
            thr = baseline_threshold(bl, val_s, h["target_session_fpr"], level="session")
            frames.append(baseline_frame(bl, test_s, thr, seed, name))
        log(f"seed {seed}: done ({len(test_s)} test sessions)")
        prog.done(seed)
    df = pd.concat(frames, ignore_index=True)
    st = session_table(df)
    df.to_csv(os.path.join(out, "events.csv.gz"), index=False, compression="gzip")
    st.to_csv(os.path.join(out, "sessions.csv"), index=False)
    save_df(pd.DataFrame(fit_rows), out, "fitted_parameters")
    if cost_rows:
        cost = pd.DataFrame(cost_rows).groupby("method").mean(numeric_only=True).reset_index().drop(columns="seed")
        save_df(cost, out, "cost", title="H1: cost per configuration (latency per event, memory per session)")
    else:
        cost = pd.DataFrame()

    # ---- summary ----
    per_seed = summarize(df, by=("method", "seed"))
    # did the method actually land on the common operating point?  A detector whose score saturates (a banned-term
    # list that matches benign text verbatim, a classifier that calls almost every benign message an attack) cannot
    # be placed at the target false-positive rate by any threshold, and then its detection rate is not comparable with
    # the others; we mark that rather than let a silent 0 look like a measurement (tolerance: within 25% of the
    # target, as in Jacob et al., 2025).
    per_seed["target_reached"] = [float(target_reached(f, h["target_session_fpr"])) for f in per_seed["session_fpr"]]
    overall = mean_ci_table(per_seed, ["method"],
                            ["session_detection", "session_fpr", "target_reached", "session_precision", "session_F1",
                             "delay_median", "delay_censored_mean", "TPR", "FPR", "Precision", "F1", "AUC", "ASR",
                             "untrusted_share_attacks"])
    # ranking at a common achieved false-positive rate (Section IV-B): immune to the threshold transfer
    rank = session_ranking(df)
    fprs = [c for c in rank.columns if c.startswith("det_at_fpr_")]
    save_df(mean_ci_table(rank, ["method"], ["session_AUC"] + fprs), out, "matched_fpr",
            title="H1: session-level ranking. session_AUC scores a session by its highest event score; "
                  "det_at_fpr_x is the detection rate when every method is placed at the same *achieved* "
                  "false-positive rate x on benign sessions, so that the comparison does not also compare how well "
                  "each threshold transferred from validation to test.")
    save_df(overall, out, "overall", title="H1: all attack kinds pooled (mean across seeds [95% CI]). "
                                          "delay_censored_mean charges a missed session the whole remaining horizon, "
                                          "so it cannot be lowered by detecting only the easy sessions. "
                                          "target_reached is the share of seeds whose session false-positive rate lands "
                                          f"within 25% of the {h['target_session_fpr']:.0%} target; where it is 0 the "
                                          "detection rate is not a rate at that operating point.")
    # the share of benign sessions that end up in the low-trust state (Section IV-A): among the benign sessions that
    # set a reference at all, the ones in which a later event is still judged with no trusted reference (z = 1)
    if {"x_z", "ref_updated"} <= set(df.columns):
        rows_lt = []
        for (method, seed, sid), g in df[df["label"] == 0].groupby(["method", "seed", "sid"], sort=False):
            g = g.sort_values("idx")
            first_ref = g.loc[g["ref_updated"] == 1, "idx"]
            if not len(first_ref):
                continue                                   # no reference was ever set: nothing to lose trust in
            later = g[g["idx"] > int(first_ref.min())]
            rows_lt.append(dict(method=method, seed=seed, sid=sid,
                                low_trust=int(len(later) > 0 and bool((later["x_z"] == 1).any()))))
        if rows_lt:
            lt = (pd.DataFrame(rows_lt).groupby(["method", "seed"])["low_trust"].mean()
                  .reset_index(name="low_trust_share"))
            save_df(mean_ci_table(lt, ["method"], ["low_trust_share"]), out, "low_trust_share",
                    title="H1: share of benign sessions that end up in the low-trust state -- of the benign sessions "
                          "that set a reference, the ones in which a later event is still judged with no trusted "
                          "reference (z = 1)")
    att = st[st["is_attack"] == 1]
    by_kind = att.groupby(["method", "se_attack_kind"]).agg(session_detection=("detected", "mean"),
                                                             delay_median=("delay", "median"),
                                                             n=("detected", "size")).reset_index()
    save_df(by_kind, out, "by_attack_kind", title="H1: by attack kind")
    by_src = att.groupby(["method", "se_subset"]).agg(session_detection=("detected", "mean"),
                                                      n=("detected", "size")).reset_index()
    save_df(by_src, out, "by_attack_source", title="H1: by attack source")

    # ---- the H1 criterion: paired bootstrap of detection (full - memoryless) on the same attack sessions ----
    tests = []
    for other in ("memoryless", "ref_only"):
        a = att[att["method"] == "full"].set_index(["seed", "sid"])["detected"]
        b = att[att["method"] == other].set_index(["seed", "sid"])["detected"]
        common = a.index.intersection(b.index)
        res = paired_bootstrap(a.loc[common].values, b.loc[common].values,
                               strata=[s for s, _ in common])
        tests.append(dict(comparison=f"full - {other}", **res))
        for kind in sorted(att["se_attack_kind"].dropna().unique()):
            ak = att[att["se_attack_kind"] == kind]
            a2 = ak[ak["method"] == "full"].set_index(["seed", "sid"])["detected"]
            b2 = ak[ak["method"] == other].set_index(["seed", "sid"])["detected"]
            c2 = a2.index.intersection(b2.index)
            if len(c2):
                r2 = paired_bootstrap(a2.loc[c2].values, b2.loc[c2].values)
                tests.append(dict(comparison=f"full - {other} [{kind}]", **r2))
    tests = pd.DataFrame(tests)
    save_df(tests, out, "h1_tests", title="H1: paired bootstrap of the session detection rate difference")
    main_t = tests[tests["comparison"] == "full - memoryless"].iloc[0]
    v = verdict(bool(main_t["excludes_zero"] and main_t["diff"] > 0),
                f"difference = {main_t['diff']:+.3f}, 95% CI [{main_t['lo']:+.3f}, {main_t['hi']:+.3f}]")
    baseline_note = (
        "B1 is a keyword/regex filter and has no threshold to set: a pattern match is a block, so its false-positive "
        "rate is whatever the patterns give and is usually above the target.\n"
        "B2 scores the maximum of a verbatim banned-phrase match (1.0) and the embedding similarity to a banned "
        "phrase. Benign text does contain such phrases (a security article quoting an attack), so more than the "
        "allowed share of benign sessions sit at the top of its score scale and no threshold reaches the target; "
        "target_reached is then 0 and the detection rate is not a rate at the common operating point.\n"
        "B3a (Prompt-Guard) is scored as its model card prescribes, which differs by channel: the jailbreak "
        "probability for messages written by the user, and the injection-or-jailbreak probability for third-party "
        "content. Scoring user messages with the stricter third-party rule flags almost every benign message and "
        "would understate the model.\n"
        "B3a and B3c read at most 512 tokens at a time, so longer texts are split into overlapping segments and "
        "scored segment by segment (the model cards ask for this); a text counts as high as its worst segment.")
    write_summary(out, "H1: Behavior-Based versus Memoryless Detection", [
        ("Verdict against the paper's criterion (Table IV)", v),
        ("Pooled results (mean across seeds [95% CI])", overall),
        ("Ranking at a common achieved false-positive rate", mean_ci_table(rank, ["method"], ["session_AUC"] + fprs)),
        ("How the baselines are configured", baseline_note),
        ("Differences (paired bootstrap)", tests),
        ("By attack kind", by_kind),
        ("By attack source", by_src),
        ("Cost per configuration", cost if len(cost) else "not measured")])
    write_run_info(out, cfg, world, emb, dict(experiment="H1"))
    log(f"results written to {out}  ->  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
