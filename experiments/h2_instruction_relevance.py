#!/usr/bin/env python3
"""H2 : instruction relevance on retrieved content  (Section IV-E of the paper)

Question : does comparing the most instruction-like window (omega*) of a chunk with the user's task (Stage 3)
           catch indirect injections that Stages 1-2 miss, and is the window better than the whole chunk?
Compares : S12 (Stages 1+2 only) | S12+r_chunk (relevance of the whole chunk) | S12+r_window (Stage 3) | full
         + baseline B2, B3a, B3c
Test subsets : bipia (instructions embedded in documents), tasktracker, kb_buried (instructions hidden in long documents),
               split_windows (the instruction split across windows -- a failure case), ontopic (PoisonedRAG style -- a failure case)
Threshold : set on val so that the indirect-event FPR is h2.headline_fpr (5%), separately for each method.
            The same run is repeated at the 1% target, which is only estimable with enough benign events: the table reports, for every method, the achieved FPR on test, the number of
            benign events behind the threshold, and whether the target was reached within 25% (Jacob et al., 2025).
Criterion : the 95% interval of the TPR difference (S12+r_window - S12) on the main subsets (bipia + tasktracker + kb_buried) excludes 0
            at the 5% headline target (Section IV-A); the same interval at the 1% target is reported next to it

How to run:
  python experiments/h2_instruction_relevance.py --smoke --data synthetic --embedder hashing
  python experiments/h2_instruction_relevance.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pidetect.evaluate import baseline_frame, baseline_threshold, records_frame  # noqa: E402
from pidetect.baselines import ExternalContentScore, ext_pipe_overrides  # noqa: E402
from pidetect.expkit import build_baselines, mean_ci_table, verdict, write_summary  # noqa: E402
from pidetect.fit import refit_variant  # noqa: E402
from pidetect.metrics import auc_score, expected_fp, target_reached, threshold_for_fpr  # noqa: E402
from pidetect.pipeline import CONTENT_FEATURES  # noqa: E402
from pidetect.progress import SeedProgress  # noqa: E402
from pidetect.report import save_df  # noqa: E402
from pidetect.runtime import add_common_args, build_full_system, fit_config, log, out_dir, setup, write_run_info  # noqa: E402
from pidetect.schema import INDIRECT  # noqa: E402
from pidetect.sessions import bipia_session, kb_session, ontopic_attack, split_attack, tasktracker_session  # noqa: E402
from pidetect.stats import paired_bootstrap  # noqa: E402

MAIN_SUBSETS = ("bipia", "tasktracker", "kb_buried")


def h2_sessions(sp, split: str, seed: int, n: int):
    rng = random.Random(seed * 53 + (1 if split == "val" else 2))
    kb, bip = sp.kb[split], sp.bipia[split]
    bip_atk = [a for v in bip["attacks"].values() for a in v]
    out = []
    for i in range(n):
        t = f"h2-{split}-{i}"
        atk = rng.random() < 0.5
        if bip["contexts"] and bip_atk:
            out.append(bipia_session(rng.choice(bip["contexts"]), rng.choice(bip_atk) if atk else None, rng,
                                     t + "-bip", inject="inline"))
        if sp.tt.get(split):
            out.append(tasktracker_session(rng.choice(sp.tt[split]), t + "-tt"))
        if kb:
            c = rng.choice(kb)
            a = rng.choice(rng.choice([x for x in kb if x["id"] != c["id"]] or kb)["attacks"]) if atk else None
            s = kb_session(c, rng, t + "-kbb", n_turns=1, docs_per_turn=(2, 3), attack=a, attack_kind="buried",
                           long_doc=True, subset="kb_buried")
            out.append(s)
            # the instruction split into two distant pieces (a failure case the paper acknowledges)
            if atk:
                s2 = kb_session(c, rng, t + "-spw", n_turns=1, docs_per_turn=(3, 3), long_doc=True,
                                subset="split_windows")
                e = s2.events[-1]
                sents = e.text.split(". ")
                parts = split_attack(a, 2)
                if len(parts) == 2 and len(sents) > 4:
                    sents.insert(1, parts[0].rstrip("."))
                    sents.insert(len(sents) - 1, parts[1].rstrip("."))
                    e.text, e.label = ". ".join(sents), 1
                    e.meta.update(role="attack", phase="attack")
                    s2.meta["attack_kind"] = "split_windows"
                    out.append(s2)
            # an on-topic attack (PoisonedRAG style)
            out.append(kb_session(c, rng, t + "-ont", n_turns=1, attack=ontopic_attack(c, rng) if atk else None,
                                  attack_kind="single_chunk", subset="ontopic"))
    return out


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    cfg, world, emb = setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
    out = out_dir(args, "h2")
    h = cfg["h2"]
    fc = fit_config(cfg)
    target = float(h.get("headline_fpr", 0.05))      # the headline target of Section IV-A (1% is under-powered)
    targets = sorted({target, 0.01})                 # ... and the 1% figure, reported next to it
    rows, paired, frames = [], [], []
    # loaded once: a guard model takes minutes to load, and its score cache serves every seed
    bls = build_baselines(cfg, emb, h["baselines"], args.skip_baselines)
    ext_bls = {}
    for bname in h.get("ext_content", []):
        hit = [v for k, v in bls.items() if k.startswith(bname + "-") or k == bname]
        if hit:
            ext_bls[bname] = hit[0]
    prog = SeedProgress("H2", cfg["seeds"])
    for seed in cfg["seeds"]:
        prog.start(seed)
        det, sp, bank, va, vb, info = build_full_system(cfg, world, emb, seed)
        variants = [refit_variant(det, "S12", vb, fc, seed, features=CONTENT_FEATURES),
                    refit_variant(det, "S12+r_chunk", vb, fc, seed, features=CONTENT_FEATURES + ("r", "z"),
                                  relevance_unit="chunk"),
                    refit_variant(det, "S12+r_window", vb, fc, seed, features=CONTENT_FEATURES + ("r", "z"),
                                  relevance_unit="window"),
                    det,
                    refit_variant(det, "full(per-channel)", vb, fc, seed,
                                  interactions=("s", "n_imp", "n_ai"))]
        # the same pipeline with a public guard model as its content score (see H1), to see what the window
        # relevance adds on top of the strongest per-message detector
        for bname, bl in ext_bls.items():
            for tag in h.get("ext_modes", ["ext_win"]):
                d2 = build_full_system(cfg, world, emb, seed, ext=ExternalContentScore.from_mode(bl, tag),
                                       **ext_pipe_overrides(tag))[0]
                d2.name = f"full+{bname}({tag})"
                variants.append(d2)
        val_s = h2_sessions(sp, "val", seed, h["n_test_per_subset"])
        test_s = h2_sessions(sp, "test", seed, h["n_test_per_subset"])
        val_raw = {d.name: records_frame(d, val_s, seed) for d in variants}
        test_raw = {d.name: records_frame(d, test_s, seed) for d in variants}
        for tgt in targets:
            scored = {}
            for d in variants:
                vdf = val_raw[d.name]
                vdf = vdf[vdf["channel"] == INDIRECT]
                neg_val = vdf.loc[vdf["label"] == 0, "score"].values
                thr = threshold_for_fpr(neg_val, tgt)
                tdf = test_raw[d.name]
                tdf = tdf[tdf["channel"] == INDIRECT].copy()
                tdf["hit"] = (tdf["score"] > thr).astype(int)
                tdf["thr"] = thr
                tdf["n_neg_val"] = len(neg_val)
                scored[d.name] = tdf
            for name, bl in bls.items():
                thr = baseline_threshold(bl, val_s, tgt, level="event", channels=[INDIRECT])
                tdf = baseline_frame(bl, test_s, thr, seed, name)
                tdf = tdf[tdf["channel"] == INDIRECT].copy()
                tdf["hit"] = (tdf["score"] > thr).astype(int)
                tdf["n_neg_val"] = np.nan
                scored[name] = tdf
            for name, tdf in scored.items():
                if tgt == target:
                    frames.append(tdf.assign(method=name, target_fpr=tgt))
                for sub, g in [("main", tdf[tdf["se_subset"].isin(MAIN_SUBSETS)])] + list(tdf.groupby("se_subset")):
                    pos, neg = g[g["label"] == 1], g[g["label"] == 0]
                    fpr = neg["hit"].mean() if len(neg) else np.nan
                    rows.append(dict(seed=seed, method=name, subset=sub, target_fpr=tgt,
                                     TPR=pos["hit"].mean() if len(pos) else np.nan, FPR=fpr,
                                     AUC=auc_score(g["label"].values, g["score"].values), n_pos=len(pos),
                                     n_neg=len(neg), target_reached=int(target_reached(fpr, tgt)),
                                     n_neg_val=float(tdf["n_neg_val"].iloc[0]) if len(tdf) else np.nan,
                                     **{k: v for k, v in expected_fp(len(neg), tgt).items() if k != "n_neg"}))
            # paired: TPR over the same attack events (aligned across every variant of our system)
            key = ["sid", "idx"]
            for a_name, b_name in [("S12+r_window", "S12"), ("S12+r_window", "S12+r_chunk"), ("full", "S12")]:
                A = scored[a_name].set_index(key)
                B = scored[b_name].set_index(key)
                for sub in ("main",) + tuple(sorted(A["se_subset"].unique())):
                    m = A["label"] == 1
                    m &= A["se_subset"].isin(MAIN_SUBSETS) if sub == "main" else (A["se_subset"] == sub)
                    idx = A.index[m].intersection(B.index)
                    if len(idx):
                        paired.append(dict(seed=seed, comparison=f"{a_name} - {b_name}", subset=sub, target_fpr=tgt,
                                           a=A.loc[idx, "hit"].values, b=B.loc[idx, "hit"].values))
        log(f"seed {seed}: done")
        prog.done(seed)
    res = pd.DataFrame(rows)
    save_df(res, out, "per_seed")
    table = mean_ci_table(res, ["target_fpr", "subset", "method"],
                          ["TPR", "FPR", "AUC", "target_reached", "expected_fp", "rel_se"])
    save_df(table, out, "tpr_at_fpr",
            title="H2: TPR at a threshold calibrated on validation for each target FPR, by subset. FPR is the rate "
                  "actually achieved on test; target_reached = 1 when it lands within 25% of the target; expected_fp "
                  "and rel_se say how precisely that target can be estimated from the benign events available.")
    tests = []
    for (comp, sub, tgt), _ in pd.DataFrame(paired).groupby(["comparison", "subset", "target_fpr"]) if paired else []:
        P = [p for p in paired if p["comparison"] == comp and p["subset"] == sub and p["target_fpr"] == tgt]
        a = np.concatenate([p["a"] for p in P])
        b = np.concatenate([p["b"] for p in P])
        st = np.concatenate([[p["seed"]] * len(p["a"]) for p in P])
        tests.append(dict(comparison=comp, subset=sub, target_fpr=tgt, **paired_bootstrap(a, b, strata=st)))
    tests = pd.DataFrame(tests)
    save_df(tests, out, "h2_tests", title="H2: paired bootstrap of the TPR difference")
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(out, "events.csv.gz"), index=False, compression="gzip")
    main_t = tests[(tests["comparison"] == "S12+r_window - S12") & (tests["subset"] == "main") &
                   (tests["target_fpr"] == target)]
    if len(main_t):
        mt = main_t.iloc[0]
        v = verdict(bool(mt["excludes_zero"] and mt["diff"] > 0),
                    f"TPR gain = {mt['diff']:+.3f}, 95% CI [{mt['lo']:+.3f}, {mt['hi']:+.3f}] "
                    f"(main subsets, {target:.0%} target)")
        for tgt in targets:
            if tgt == target:
                continue
            o = tests[(tests["comparison"] == "S12+r_window - S12") & (tests["subset"] == "main") &
                      (tests["target_fpr"] == tgt)]
            if len(o):
                v += (f"  |  at the {tgt:.0%} target: {o.iloc[0]['diff']:+.3f} "
                      f"[{o.iloc[0]['lo']:+.3f}, {o.iloc[0]['hi']:+.3f}]")
    else:
        v = "not enough data for the main criterion"
    write_summary(out, "H2: Instruction Relevance on Retrieved Content", [
        ("Verdict against the paper's criterion (Table IV)", v),
        ("TPR by target FPR and subset", table),
        ("Was the operating point reached?",
         "A threshold set on validation only transfers if the FPR measured on test stays near the target; we accept "
         "it within 25% (Jacob et al., 2025). Where target_reached = 0 the TPR is not a TPR at that FPR, and the "
         "achieved FPR column is the number to read. With n benign events a target FPR of nu can only be estimated "
         "to about 1/sqrt(n*nu) relative error, which is why Section IV-A falls back to 5% when 1% is out of reach."),
        ("Differences (paired bootstrap)", tests),
        ("Note", "ontopic and split_windows are the failure cases the paper expects Stage 3 to miss; report them as they are.\n"
                 "The numbers of Abdelnabi et al. (white-box) belong in the discussion only, never in this table.")])
    write_run_info(out, cfg, world, emb, dict(experiment="H2"))
    log(f"results written to {out}  ->  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
