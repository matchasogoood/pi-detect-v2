#!/usr/bin/env python3
"""Secondary : component checks on Stages 1-2 (Section IV-G of the paper) -- exploratory, not a main claim

(i)   the soft rule flag vs a hard reject (a rule match blocks), and a fixed lambda from secondary.lambdas
(ii)  clip vs no clip, plus the rate of rule-tipped false positives (benign events blocked because the rule term pushed them over)
(iii) the gated fast path vs an ungated one vs none
(iv)  canonicalization on vs off, on the obfuscated-attack robustness set
plus cost: latency per event (mean, p95), memory per session, and embedding calls per event
Every comparison is reported with a confidence interval and a Holm-adjusted p-value

How to run:
  python experiments/s_component_checks.py --smoke --data synthetic --embedder hashing
  python experiments/s_component_checks.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pidetect.canon import obfuscate  # noqa: E402
from pidetect.evaluate import records_frame  # noqa: E402
from pidetect.fit import fit_decision, fit_stage2  # noqa: E402
from pidetect.expkit import measure_cost, verdict, write_summary  # noqa: E402
from pidetect.pipeline import ProposedDetector  # noqa: E402
from pidetect.progress import SeedProgress  # noqa: E402
from pidetect.report import save_df  # noqa: E402
from pidetect.runtime import (add_common_args, build_full_system, fit_config, log, out_dir, setup,  # noqa: E402
                              stage2_config, write_run_info)
from pidetect.schema import DIRECT, INDIRECT, BLOCK  # noqa: E402
from pidetect.sessions import (benign_direct_session, direct_multiturn_session, kb_session,  # noqa: E402
                               security_article_session, topic_change_session)
from pidetect.stats import holm, paired_bootstrap  # noqa: E402

OBF_KINDS = ("base64", "leet", "homoglyph", "spaced", "zerowidth")


def s_sessions(sp, seed: int, n_robust: int):
    rng = random.Random(seed * 97 + 3)
    out = []
    for i, d in enumerate(sp.att["test"]):
        out.append(direct_multiturn_session([], d["text"], rng, f"s-att-{i}", kind="single", subset="direct_attack"))
    for i, d in enumerate(sp.notinject["test"]):
        out.append(benign_direct_session([d["prompt"]], f"s-ni-{i}", subset="notinject"))
    for i, t in enumerate(sp.triggers["test"]):
        out.append(benign_direct_session([t], f"s-trg-{i}", subset="trigger_direct"))
    for i, d in enumerate(sp.ben["test"]):
        out.append(benign_direct_session([d["text"]], f"s-ben-{i}", subset="std_benign"))
    if sp.sec_instr["test"] and sp.articles["test"]:
        for i in range(max(10, len(sp.articles["test"]))):
            out.append(security_article_session(rng.choice(sp.sec_instr["test"]),
                                                rng.sample(sp.articles["test"], min(2, len(sp.articles["test"]))),
                                                f"s-sec-{i}"))
    kb = sp.kb["test"]
    if len(kb) >= 2:
        for i in range(30):
            out.append(topic_change_session(*rng.sample(kb, 2), rng, f"s-tc-{i}"))
    atk = [d["text"] for d in sp.att["test"]]
    for i in range(n_robust if atk else 0):
        kind = OBF_KINDS[i % len(OBF_KINDS)]
        a = obfuscate(rng.choice(atk), kind)
        s = direct_multiturn_session([], a, rng, f"s-rob-d-{i}", kind="single", subset="robust_direct")
        s.meta["obf"] = kind
        out.append(s)
        if kb:
            s2 = kb_session(rng.choice(kb), rng, f"s-rob-i-{i}", n_turns=1, attack=a, subset="robust_indirect")
            s2.meta["obf"] = kind
            out.append(s2)
    return out


def rule_tipped(det: ProposedDetector, df: pd.DataFrame) -> float:
    """Share of benign events blocked because of the rule term: removing that term would give p <= t_blk"""
    ben = df[(df["label"] == 0) & (df["blocked"] == 1) & (df["stage"] == "lr")]
    n_ben = int((df["label"] == 0).sum())
    if n_ben == 0:
        return np.nan
    tipped = 0
    for _, r in ben.iterrows():
        x = {k[2:]: r[k] for k in r.index if k.startswith("x_")}
        out = det.s2.process(r["text_full"], r["channel"])
        x["s"] = max(v.base for v in out.views)
        if det.predict_p(x) <= det.cfg.t_blk:
            tipped += 1
    return tipped / n_ben


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    cfg, world, emb = setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
    out = out_dir(args, "secondary")
    fc = fit_config(cfg)
    base_s2 = stage2_config(cfg)
    rows, cost_rows, pair_store = [], [], {}
    prog = SeedProgress("secondary", cfg["seeds"])
    for seed in cfg["seeds"]:
        prog.start(seed)
        det, sp, bank, va, vb, info = build_full_system(cfg, world, emb, seed)
        det.name = "soft (proposed)"
        variants = {"soft (proposed)": base_s2, "hard_reject": replace(base_s2, hard_reject=True),
                    "no_clip": replace(base_s2, clip=False), "fast_ungated": replace(base_s2, fast_path="ungated"),
                    "fast_off": replace(base_s2, fast_path="off"), "canon_off": replace(base_s2, use_canonical=False)}
        for lam in cfg["secondary"]["lambdas"]:
            variants[f"lambda={lam}"] = replace(base_s2, fixed_lambda=float(lam))
        test = s_sessions(sp, seed, cfg["secondary"]["n_robust"])
        texts = {(s.sid, i): e.text for s in test for i, e in enumerate(s.events)}
        for name, s2cfg in variants.items():
            if name == "soft (proposed)":
                d = det
            else:
                s2, _ = fit_stage2(emb, bank, None, s2cfg, va, seed, heads_from=det.s2)
                d = ProposedDetector(s2, replace(det.cfg), name)
                fit_decision(d, vb, fc, seed)
            df = records_frame(d, test, seed)
            df["text_full"] = [texts[(a, b)] for a, b in zip(df["sid"], df["idx"])]
            for sub, g in df.groupby("se_subset"):
                pos, neg = g[g["label"] == 1], g[g["label"] == 0]
                rows.append(dict(seed=seed, method=name, subset=sub, TPR=pos["blocked"].mean() if len(pos) else np.nan,
                                 FPR=neg["blocked"].mean() if len(neg) else np.nan, n_pos=len(pos), n_neg=len(neg)))
            hard_ben = df[(df["label"] == 0) & df["se_subset"].isin(["notinject", "trigger_direct", "security",
                                                                    "topic_change"])]
            rows.append(dict(seed=seed, method=name, subset="hard_benign_all", TPR=np.nan,
                             FPR=hard_ben["blocked"].mean(), n_pos=0, n_neg=len(hard_ben)))
            rows.append(dict(seed=seed, method=name, subset="rule_tipped_FP", TPR=np.nan, FPR=rule_tipped(d, df),
                             n_pos=0, n_neg=int((df["label"] == 0).sum())))
            pair_store.setdefault(name, []).append(hard_ben.set_index(["sid", "idx"])["blocked"].rename(seed))
            if name in ("soft (proposed)", "hard_reject", "fast_off"):
                c = measure_cost(d, test, n_events=60 if cfg.get("_smoke") else 200)
                cost_rows.append(dict(seed=seed, method=name, **c))
        log(f"seed {seed}: done")
        prog.done(seed)
    res = pd.DataFrame(rows)
    save_df(res, out, "per_seed")
    tab = res.groupby(["subset", "method"]).agg(TPR=("TPR", "mean"), FPR=("FPR", "mean"),
                                                n_pos=("n_pos", "sum"), n_neg=("n_neg", "sum")).reset_index()
    save_df(tab, out, "components", title="Secondary: TPR / FPR by subset and variant")
    # paired bootstrap of the FPR on hard benign text (variant - soft) + Holm
    tests, pv = [], {}
    ref = pd.concat(pair_store["soft (proposed)"], axis=1)
    for name, lst in pair_store.items():
        if name == "soft (proposed)":
            continue
        other = pd.concat(lst, axis=1)
        a, b, st = [], [], []
        for sd in ref.columns:
            j = ref[sd].dropna().index.intersection(other[sd].dropna().index)
            a += list(other.loc[j, sd].values)
            b += list(ref.loc[j, sd].values)
            st += [sd] * len(j)
        r = paired_bootstrap(a, b, strata=st)
        pv[name] = r["p"]
        tests.append(dict(comparison=f"FPR hard benign: {name} - soft", **r))
    tests = pd.DataFrame(tests)
    if not tests.empty:
        adj = holm(pv)
        tests["p_holm"] = [adj[c.split(": ")[1].split(" - ")[0]] for c in tests["comparison"]]
    save_df(tests, out, "secondary_tests", title="Secondary: FPR difference on hard benign text (Holm-adjusted)")
    cost = pd.DataFrame(cost_rows).groupby("method").mean(numeric_only=True).reset_index().drop(columns="seed")
    save_df(cost, out, "cost", title="Cost: latency / memory / texts embedded per event")
    write_summary(out, "Secondary: Stage 1-2 Component Checks (exploratory)", [
        ("Note", "This section is not a main claim of the paper -- report it as exploratory, with the Holm correction"),
        ("TPR / FPR", tab), ("FPR difference on hard benign text", tests), ("Cost", cost)])
    write_run_info(out, cfg, world, emb, dict(experiment="Secondary"))
    log(f"results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
