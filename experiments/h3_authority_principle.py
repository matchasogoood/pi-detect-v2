#!/usr/bin/env python3
"""H3 : the authority principle under topic changes, drift and probing  (Section IV-F of the paper)

H3a  genuine topic changes and narrowing follow-ups (all benign)  -> session false-positive rate
     plus the error rate of the follow-up / new-task rule (IsTask), its sensitivity to tau_same, and whether a
     false block cascades through u
H3b  boiling frog: N slightly off-topic chunks (label 0) followed by the attack (label 1)  -> ASR, detection delay
     plus turning off o, fbar and delta one at a time, and the weak case (a direct attack that becomes the anchor):
     with and without tau_a, and dropping the reference
H3c  a threshold-aware attacker: chunks whose relevance sits just above tau_o -> full system vs the rule-based Stage 4
H3d  probing (retyping a rephrased attack up to 10 times) and forced resets -> with and without u and k
Designs compared : full | no_stage4 | accumulated (a running profile = the earlier system) | channel_agnostic (no authority principle)
Criterion : the 95% interval of the false-positive reduction FPR(accumulated) - FPR(full) in H3a is > 0, and the ASR reduction
            ASR(accumulated) - ASR(full) in H3b is >= 0
Sign      : every difference in the tests is oriented so that a POSITIVE value means the full system is better
            (lower false-positive rate, lower attack success rate)

How to run:
  python experiments/h3_authority_principle.py --smoke --data synthetic --embedder hashing
  python experiments/h3_authority_principle.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pidetect.evaluate import records_frame, session_table  # noqa: E402
from pidetect.expkit import mean_ci_table, verdict, write_summary  # noqa: E402
from pidetect.fit import calibrate_session_threshold, refit_variant  # noqa: E402
from pidetect.pipeline import FEATURES, STAGE4_IND  # noqa: E402
from pidetect.progress import SeedProgress  # noqa: E402
from pidetect.report import save_df  # noqa: E402
from pidetect.rulebased import Detector as RuleDetector  # noqa: E402
from pidetect.rulebased_tuning import finalize as rule_finalize  # noqa: E402
from pidetect.runtime import add_common_args, build_full_system, fit_config, log, out_dir, setup, write_run_info  # noqa: E402
from pidetect.schema import DIRECT, INDIRECT, Theta  # noqa: E402
from pidetect.sessions import (boiling_frog_session, fitting_sessions, forced_reset_session,  # noqa: E402
                               narrowing_followup_session, probing_session, topic_change_session,
                               weak_case_sessions)
from pidetect.stats import paired_bootstrap  # noqa: E402


def without(*drop):
    return tuple(f for f in FEATURES if f not in drop)


def same_model(base, name, **kw):
    """Variants that change the policy only (not the features) -> they reuse the same LR"""
    d = base.variant(name, **kw)
    d.model, d.medians = base.model, base.medians
    return d


def tau_same_errors(det, sessions):
    """Error rate of the follow-up / new-task rule against the true role, plus its sensitivity to tau_same"""
    pairs = []
    for s in sessions:
        R = None
        for e in s.events:
            if e.channel != DIRECT:
                continue
            vec = det.s2.process(e.text, e.channel).raw.e
            role = e.meta.get("role")
            if R is not None and role in ("followup", "switch"):
                from pidetect.pipeline import has_content
                pairs.append(dict(cos=float(np.dot(vec, R)), is_new=int(role == "switch"),
                                  content=int(has_content(e.text))))
            if role in ("instruction", "switch") or R is None:
                R = vec
    P = pd.DataFrame(pairs)
    rows = []
    if P.empty:
        return P
    grid = sorted(set([det.cfg.tau_same] + list(np.quantile(P["cos"], [0.1, 0.25, 0.5, 0.75, 0.9]))))
    for t in grid:
        pred_new = (P["content"] == 1) & (P["cos"] < t)
        fu = P[P["is_new"] == 0]
        nw = P[P["is_new"] == 1]
        rows.append(dict(tau_same=t, chosen=int(abs(t - det.cfg.tau_same) < 1e-12),
                         followup_as_new=float(pred_new[fu.index].mean()) if len(fu) else np.nan,
                         new_as_followup=float((~pred_new[nw.index]).mean()) if len(nw) else np.nan,
                         n_followup=len(fu), n_new=len(nw)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--no-session-calibration", action="store_true",
                    help="do not threshold every design to the same benign session false-positive rate "
                         "(each design then sits at its own session operating point)")
    args = ap.parse_args()
    cfg, world, emb = setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
    out = out_dir(args, "h3")
    h = cfg["h3"]
    fc = fit_config(cfg)
    A, B, C, D, W, TS = [], [], [], [], [], []
    H3A_EVENTS = []                                  # every H3a event of every design (to see which events raise the false positives)
    prog = SeedProgress("H3", cfg["seeds"])
    for seed in cfg["seeds"]:
        prog.start(seed)
        rng = random.Random(seed * 71 + 5)
        det, sp, bank, va, vb, info = build_full_system(cfg, world, emb, seed)
        kb = sp.kb["test"]
        # Common operating point. fit_decision gives every design the same EVENT-level benign FPR (nu) on the
        # fitting mix, but H3a reports a SESSION-level false-positive rate and H3b an attack success rate, and both
        # move together with t_blk: a design that blocks more then shows a lower ASR and a higher false-positive rate
        # at the same time, so the two numbers are one dial, not two findings. H1 already avoids this by
        # thresholding every method to the same benign session false-positive rate, and H3 now does the same.
        # The threshold is set on the ORDINARY benign sessions of the fitting mix, never on the topic-change and
        # narrowing sessions that H3a measures, because those are the contrast the hypothesis is about.
        # --no-session-calibration keeps the older behavior (each design at its own session operating point).
        cal_benign: list = []
        if h.get("target_session_fpr") and not args.no_session_calibration:
            cal_benign = [x for x in list(va) + list(vb) if not x.is_attack][: int(h.get("n_cal", 60))]

        def align(d):
            """Thresholds a design to the shared benign session false-positive rate (see the note above)"""
            if not cal_benign:
                return d
            before = d.cfg.t_blk
            ci = calibrate_session_threshold(d, cal_benign, h["target_session_fpr"])
            log(f"seed {seed}: {d.name}: t_blk {before:.4f} -> {ci['t_blk']:.4f}, "
                f"val session FPR={ci['val_session_fpr']:.3f} on {len(cal_benign)} benign sessions"
                f"{'' if ci['feasible'] else '  (target not reachable: the fast path alone already exceeds it)'}")
            return d

        designs = {
            "full": det,
            "no_stage4": refit_variant(det, "no_stage4", vb, fc, seed, features=without(*STAGE4_IND)),
            "accumulated": refit_variant(det, "accumulated", vb, fc, seed, reference_mode="accumulated"),
            "channel_agnostic": refit_variant(det, "channel_agnostic", vb, fc, seed, channel_agnostic=True),
        }
        # Reference refinement: a direct follow-up the pipeline forwarded nudges R towards itself.  The authority
        # principle is intact (indirect content still cannot move R), and this is the design answer to the cost
        # H3a measures, namely that a fixed reference keeps pointing at the broad original question while the
        # user narrows it.  Every value of gamma on the sweep is a separate design, chosen on validation.
        for g in h.get("gamma_follow", []):
            if float(g) > 0:
                designs[f"refine_gamma{g:g}"] = refit_variant(det, f"refine_gamma{g:g}", vb, fc, seed,
                                                              gamma_follow=float(g))
        designs = {k: align(v) for k, v in designs.items()}
        # ---------------- H3a ----------------
        h3a = [topic_change_session(*rng.sample(kb, 2), rng, f"h3a-tc-{seed}-{i}") for i in range(h["n_topic_change"])]
        h3a += [narrowing_followup_session(rng.choice(kb), rng, f"h3a-nar-{seed}-{i}", n_followups=rng.randint(2, 4))
                for i in range(h["n_narrowing"])]
        for name, d in designs.items():
            df = records_frame(d, h3a, seed)
            H3A_EVENTS.append(df.assign(method=name))
            st = session_table(df)
            casc_sids = set(df[(df["blocked"] == 1) & (df["x_u"] > 0)]["sid"])
            cascade = len(casc_sids)
            for _, r in st.iterrows():
                A.append(dict(seed=seed, method=name, sid=r["sid"], subset=r["se_subset"], false_positive=r["any_block"],
                              cascade=int(r["sid"] in casc_sids)))
            log(f"seed {seed}: H3a {name}: session FPR={st['any_block'].mean():.3f} "
                f"(sessions blocked again because of u: {cascade})")
        ts = tau_same_errors(det, h3a)
        if not ts.empty:
            TS.append(ts.assign(seed=seed))
        # ---------------- H3b ----------------
        bf = {}
        for N in h["Ns"]:
            bf[N] = [boiling_frog_session(c, N, rng, f"h3b-{seed}-{N}-{i}", rng.choice(c["attacks"]))
                     for i, c in enumerate([rng.choice(kb) for _ in range(h["n_per_N"])])]
        abl = {"minus_o": align(refit_variant(det, "minus_o", vb, fc, seed, features=without("o"))),
               "minus_fbar": align(refit_variant(det, "minus_fbar", vb, fc, seed, features=without("fbar"))),
               "minus_delta": align(refit_variant(det, "minus_delta", vb, fc, seed, features=without("delta")))}
        for name, d in {**designs, **abl}.items():
            for N, sess in bf.items():
                st = session_table(records_frame(d, sess, seed))
                for _, r in st.iterrows():
                    B.append(dict(seed=seed, method=name, N=N, sid=r["sid"], detected=r["detected"],
                                  asr=1 - r["detected"], delay=r["delay"], early_block=r["early_block"]))
        # the weak case: a direct attack that becomes the anchor
        atk = [d_["text"] for d_ in sp.att["test"]][: h["n_weak"]]
        if len(atk) >= 2:
            E = det.s2.emb.encode(atk)
            partners = [atk[int(np.argsort(-(E @ E[i]))[1])] for i in range(len(atk))]
            ctrl_instr = [x for c in kb for x in c["instructions"]]
            weak, ctrl = weak_case_sessions(atk, partners, ctrl_instr, f"h3w-{seed}")
            for name, d in {"full (low-trust anchor)": det,
                            "no_tau_a (always trusted)": same_model(det, "no_tau_a", use_trust=False),
                            "drop_low_trust": same_model(det, "drop_low_trust", drop_low_trust=True)}.items():
                for tag, sess in (("weak", weak), ("control", ctrl)):
                    df = records_frame(d, sess, seed)
                    part = df[df["role"] == "partner"]
                    anc = df[df["role"] == "anchor"]
                    W.append(dict(seed=seed, method=name, case=tag, partner_ASR=1 - part["blocked"].mean(),
                                  partner_untrusted=part["untrusted"].mean(),
                                  anchor_blocked=anc["blocked"].mean() if len(anc) else np.nan))
        # ---------------- H3c ----------------
        tau_o = det.cfg.tau_o

        def pick_full(instr, bridges, N):
            e_i = det.s2.process(instr, DIRECT).raw.e
            r = [float(np.dot(det.s2.process(b, INDIRECT).raw.e_top, e_i)) for b in bridges]
            order = np.argsort(r)
            above = [bridges[j] for j in order if r[j] >= tau_o][:N]
            return above + [bridges[j] for j in order[::-1] if bridges[j] not in above][: N - len(above)]

        rule = RuleDetector(emb, bank, Theta(), name="rule_based")
        rule_val = fitting_sessions(sp, "val", seed, cfg["fit"]["n_fit_sessions"])
        rule, rinfo = rule_finalize(rule, rule_val, fc.nu, verbose=False)
        th = rule.theta

        def pick_rule(instr, bridges, N):
            e_i = emb.encode([instr])[0]
            r = list(emb.encode(bridges) @ e_i)
            order = np.argsort(r)
            lo = max(th.tau_o if math.isfinite(th.tau_o) else -1, th.tau_r if math.isfinite(th.tau_r) else -1)
            above = [bridges[j] for j in order if r[j] >= lo][:N]
            return above + [bridges[j] for j in order[::-1] if bridges[j] not in above][: N - len(above)]

        for N in h["Ns"]:
            cs = [rng.choice(kb) for _ in range(h["n_per_N"])]
            s_full = [boiling_frog_session(c, N, rng, f"h3c-f-{seed}-{N}-{i}", rng.choice(c["attacks"]),
                                           bridge_pick=pick_full) for i, c in enumerate(cs)]
            s_rule = [boiling_frog_session(c, N, rng, f"h3c-r-{seed}-{N}-{i}", rng.choice(c["attacks"]),
                                           bridge_pick=pick_rule) for i, c in enumerate(cs)]
            st = session_table(records_frame(det, s_full, seed))
            C.append(dict(seed=seed, method="full (LR)", N=N, detection=st["detected"].mean(),
                          delay_median=st["delay"].median()))
            dets = []
            for s in s_rule:
                decs = rule.run_session(s)
                dets.append(int(any(dd.verdict == 1 and e.label == 1 for dd, e in zip(decs, s.events))))
            C.append(dict(seed=seed, method="rule-based Stage 4", N=N, detection=float(np.mean(dets)),
                          delay_median=np.nan))
        # ---------------- H3d ----------------
        no_uk = align(refit_variant(det, "no_u_k", vb, fc, seed, features=without("u", "k")))
        atk_all = [d_["text"] for d_ in sp.att["test"]]
        ben = [d_["text"] for d_ in sp.ben["test"]]
        probes = []
        if len(atk_all) > 2:
            E = det.s2.emb.encode(atk_all)
            for i in range(h["n_probing"]):
                j0 = rng.randrange(len(atk_all))
                nn = list(np.argsort(-(E @ E[j0])))[: h["max_attempts"]]
                probes.append(probing_session([atk_all[j] for j in nn], [rng.choice(ben)] if ben else [],
                                              f"h3d-p-{seed}-{i}"))
        resets = [forced_reset_session(c, rng.sample([x for x in kb if x["id"] != c["id"]], min(3, len(kb) - 1)),
                                       rng.choice(c["attacks"]), rng, f"h3d-r-{seed}-{i}")
                  for i, c in enumerate([rng.choice(kb) for _ in range(h["n_reset"])])]
        for name, d in (("full", det), ("no_u_k", no_uk)):
            if probes:
                df = records_frame(d, probes, seed)
                at = df[df["role"] == "attack"].copy()
                at["attempt"] = at["ev_attempt"].astype(int)
                for j, g in at.groupby("attempt"):
                    first_block = at[at["blocked"] == 1].groupby("sid")["attempt"].min()
                    D.append(dict(seed=seed, method=name, scenario="probing", attempt=int(j),
                                  block_rate=g["blocked"].mean(),
                                  blocked_by_attempt=float((first_block <= j).sum() / at["sid"].nunique())))
            df = records_frame(d, resets, seed)
            fin = df[df["label"] == 1]
            D.append(dict(seed=seed, method=name, scenario="forced_reset", attempt=np.nan,
                          block_rate=fin["blocked"].mean(), blocked_by_attempt=np.nan))
        log(f"seed {seed}: done")
        prog.done(seed)

    A, B, C, D, W = (pd.DataFrame(x) for x in (A, B, C, D, W))
    if H3A_EVENTS:
        pd.concat(H3A_EVENTS, ignore_index=True).to_csv(os.path.join(out, "h3a_events.csv.gz"), index=False,
                                                         compression="gzip")
    # ---- tables ----
    a_tab = A.groupby(["method", "subset"])["false_positive"].mean().unstack().reset_index()
    a_tab["all"] = A.groupby("method")["false_positive"].mean().reindex(a_tab["method"]).values
    a_tab["cascade_via_u"] = A.groupby("method")["cascade"].mean().reindex(a_tab["method"]).values
    save_df(a_tab, out, "h3a_false_positives", title="H3a: session false-positive rate (benign sessions with at least one block)")
    ts = pd.concat(TS, ignore_index=True) if TS else pd.DataFrame()
    if not ts.empty:
        save_df(ts, out, "h3a_tau_same_sensitivity", title="H3a: error rate of the follow-up / new-task rule against tau_same")
    b_tab = B.groupby(["method", "N"]).agg(ASR=("asr", "mean"), detection=("detected", "mean"),
                                           delay_median=("delay", "median"),
                                           early_block=("early_block", "mean")).reset_index()
    save_df(b_tab, out, "h3b_boiling_frog", title="H3b: boiling frog (ASR = attack chunks not blocked)")
    save_df(W.groupby(["method", "case"]).mean(numeric_only=True).reset_index().drop(columns="seed"), out,
            "h3b_weak_anchor", title="H3b: the weak case -- a direct attack that becomes the anchor")
    save_df(C.groupby(["method", "N"]).mean(numeric_only=True).reset_index().drop(columns="seed"), out,
            "h3c_threshold_aware", title="H3c: a threshold-aware attacker (detection rate of the attack chunk)")
    if not D.empty:
        save_df(D.groupby(["scenario", "method", "attempt"], dropna=False).mean(numeric_only=True).reset_index()
                .drop(columns="seed"), out, "h3d_probing_reset", title="H3d: probing and forced resets")
    # ---- criteria ----
    tests = []
    a1 = A[A["method"] == "accumulated"].set_index(["seed", "sid"])["false_positive"]
    a0 = A[A["method"] == "full"].set_index(["seed", "sid"])["false_positive"]
    idx = a1.index.intersection(a0.index)
    t_fpr = paired_bootstrap(a1.loc[idx].values, a0.loc[idx].values, strata=[s for s, _ in idx])
    tests.append(dict(comparison="H3a: FPR reduction vs accumulated = FPR(accumulated) - FPR(full)", **t_fpr))
    t_ca = A[A["method"] == "channel_agnostic"].set_index(["seed", "sid"])["false_positive"]
    idx2 = t_ca.index.intersection(a0.index)
    tests.append(dict(comparison="H3a: FPR reduction vs channel_agnostic = FPR(channel_agnostic) - FPR(full)",
                      **paired_bootstrap(t_ca.loc[idx2].values, a0.loc[idx2].values, strata=[s for s, _ in idx2])))
    b1 = B[B["method"] == "full"].set_index(["seed", "N", "sid"])["asr"]
    b0 = B[B["method"] == "accumulated"].set_index(["seed", "N", "sid"])["asr"]
    idx3 = b1.index.intersection(b0.index)
    # ASR reduction = ASR(other) - ASR(full): positive = the full system lets fewer attacks through
    t_asr = paired_bootstrap(b0.loc[idx3].values, b1.loc[idx3].values, strata=[s for s, _, _ in idx3])
    tests.append(dict(comparison="H3b: ASR reduction vs accumulated = ASR(accumulated) - ASR(full)", **t_asr))
    b2 = B[B["method"] == "no_stage4"].set_index(["seed", "N", "sid"])["asr"]
    idx4 = b1.index.intersection(b2.index)
    tests.append(dict(comparison="H3b: ASR reduction vs no_stage4 = ASR(no_stage4) - ASR(full)",
                      **paired_bootstrap(b2.loc[idx4].values, b1.loc[idx4].values, strata=[s for s, _, _ in idx4])))
    # every reference-refinement design, against the same baseline the hypothesis names, so that the sweep is
    # reported whole and not only at its best point
    for m in sorted(x for x in A["method"].unique() if str(x).startswith("refine_gamma")):
        g1 = A[A["method"] == m].set_index(["seed", "sid"])["false_positive"]
        ix = a1.index.intersection(g1.index)
        if len(ix):
            tests.append(dict(comparison=f"H3a: FPR reduction vs accumulated, {m} = FPR(accumulated) - FPR({m})",
                              **paired_bootstrap(a1.loc[ix].values, g1.loc[ix].values,
                                                 strata=[sd for sd, _ in ix])))
        gb = B[B["method"] == m].set_index(["seed", "N", "sid"])["asr"] if m in set(B["method"]) else None
        if gb is not None:
            ix2 = b0.index.intersection(gb.index)
            if len(ix2):
                tests.append(dict(comparison=f"H3b: ASR reduction vs accumulated, {m}",
                                  **paired_bootstrap(b0.loc[ix2].values, gb.loc[ix2].values,
                                                     strata=[sd for sd, _, _ in ix2])))
    tests = pd.DataFrame(tests)
    save_df(tests, out, "h3_tests", title="H3: paired bootstrap")
    ok = bool(t_fpr["excludes_zero"] and t_fpr["diff"] > 0 and t_asr["diff"] >= 0)
    v = verdict(ok, f"false-positive reduction vs accumulated = {t_fpr['diff']:+.3f} "
                    f"[{t_fpr['lo']:+.3f}, {t_fpr['hi']:+.3f}] ; "
                    f"ASR reduction vs accumulated = {t_asr['diff']:+.3f} [{t_asr['lo']:+.3f}, {t_asr['hi']:+.3f}] "
                    f"(positive = the full system is better)")
    if a0.loc[idx].sum() == 0 and a1.loc[idx].sum() == 0:
        v = ("**criterion untestable** -- neither full nor accumulated raised a single false positive in the H3a set "
             "(report them as indistinguishable and read H3b alongside)")
    write_summary(out, "H3: Authority Principle under Topic Change, Drift, and Probing", [
        ("Verdict against the paper's criterion (Table IV)", v),
        ("How to read the differences", "Every difference below is oriented so that a positive value means the full system "
                                        "is better: a reduction of the false-positive rate (FPR) or of the attack success "
                                        "rate (ASR). In the tables of rates themselves, lower is better for both."),
        ("Tests (paired bootstrap)", tests),
        ("H3a: false positive", a_tab), ("H3b: boiling-frog", b_tab),
        ("H3c: threshold-aware", C.groupby(["method", "N"]).mean(numeric_only=True).reset_index().drop(columns="seed")),
        ("H3d: probing / reset", D.groupby(["scenario", "method", "attempt"], dropna=False).mean(numeric_only=True)
         .reset_index().drop(columns="seed") if not D.empty else "no data")])
    write_run_info(out, cfg, world, emb, dict(experiment="H3"))
    log(f"results written to {out}  ->  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
