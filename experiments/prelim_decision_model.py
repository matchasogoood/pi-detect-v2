#!/usr/bin/env python3
"""Preliminary : choosing the decision model (rerunning Table VI -- Section IV-C of the paper)

The important difference from the first run: the attack bank is rebuilt in every fold from the attack prompts of
that fold's training part only -> a test prompt can never find itself in the bank (closing the leak the paper notes)

Data    : all of deepset/prompt-injections (train + test = 662 prompts, 354 after the English filter)
Splits  : stratified 5-fold, repeated 3 times (15 folds)
Inputs  : old = (f, m, n_imp, n_ai) as in Table VI  |  new = (s, n_imp, n_ai) with the new Stage 2 (fitted per fold)
Models  : rule (m + lambda f > tau), logistic regression, decision tree, random forest, gradient boosting,
          plus an LR without n_imp and n_ai (to show how much those two counters add)
Threshold : set on the training part for a 5% FPR (from out-of-fold scores within that part), then applied unchanged to the test part
Statistics: corrected SE, corrected resampled t-test and McNemar against LR, and the one-standard-error rule,
          plus a label permutation check (every model should fall to chance, AUC ~0.5)

How to run:
  python experiments/prelim_decision_model.py --smoke --data synthetic --embedder hashing
  python experiments/prelim_decision_model.py                 # the real run (inputs taken from prelim.inputs)
  python experiments/prelim_decision_model.py --inputs new    # use the new Stage 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pidetect.attackbank import HAND_ATTACKS, build_bank  # noqa: E402
from pidetect.canon import canonicalize  # noqa: E402
from pidetect.expkit import write_summary  # noqa: E402
from pidetect.lexical import lexical_counts  # noqa: E402
from pidetect.metrics import auc_score, rates, target_reached, threshold_for_fpr  # noqa: E402
from pidetect.report import save_df  # noqa: E402
from pidetect.rules import rule_flag  # noqa: E402
from pidetect.runtime import add_common_args, log, out_dir, setup, write_run_info  # noqa: E402
from pidetect.stage2 import fit_fusion_arrays, fused_score  # noqa: E402
from pidetect.stats import corrected_se, corrected_ttest, mcnemar  # noqa: E402

COMPLEXITY = ["rule", "logistic_regression", "decision_tree", "gradient_boosting", "random_forest"]


def model_specs(seed: int):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    return {
        "logistic_regression": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
                                {"m__C": [0.01, 0.1, 1.0, 10.0]}),
        "decision_tree": (DecisionTreeClassifier(random_state=seed),
                          {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [5, 10]}),
        "random_forest": (RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
                          {"max_depth": [3, 5, None]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=seed),
                              {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    }


def fold_features(texts_tr, y_tr, texts_te, emb, cfg, inputs: str, seed: int):
    """Computes this fold's features -- the bank (and the head, when inputs=new) comes from the training part only"""
    bank, _ = build_bank([t for t, y in zip(texts_tr, y_tr) if y == 1], emb, n_clusters=cfg["bank"]["n_clusters"],
                         n_holdout=0, seed=seed, extra=HAND_ATTACKS)
    E_tr, E_te = emb.encode(texts_tr), emb.encode(texts_te)
    m_tr, m_te = bank.max_cos_batch(E_tr), bank.max_cos_batch(E_te)
    f_tr = np.array([rule_flag(t) for t in texts_tr])
    f_te = np.array([rule_flag(t) for t in texts_te])
    lex_tr = np.array([lexical_counts(t) for t in texts_tr], dtype=float)
    lex_te = np.array([lexical_counts(t) for t in texts_te], dtype=float)
    if inputs == "old":
        X_tr = np.column_stack([f_tr, m_tr, lex_tr])
        X_te = np.column_stack([f_te, m_te, lex_te])
        names = ["f", "m", "n_imp", "n_ai"]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        head = LogisticRegression(C=cfg["stage2"].get("head_C", 1.0), class_weight="balanced", max_iter=3000)
        q_tr = cross_val_predict(head, E_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                 method="predict_proba")[:, 1]
        q_te = head.fit(E_tr, y_tr).predict_proba(E_te)[:, 1]
        g_tr = np.array([canonicalize(t).g for t in texts_tr])
        g_te = np.array([canonicalize(t).g for t in texts_te])
        eta = cfg["stage2"]["eta"]
        par = fit_fusion_arrays(q_tr, m_tr, f_tr, g_tr, y_tr, eta=eta)
        s_tr, s_te = fused_score(q_tr, m_tr, f_tr, g_tr, par, eta), fused_score(q_te, m_te, f_te, g_te, par, eta)
        X_tr = np.column_stack([s_tr, lex_tr])
        X_te = np.column_stack([s_te, lex_te])
        names = ["s", "n_imp", "n_ai"]
    return X_tr, X_te, names


def run_cv(texts, y, emb, cfg, inputs: str, n_rep: int, n_folds: int, target: float, permute: bool = False):
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y).astype(int)
    if permute:
        y = np.random.RandomState(123).permutation(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_rep, random_state=0)
    rows, preds, coefs = [], [], []
    for k, (tr, te) in enumerate(rskf.split(texts, y)):
        rep = k // n_folds
        X_tr, X_te, names = fold_features(list(texts[tr]), y[tr], list(texts[te]), emb, cfg, inputs, seed=k)
        inner = StratifiedKFold(3, shuffle=True, random_state=k)
        oofcv = StratifiedKFold(5, shuffle=True, random_state=k)
        variants = {}
        # rule : m + lambda f (inputs=old) or s (inputs=new)
        if inputs == "old":
            best = None
            for lam in (0.0, 0.1, 0.2, 0.3, 0.5):
                sc = X_tr[:, 1] + lam * X_tr[:, 0]
                thr = threshold_for_fpr(sc[y[tr] == 0], target)
                tpr = rates(y[tr], (sc > thr).astype(int))["TPR"]
                if best is None or tpr > best[0]:
                    best = (tpr, lam, thr)
            _, lam, thr = best
            variants["rule"] = (X_te[:, 1] + lam * X_te[:, 0], thr, None)
        else:
            thr = threshold_for_fpr(X_tr[y[tr] == 0, 0], target)
            variants["rule"] = (X_te[:, 0], thr, None)
        for name, (est, grid) in model_specs(k).items():
            gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc").fit(X_tr, y[tr])
            oof = cross_val_predict(clone(gs.best_estimator_), X_tr, y[tr], cv=oofcv, method="predict_proba")[:, 1]
            thr = threshold_for_fpr(oof[y[tr] == 0], target)
            model = gs.best_estimator_
            t0 = time.perf_counter()
            for row in X_te[:50]:
                model.predict_proba(row[None, :])
            lat = (time.perf_counter() - t0) / min(50, len(X_te)) * 1e6
            variants[name] = (model.predict_proba(X_te)[:, 1], thr, lat)
            if name == "logistic_regression":
                coefs.append(dict(zip(names, model.named_steps["m"].coef_[0])))
        # the LR without the n_imp and n_ai counters
        keep = [i for i, n in enumerate(names) if n not in ("n_imp", "n_ai")]
        est, grid = model_specs(k)["logistic_regression"]
        gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc").fit(X_tr[:, keep], y[tr])
        oof = cross_val_predict(clone(gs.best_estimator_), X_tr[:, keep], y[tr], cv=oofcv, method="predict_proba")[:, 1]
        variants["lr_without_counts"] = (gs.best_estimator_.predict_proba(X_te[:, keep])[:, 1],
                                         threshold_for_fpr(oof[y[tr] == 0], target), None)
        # latency of the rule
        t0 = time.perf_counter()
        for row in X_te[:50]:
            _ = row[1] + 0.2 * row[0]
        variants["rule"] = variants["rule"][:2] + ((time.perf_counter() - t0) / min(50, len(X_te)) * 1e6,)
        for name, (score, thr, lat) in variants.items():
            pred = (score > thr).astype(int)
            rr = rates(y[te], pred)
            rows.append(dict(fold=k, rep=rep, model=name, TPR=rr["TPR"], FPR=rr["FPR"], AUC=auc_score(y[te], score),
                             latency_us=lat, n_train=len(tr), n_test=len(te),
                             n_distinct_scores=int(len(np.unique(score))),
                             target_reached=int(target_reached(rr["FPR"], target)),
                             flagged=int((pred == 1).sum())))
            preds.append(pd.DataFrame(dict(fold=k, rep=rep, model=name, i=te, correct=(pred == y[te]).astype(int))))
        log(f"  fold {k + 1}/{n_rep * n_folds} done{' (permuted labels)' if permute else ''}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(coefs)


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--inputs", choices=["old", "new"], default=None, help="overrides prelim.inputs in the config")
    ap.add_argument("--no-permutation", action="store_true", help="skip the label permutation check")
    args = ap.parse_args()
    cfg, world, emb = setup(args, ("deepset",))
    p = cfg["prelim"]
    inputs = args.inputs or p.get("inputs", "old")
    out = out_dir(args, "prelim_" + inputs)
    rows = world.deepset.get("train", []) + world.deepset.get("test", [])
    texts, y = [r["text"] for r in rows], [int(r["label"]) for r in rows]
    log(f"deepset: {len(y)} prompts ({sum(y)} injections, {len(y) - sum(y)} benign) | inputs={inputs}")
    res, preds, coefs = run_cv(texts, y, emb, cfg, inputs, p["n_repeats"], p["n_folds"], p["target_fpr"])
    res.to_csv(os.path.join(out, "per_fold.csv"), index=False)
    n_tr, n_te = int(res["n_train"].mean()), int(res["n_test"].mean())
    lr = res[res["model"] == "logistic_regression"].set_index("fold")
    table = []
    for name, g in res.groupby("model", sort=False):
        g = g.set_index("fold")
        row = dict(model=name, TPR=g["TPR"].mean(), corr_SE=corrected_se(g["TPR"].values, n_tr, n_te),
                   achieved_FPR=g["FPR"].mean(), AUC=g["AUC"].mean(), latency_us=g["latency_us"].median(),
                   n_distinct_scores=g["n_distinct_scores"].mean(), folds_target_reached=g["target_reached"].mean(),
                   target_reached_on_mean=int(target_reached(g["FPR"].mean(), p["target_fpr"])))
        if name not in ("logistic_regression", "lr_without_counts"):
            _, _, p_t = corrected_ttest((lr["TPR"] - g["TPR"]).values, n_tr, n_te)
            pm = []
            for rep in sorted(preds["rep"].unique()):
                a = preds[(preds["model"] == "logistic_regression") & (preds["rep"] == rep)].sort_values("i")
                b = preds[(preds["model"] == name) & (preds["rep"] == rep)].sort_values("i")
                pm.append(mcnemar(a["correct"].values, b["correct"].values))
            row.update(p_ttest=p_t, p_mcnemar=float(np.median(pm)))
        elif name == "lr_without_counts":
            rule = res[res["model"] == "rule"].set_index("fold")
            _, _, p_t = corrected_ttest((g["TPR"] - rule["TPR"]).values, n_tr, n_te)
            row.update(p_ttest_vs_rule=p_t)
        table.append(row)
    table = pd.DataFrame(table)
    # one-standard-error rule
    main_models = table[table["model"].isin(COMPLEXITY)].copy()
    best = main_models.loc[main_models["TPR"].idxmax()]
    within = main_models[main_models["TPR"] >= best["TPR"] - best["corr_SE"]]
    chosen = min(within["model"], key=COMPLEXITY.index)
    table["selected_by_1SE"] = (table["model"] == chosen).astype(int)
    save_df(table, out, "table_v", title=f"Table VI (rerun, inputs={inputs}, bank rebuilt per fold)")
    coef = coefs.mean().to_frame("mean_standardized_coef").reset_index().rename(columns={"index": "feature"})
    save_df(coef, out, "lr_coefficients", title="Standardized LR weights, averaged over the folds")
    coarse = table[(table["folds_target_reached"] < 0.5) & (table["model"] != "lr_without_counts")]["model"].tolist()
    note = (f"A threshold set for a {p['target_fpr']:.0%} FPR can only be placed on a score with enough distinct "
            f"values. n_distinct_scores is the mean number of distinct scores a model produces on a held-out fold "
            f"(about {int(n_te)} prompts) and folds_target_reached the share of folds whose achieved FPR landed "
            f"within 25% of the target. Where that share is low the model's TPR is not a TPR at that FPR: a shallow "
            f"tree puts almost every prompt in the same leaf, the threshold lands at or above its highest benign "
            f"score, and nothing is flagged, which shows as TPR 0.000 with FPR 0.000. For those models the AUC, "
            f"which needs no threshold, is the comparable number. target_reached_on_mean applies the same 25% test "
            f"to the FPR averaged over all folds; folds_target_reached is lower for every model simply because one "
            f"fold holds only about {int(n_te)} prompts, so its FPR moves in steps of a few percent."
            + (f" Affected here: {', '.join(coarse)}." if coarse else " No model is affected here."))
    sections = [("Choice under the one-standard-error rule", f"**{chosen}** (highest TPR = {best['model']} "
                 f"{best['TPR']:.3f}, corrected SE = {best['corr_SE']:.3f})"), ("Table V", table),
                ("Why some models show TPR 0.000", note), ("LR weights", coef)]
    if not args.no_permutation:
        perm, _, _ = run_cv(texts, y, emb, cfg, inputs, 1, p["n_folds"], p["target_fpr"], permute=True)
        pt = perm.groupby("model")["AUC"].mean().reset_index().rename(columns={"AUC": "AUC_permuted_labels"})
        save_df(pt, out, "permutation_check", title="Label permutation: every model should land near AUC 0.5")
        sections.append(("Label permutation (expect ~0.5)", pt))
    tex = []
    for r in table.to_dict("records"):
        pv = "--" if pd.isna(r.get("p_ttest", np.nan)) else f"{r['p_ttest']:.3f} / {r['p_mcnemar']:.3f}"
        lat = "--" if pd.isna(r["latency_us"]) else f"{r['latency_us']:,.0f}"
        tex.append(f"{r['model']} & {r['TPR']:.3f} & {r['corr_SE']:.3f} & {r['achieved_FPR']:.3f} & "
                   f"{r['AUC']:.3f} & {pv} & {lat} \\\\")
    sections.append(("LaTeX (drop-in replacement for the rows of Table VI)", "```\n" + "\n".join(tex) + "\n```"))
    write_summary(out, f"Preliminary: Choice of the Decision Model (inputs={inputs})", sections)
    write_run_info(out, cfg, world, emb, dict(experiment="prelim", inputs=inputs))
    log(f"results written to {out}  ->  chose {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
