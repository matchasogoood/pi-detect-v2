"""Supplementary: choice of the decision model, evaluated on the SAME data as the final run (seeds 20-24).

For every seed the data are the final run's own splits (same make_splits, de-duplication on):
  fit set   : the direct-channel texts of the DEVELOPMENT split (attacks vs benign), with out-of-fold FT-C scores (3 inner group folds; FT-C
              retrained without every text in a cluster of the held-out texts, so a text is never scored by a model that saw it or a copy of it)
  test set  : the direct-channel texts of the TEST split that the final run builds its direct sessions from: sp.att['test'] (attacks) versus
              sp.ben['test'] + sp.triggers['test'] (benign)   -- see h1_sessions in experiments/h1_behavior_vs_memoryless.py
  FT-C      : for the test scores, exactly the final-run FT-C of that seed (FineTunedMiniLM(inject=True).fit_for_seed on the whole dev pool)
Decision models (rule on s, logistic regression, tree, forest, gradient boosting, LR without n_imp/n_ai), grids, a 5% FPR threshold set on the
fit set and applied unchanged, and the statistics are those of experiments/prelim_decision_model.py; each seed is one fold (5 folds, 1 repeat).
The seeds' splits overlap (the dev of one seed contains test texts of another), so the folds are not independent and the p-values are indicative.
This is a check run AFTER the final run: it does not select anything (see PREREG_ft.md, note 14:50).
usage: python followup/run_decision_ft_test.py --smoke      |      python followup/run_decision_ft_test.py
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import run_decision_ft_dev as D
import ft_scorer as F
from pidetect.attackbank import HAND_ATTACKS
from pidetect.dedup_splits import clusters
from pidetect.lexical import lexical_counts
from pidetect.runtime import add_common_args, log, setup
from pidetect.sessions import make_splits

P, S = D.P, D.S
SEP = "||"
CFG = WORLD = None
SPS = {}                    # splits built BEFORE world.deepset is replaced, reused for the training          # kept outside S, which prepare() clears for every seed


def parse(t):
    s, role, txt = t.split(SEP, 2)
    return int(s), role, txt


def prepare(seed, cfg, world):
    if S.get("seed") == seed:
        return
    sp = SPS[seed]
    pos, neg, dpos, dneg = D.build_pool(sp)
    tpos, tneg = test_sets(sp)
    allt = sorted(set(pos) | set(neg) | set(dpos) | set(dneg))
    S.clear()
    S.update(seed=seed, sp=sp, pos=pos, neg=neg, dpos=dpos, dneg=dneg, idx={t: i for i, t in enumerate(allt)})
    S["cl"] = clusters(allt)
    S["hard"] = F.alpaca_hard_negatives(F.N_HARD, seed, {F._norm(t) for t in allt + tpos + tneg})
    D.SEED = seed


def test_sets(sp):
    tpos = [d["text"] for d in sp.att["test"]]
    tneg = [d["text"] for d in sp.ben["test"]] + list(sp.triggers["test"])
    return sorted(set(map(str, tpos))), sorted(set(map(str, tneg)) - set(map(str, tpos)))


def fold_features_test(texts_tr, y_tr, texts_te, emb, cfg, inputs, seed=None):
    from sklearn.model_selection import StratifiedGroupKFold
    seed = parse(texts_tr[0])[0]
    prepare(seed, CFG, WORLD)
    tr = [parse(t)[2] for t in texts_tr]; te = [parse(t)[2] for t in texts_te]
    y_tr = np.asarray(y_tr).astype(int)
    g = np.array([S["cl"][S["idx"][t]] for t in tr])
    oof = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(tr, y_tr, g):
        ft = D.train_without([tr[i] for i in b], f"seed {seed} inner")
        oof[b] = ft.score([tr[i] for i in b]); ft.release()
    ft = F.FineTunedMiniLM(inject=True, name="B4-OursFT2")          # the final-run FT-C of this seed
    ft.fit_for_seed(seed, S["sp"], HAND_ATTACKS)
    s_te = ft.score(te)
    ft.model = ft.tok = None
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lex_tr = np.array([lexical_counts(t) for t in tr], dtype=float)
    lex_te = np.array([lexical_counts(t) for t in te], dtype=float)
    log(f"  seed {seed}: fit set {len(tr)} ({int(y_tr.sum())} attacks), test set {len(te)}")
    return np.column_stack([oof, lex_tr]), np.column_stack([s_te, lex_te]), ["s", "n_imp", "n_ai"]


class SeedFolds:
    """Drop-in for RepeatedStratifiedKFold: fold k = seed k (fit on its dev texts, test on its test texts)."""
    def __init__(self, n_splits=5, n_repeats=1, random_state=0):
        pass

    def split(self, X, y):
        meta = [parse(t) for t in X]
        for seed in dict.fromkeys(m[0] for m in meta):
            yield (np.array([i for i, m in enumerate(meta) if m[0] == seed and m[1] == "dev"]),
                   np.array([i for i, m in enumerate(meta) if m[0] == seed and m[1] == "test"]))


def main():
    argv = sys.argv[1:]
    if "--config" not in argv:
        argv += ["--config", str(HERE / "config_ft.yaml")]
    out = argv[argv.index("--out") + 1] if "--out" in argv else str(D.ROOT / "results" / "final_ft" / "decision_ft_test")
    if "--out" not in argv:
        argv += ["--out", out]
    for a in ("--inputs", "new"), ("--no-permutation", None):
        if a[0] not in argv:
            argv += [x for x in a if x]
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--inputs"); ap.add_argument("--no-permutation", action="store_true")
    args = ap.parse_args(argv)
    cfg, world, emb = setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
    global CFG, WORLD
    CFG, WORLD = cfg, world
    rows = []
    for seed in cfg["seeds"]:
        sp = make_splits(world, seed, dedup=bool(cfg["data"].get("dedup", False)))
        SPS[seed] = sp
        _, _, dpos, dneg = D.build_pool(sp)
        tpos, tneg = test_sets(sp)
        rows += [dict(text=f"{seed}{SEP}dev{SEP}{t}", label=1) for t in dpos] + [dict(text=f"{seed}{SEP}dev{SEP}{t}", label=0) for t in dneg]
        rows += [dict(text=f"{seed}{SEP}test{SEP}{t}", label=1) for t in tpos] + [dict(text=f"{seed}{SEP}test{SEP}{t}", label=0) for t in tneg]
        log(f"seed {seed}: fit set {len(dpos)} attacks / {len(dneg)} benign; test set {len(tpos)} attacks / {len(tneg)} benign")
    world.deepset.clear(); world.deepset.update({"train": rows, "test": []})
    cfg["prelim"].update(n_repeats=1, n_folds=len(cfg["seeds"]))
    import sklearn.model_selection as skm
    skm.RepeatedStratifiedKFold = SeedFolds
    P.fold_features = fold_features_test
    P.setup = lambda a, need: (cfg, world, emb)
    sys.argv = ["prelim_decision_model"] + argv
    rc = P.main()
    Path(out, "NOTE.txt").write_text(
        "Decision-model comparison on the final-run data (followup/run_decision_ft_test.py): fit on the dev direct texts (out-of-fold FT-C scores), "
        "test on the test-split direct texts of seeds 20-24, FT-C = the final-run FT-C of each seed. One seed = one fold. Check run after the final run; selects nothing.\n"
        "The label 'inputs=new' comes from prelim_decision_model.py and means (s, n_imp, n_ai).\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
