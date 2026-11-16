"""Supplementary: choice of the decision model with the fine-tuned scorer FT-C as the content score s.

Same protocol as experiments/prelim_decision_model.py (354 English deepset prompts, 5-fold x 3 repeats, threshold set on the
training part for a 5% FPR, same models, same statistics).  Only the content score changes: inside every fold FT-C
(all-MiniLM-L6-v2, same hyper-parameters as followup/ft_scorer.py) is fine-tuned on THAT FOLD'S TRAINING PROMPTS plus
600 Alpaca hard negatives and scored on the held-out prompts, so no test prompt is ever seen by the scorer.  The scores of
the training part, which the decision models are fitted on, are out-of-fold (3 inner folds).  Features: (s_FT-C, n_imp, n_ai).
No code of the system changes; prelim_decision_model.py is imported and only its fold_features() is replaced.
usage: python followup/run_decision_ft.py --smoke        (3 folds, checks the setup)
       python followup/run_decision_ft.py                (15 folds)
"""
import math, sys, time
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("prelim_dm", ROOT / "experiments" / "prelim_decision_model.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
import ft_scorer as F
from pidetect.lexical import lexical_counts
from pidetect.runtime import log

HARD = []          # 600 Alpaca hard negatives, drawn once; texts that occur in deepset are excluded


class FoldFT(F.FineTunedMiniLM):
    name = "B4-OursFT-fold"

    def fit_texts(self, texts, y, seed):
        import torch
        from torch import nn
        y = np.asarray(y).astype(int)
        torch.manual_seed(seed)
        model, tok = F._build(self.device)
        w = len(y) / (2 * np.bincount(y, minlength=2).clip(min=1))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.device))
        HP = F.HP
        opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": HP["lr_enc"]},
                                 {"params": model.head.parameters(), "lr": HP["lr_head"]}], weight_decay=HP["wd"])
        steps = HP["epochs"] * math.ceil(len(texts) / HP["batch"])
        warm = max(1, int(HP["warmup"] * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        rng = np.random.default_rng(seed)
        model.train()
        for _ in range(HP["epochs"]):
            order = rng.permutation(len(texts))
            for i in range(0, len(order), HP["batch"]):
                idx = order[i:i + HP["batch"]]
                enc = tok([texts[j] for j in idx], truncation=True, padding=True,
                          max_length=HP["max_len"], return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                loss = loss_fn(model(**enc), torch.tensor(y[idx], device=self.device))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        self.model, self.tok, self.seed = model, tok, seed
        self._cache = {}

    def release(self):
        import torch
        self.model = self.tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fold_features_ft(texts_tr, y_tr, texts_te, emb, cfg, inputs, seed):
    from sklearn.model_selection import StratifiedKFold
    texts_tr, texts_te = list(texts_tr), list(texts_te)
    y_tr = np.asarray(y_tr).astype(int)
    t0 = time.time()

    def train(idx):
        ft = FoldFT()
        ft.fit_texts([texts_tr[i] for i in idx] + HARD,
                     np.concatenate([y_tr[idx], np.zeros(len(HARD), dtype=int)]), seed)
        return ft

    oof = np.zeros(len(texts_tr))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(texts_tr, y_tr):
        ft = train(a)
        oof[b] = ft.score([texts_tr[i] for i in b])
        ft.release()
    ft = train(np.arange(len(texts_tr)))
    s_te = ft.score(texts_te)
    ft.release()
    lex_tr = np.array([lexical_counts(t) for t in texts_tr], dtype=float)
    lex_te = np.array([lexical_counts(t) for t in texts_te], dtype=float)
    log(f"  fold {seed}: FT-C trained 4 times in {time.time() - t0:.0f}s")
    return np.column_stack([oof, lex_tr]), np.column_stack([s_te, lex_te]), ["s", "n_imp", "n_ai"]


def main():
    global HARD
    out = None
    argv = sys.argv[1:]
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    else:
        out = str(ROOT / "results" / "final_ft" / "decision_ft")
        argv += ["--out", out]
    if "--inputs" not in argv:
        argv += ["--inputs", "new"]          # the label 'new' is only used to reach the branch with inputs (s, n_imp, n_ai)
    if "--no-permutation" not in argv:
        argv += ["--no-permutation"]         # a permutation check would cost 4 more fine-tunings x 5 folds
    # draw the hard negatives once; banned = every deepset prompt (normalised), so none can coincide with a test prompt
    from pidetect.runtime import add_common_args, setup
    import argparse
    ap = add_common_args(argparse.ArgumentParser()); ap.add_argument("--inputs"); ap.add_argument("--no-permutation", action="store_true")
    args = ap.parse_args(argv)
    _, world, _ = setup(args, ("deepset",))
    banned = {F._norm(r["text"]) for part in ("train", "test") for r in world.deepset.get(part, [])}
    HARD = F.alpaca_hard_negatives(F.N_HARD, 0, banned)
    log(f"{len(HARD)} Alpaca hard negatives (none equal to a deepset prompt)")
    P.fold_features = fold_features_ft
    sys.argv = ["prelim_decision_model"] + argv
    rc = P.main()
    Path(out, "NOTE.txt").write_text(
        "Decision-model comparison with FT-C as the content score s (followup/run_decision_ft.py).\n"
        "The label 'inputs=new' in the files comes from prelim_decision_model.py and means (s, n_imp, n_ai); here s is the FT-C score.\n"
        "FT-C is fine-tuned inside every fold on the fold's training prompts + 600 Alpaca hard negatives; training-part scores are out-of-fold.\n",
        encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
