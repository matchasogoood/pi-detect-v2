"""Colour operating curve: session detection rate vs session false-positive rate (mean of the seeds in a run).
A session is scored by the highest event score it produces; every threshold is swept.
Used by plot_roc_color.py (command line) and called automatically at the end of followup/run_ft.py and
followup/run_ablation.py, which save the figure inside the run's results folder.  Post-processing only:
it reads events.csv.gz and sessions.csv and changes nothing in the experiment."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# (method key in the CSV, legend label, colour, line style, line width)   colours: Okabe-Ito, colour-blind safe
SYSTEMS = [
    ("full+B5(ext)",                 "Proposed (+FT-C+block)", "#D55E00", "-",  2.6),
    ("B4-OursFT2",                   "FT-C alone",             "#0072B2", "--", 1.8),
    ("full+B3b(ind_win_logit_only)", "PIGuard + session",      "#009E73", "-.", 1.8),
    ("B3b-PIGuard",                  "PIGuard",                "#CC79A7", ":",  1.8),
    ("B3c-ProtectAIv2",              "ProtectAI v2",           "#E69F00", (0, (3, 1, 1, 1)), 1.8),
    ("B3a-PromptGuard",              "Prompt-Guard",           "#56B4E9", (0, (5, 2)), 1.8),
]
_EXTRA = ["#000000", "#8C564B", "#7F7F7F", "#F0E442", "#AA4499"]
_EXTRA_LS = ["-", "--", "-.", ":", (0, (5, 2))]


def make_figure(results_dir, out_stem, xmax=0.25, paper=False, all_methods=False, verbose=True):
    """Write {out_stem}.pdf and {out_stem}.png.  all_methods=True also draws every other method found in the
    CSV (used for the ablation run, whose variants are not in SYSTEMS).  Returns the list of drawn methods."""
    fs = 7 if paper else 11
    plt.rcParams.update({"font.size": fs, "axes.linewidth": 0.8, "legend.frameon": False})
    ev = pd.read_csv(os.path.join(results_dir, "events.csv.gz"), usecols=["method", "seed", "sid", "score"])
    ss = pd.read_csv(os.path.join(results_dir, "sessions.csv"), usecols=["method", "seed", "sid", "is_attack"])
    g = ev.groupby(["method", "seed", "sid"], observed=True)["score"].max().reset_index() \
          .merge(ss, on=["method", "seed", "sid"])
    systems = list(SYSTEMS)
    if all_methods:
        known = {s[0] for s in SYSTEMS}
        for i, m in enumerate(sorted(set(g.method) - known)):
            systems.append((m, m, _EXTRA[i % len(_EXTRA)], _EXTRA_LS[i % len(_EXTRA_LS)], 1.8))
    fig, ax = plt.subplots(figsize=(3.45, 2.5) if paper else (7.0, 4.6))
    grid = np.linspace(0, xmax, 501)
    drawn = []
    for key, label, color, ls, lw in systems:
        sub = g[g.method == key]
        if sub.empty:
            if verbose: print("  figure: skipped (not in the CSV):", key)
            continue
        curves = []
        for _, part in sub.groupby("seed"):
            att = part.loc[part.is_attack == 1, "score"].values
            ben = part.loc[part.is_attack == 0, "score"].values
            if len(att) == 0 or len(ben) == 0:
                continue
            thr = np.unique(np.concatenate([att, ben]))[::-1]
            fpr = np.concatenate([[0], [(ben >= t).mean() for t in thr]])
            tpr = np.concatenate([[0], [(att >= t).mean() for t in thr]])
            curves.append(np.interp(grid, fpr, tpr))
        if not curves:
            continue
        ax.plot(grid, np.mean(curves, axis=0), color=color, ls=ls, lw=lw * (0.6 if paper else 1), label=label)
        drawn.append(key)
    ax.axvline(0.05, color="0.6", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(0.052, 0.03, "5% FPR", fontsize=fs - 1.5, color="0.35")
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.0)
    ax.set_xlabel("session false-positive rate"); ax.set_ylabel("session detection rate")
    ax.grid(linewidth=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30 if paper else -0.17), ncol=3, fontsize=fs - 1,
              handlelength=2.6, columnspacing=1.2, labelspacing=0.35)
    fig.tight_layout(pad=0.4)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    if verbose: print(f"  figure: wrote {out_stem}.pdf and .png ({len(drawn)} curves)")
    return drawn


def auto_figures(out_dir, all_methods=False):
    """Called at the end of a run; never lets a plotting problem break the run."""
    try:
        out_dir = str(out_dir)
        make_figure(out_dir, os.path.join(out_dir, "fig_roc_color"), all_methods=all_methods)
        make_figure(out_dir, os.path.join(out_dir, "fig_roc_color_paper"), paper=True, all_methods=all_methods)
    except Exception as e:  # noqa
        print(f"  figure: not drawn ({type(e).__name__}: {e}); run  python plot_roc_color.py --results {out_dir}")
