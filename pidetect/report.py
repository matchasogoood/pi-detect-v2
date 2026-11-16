"""report.py : writes results as CSV / Markdown / plots (PNG) / metadata files"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def to_markdown(df: pd.DataFrame, floatfmt: int = 3) -> str:
    """A simple Markdown table (no tabulate dependency)"""
    if df is None or df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "-" if v != v else f"{v:.{floatfmt}f}"
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    data = [df[c].tolist() for c in cols]            # tolist() keeps each column's dtype (integers do not become 1.000)
    for i in range(len(df)):
        lines.append("| " + " | ".join(cell(col[i]) for col in data) + " |")
    return "\n".join(lines) + "\n"


def save_df(df: pd.DataFrame, out_dir: str, name: str, floatfmt: int = 3, title: str | None = None) -> None:
    ensure_dir(out_dir)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        if title:
            f.write(f"### {title}\n\n")
        f.write(to_markdown(df, floatfmt))


def fmt_ci(mean: float, lo: float, hi: float, pct: bool = True, nd: int = 1) -> str:
    if mean != mean:
        return "-"
    k = 100.0 if pct else 1.0
    return f"{mean * k:.{nd}f} [{lo * k:.{nd}f}, {hi * k:.{nd}f}]"


def paper_table(summary: pd.DataFrame, metrics: Sequence[str], pct: bool = True) -> pd.DataFrame:
    """A paper-ready table: 'mean [lo, hi]' (in %)"""
    rows = []
    for _, r in summary.iterrows():
        row = {"Method": r["method"], "target FPR": f"{r['target_fpr'] * 100:g}%"}
        for m in metrics:
            if m in r and f"{m}_lo" in r:
                row[m + (" (%)" if pct and m != "AUC" else "")] = fmt_ci(
                    r[m], r[f"{m}_lo"], r[f"{m}_hi"], pct=(pct and m != "AUC"), nd=1 if m != "AUC" else 3)
        if not r.get("feasible_all", True):
            note = "target FPR unreachable (TPR=0 by the usual ROC convention)"
            tr, fr = r.get("TPR_raw", float("nan")), r.get("FPR_raw", float("nan"))
            if tr == tr and fr == fr:            # a real measurement exists -> report it so it does not read as "caught nothing"
                note += f" - measured (mean across seeds): TPR {tr * 100:.1f}% at FPR {fr * 100:.1f}%"
            row["note"] = note
        elif not r.get("resolvable", True):
            row["note"] = "too few negatives to measure this target FPR accurately"
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots
def plot_heatmap(df: pd.DataFrame, x: str, y: str, value: str, path: str, title: str, cmap: str = "viridis",
                 fmt: str = "{:.2f}") -> None:
    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * piv.shape[1], 1.0 + 0.6 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" if isinstance(c, (int, float, np.number)) else str(c)
                                        for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{c:g}" if isinstance(c, (int, float, np.number)) else str(c)
                                        for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax.text(j, i, fmt.format(v), ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lines(df: pd.DataFrame, x: str, y: str, hue: str, path: str, title: str, ylabel: str,
               lo: Optional[str] = None, hi: Optional[str] = None, xlabel: Optional[str] = None,
               ylim: Optional[tuple] = None) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, g in df.groupby(hue, sort=False):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=str(name))
        if lo and hi and lo in g and hi in g:
            ax.fill_between(g[x], g[lo], g[hi], alpha=0.15)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bars(labels: Sequence[str], values: Sequence[float], path: str, title: str, ylabel: str,
              lo: Optional[Sequence[float]] = None, hi: Optional[Sequence[float]] = None, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(labels), 3.8))
    x = np.arange(len(labels))
    err = None
    if lo is not None and hi is not None:
        err = [np.array(values) - np.array(lo), np.array(hi) - np.array(values)]
    ax.bar(x, values, yerr=err, capsize=3, color="#4c78a8")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: Dict[str, pd.DataFrame], path: str, title: str = "ROC (thresholds from validation, evaluated on test)") -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for name, pts in curves.items():
        p = pts.sort_values("FPR")
        ax.plot(p["FPR"], p["TPR"], marker=".", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ metadata
def write_metadata(out_dir: str, args: Dict, world_summary: str, embedder, extra: Optional[Dict] = None) -> None:
    ensure_dir(out_dir)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], platform=platform.platform(),
                embedder=embedder.name, embedder_is_fallback=bool(embedder.is_fallback), world=world_summary,
                args=args)
    for mod in ("numpy", "pandas", "sklearn", "torch", "transformers", "sentence_transformers"):
        try:
            meta["ver_" + mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    warn = os.path.join(out_dir, "WARNING_NOT_FOR_PAPER.txt")
    if embedder.is_fallback or "synthetic" in world_summary:
        with open(warn, "w", encoding="utf-8") as f:
            f.write("These results come from the fallback embedder and/or synthetic data -> they only show that the code runs.\n"
                    "Do not put these numbers in the paper (rerun with Sentence-BERT and the real data).\n")
    elif os.path.exists(warn):
        os.remove(warn)
