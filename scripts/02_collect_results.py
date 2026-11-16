#!/usr/bin/env python3
"""02_collect_results.py : merges the summary.md of every experiment into results/final/RESULTS.md

  python scripts/02_collect_results.py
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["prelim_old", "prelim_new", "h1", "h2", "h3", "secondary"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "final"))
    a = ap.parse_args()
    parts = ["# All experiment results\n"]
    lang = os.path.join(a.results_dir, "language_report", "language_table.tex")
    if os.path.exists(lang):                       # from scripts/04_language_report.py (the Language composition table)
        parts.append("\n## Languages in the data (rows for the Language composition table of the paper)\n\n```latex\n"
                     + open(lang, encoding="utf-8").read() + "```\n")
    found = 0
    for name in ORDER:
        d = os.path.join(a.results_dir, name)
        p = os.path.join(d, "summary.md")
        if not os.path.exists(p):
            continue
        found += 1
        warn = os.path.exists(os.path.join(d, "WARNING_NOT_FOR_PAPER.txt"))
        info = {}
        if os.path.exists(os.path.join(d, "run_info.json")):
            info = json.load(open(os.path.join(d, "run_info.json"), encoding="utf-8"))
        parts.append(f"\n---\n\n<!-- {name} -->\n")
        if warn:
            parts.append("> **Warning: these results come from synthetic data or the fallback embedder -- not for the paper**\n")
        parts.append(f"> run at {info.get('time', '?')} | data={info.get('data_source', '?')} | "
                     f"embedder={info.get('embedder', '?')} | seeds={info.get('seeds', '?')} | "
                     f"english_only={info.get('english_only', '?')}\n")
        lf = info.get("language_filter") or {}
        if lf:
            parts.append("> English filter (before -> kept): " + ", ".join(
                f"{k} {v['before']}->{v['kept']}" for k, v in lf.items()) + "\n")
        parts.append(open(p, encoding="utf-8").read())
    out = os.path.join(a.results_dir, "RESULTS.md")
    os.makedirs(a.results_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"merged {found} experiment(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
