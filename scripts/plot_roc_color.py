# NOTE: needs the raw event logs of the final run (events.csv.gz, sessions.csv), which are too large for git.
# Run from the repository root:  python scripts/plot_roc_color.py --results <folder with those two files>
"""Colour operating curve (thin command-line wrapper; the drawing code is in pidetect/roc_plot.py).
The figure is also drawn automatically at the end of followup/run_ft.py and followup/run_ablation.py and saved in
that run's results folder (fig_roc_color.pdf/.png and fig_roc_color_paper.pdf/.png).  Use this script only to redo it.

usage (from the project folder):
    python plot_roc_color.py                                  # results/final_ft/h1  ->  <results>/fig_roc_color.pdf/.png
    python plot_roc_color.py --results results/final_ft/ablation --all-methods
    python plot_roc_color.py --paper                          # small 3.45-inch version
"""
import argparse, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):   # project root, or the parent of scripts/ in the git copy
    sys.path.insert(0, _p)
from pidetect.roc_plot import make_figure

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/final_ft/h1", help="folder with events.csv.gz and sessions.csv")
ap.add_argument("--out", default=None, help="output stem; default: <results>/fig_roc_color")
ap.add_argument("--xmax", type=float, default=0.25)
ap.add_argument("--paper", action="store_true")
ap.add_argument("--all-methods", action="store_true", help="also draw methods not in the standard list (ablation)")
a = ap.parse_args()
out = a.out or os.path.join(a.results, "fig_roc_color" + ("_paper" if a.paper else ""))
make_figure(a.results, out, a.xmax, a.paper, a.all_methods)
