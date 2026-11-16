#!/usr/bin/env python3
"""run_all.py : runs every experiment in order and merges the results into results/final/RESULTS.md

Order: language report (04_language_report) -> prelim (Table VI) -> H1 -> H2 -> H3 -> secondary -> merge
  python scripts/run_all.py --smoke --data synthetic --embedder hashing   # check that everything runs (~2-3 min, no downloads)
  python scripts/run_all.py --smoke                                       # real data, small sizes
  python scripts/run_all.py                                               # the full run (can take hours on a CPU)
  python scripts/run_all.py --only h1 h3                                  # only some experiments
Other options (--seeds, --skip-baselines, --config) are passed through to every script
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (name, script, extra arguments).  "prelim_old" reruns Table VI with the features of the published table and
# "prelim_new" repeats it with the Stage 2 score of Section III, which the paper promises.
STEPS = [("prelim", "experiments/prelim_decision_model.py", ["--inputs", "old"]),
         ("prelim_new", "experiments/prelim_decision_model.py", ["--inputs", "new"]),
         ("h1", "experiments/h1_behavior_vs_memoryless.py", []),
         ("h2", "experiments/h2_instruction_relevance.py", []),
         ("h3", "experiments/h3_authority_principle.py", []),
         ("secondary", "experiments/s_component_checks.py", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=[s for s, _, _ in STEPS], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", choices=["real", "synthetic"])
    ap.add_argument("--embedder", choices=["sbert", "hashing"])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--skip-baselines", nargs="*", default=[])
    ap.add_argument("--config")
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "final"))
    a = ap.parse_args()
    common = []
    if a.smoke:
        common.append("--smoke")
    if a.data:
        common += ["--data", a.data]
    if a.embedder:
        common += ["--embedder", a.embedder]
    if a.seeds:
        common += ["--seeds", *map(str, a.seeds)]
    if a.config:
        common += ["--config", a.config]
    failed = []
    t_all = time.time()
    if a.data != "synthetic" and not a.only:          # count the languages in the real data (~10 s) for the Language composition table
        print(f"\n{'=' * 70}\n>>> language report\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "04_language_report.py"), "--out",
                        os.path.join(a.results_dir, "language_report")], cwd=ROOT)
    for name, script, extra in STEPS:
        if a.only and name not in a.only:
            continue
        cmd = [sys.executable, os.path.join(ROOT, script), *common, *extra,
               "--out", os.path.join(a.results_dir, "prelim_old" if name == "prelim" else name)]
        if a.skip_baselines and not name.startswith("prelim"):
            cmd += ["--skip-baselines", *a.skip_baselines]
        print(f"\n{'=' * 70}\n>>> {name}: {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(f"<<< {name}: {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f} s)", flush=True)
        if r.returncode != 0:
            failed.append(name)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "02_collect_results.py"), "--results-dir",
                    a.results_dir], cwd=ROOT)
    print(f"\ntotal {(time.time() - t_all) / 60:.1f} min" + (f" | failed: {failed}" if failed else " | all succeeded"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
