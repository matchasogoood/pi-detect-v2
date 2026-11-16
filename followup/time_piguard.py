"""Measures PIGuard alone (no session logic): ms per event and ms per session, on the texts of the final run.
Read-only: touches no result file and changes no config.  Run from the project root:
    python followup/time_piguard.py
"""
import argparse, time, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from pidetect.baselines import HFClassifierBaseline

ap = argparse.ArgumentParser()
ap.add_argument("--events", default="results/final_ft/h1/events.csv.gz")
ap.add_argument("--n_sessions", type=int, default=200)
ap.add_argument("--seed", type=int, default=20)
a = ap.parse_args()

ev = pd.read_csv(a.events, usecols=["method", "seed", "sid", "idx", "text"])
m = "full+B5(ext)" if "full+B5(ext)" in set(ev.method) else ev.method.iloc[0]
ev = ev[(ev.method == m) & (ev.seed == a.seed)].sort_values(["sid", "idx"])
sids = ev.sid.drop_duplicates().sample(min(a.n_sessions, ev.sid.nunique()), random_state=0)
ev = ev[ev.sid.isin(sids)]
print(f"{len(sids)} sessions, {len(ev)} events (method rows used: {m}, seed {a.seed})")

g = HFClassifierBaseline("leolee99/PIGuard", "B3b-PIGuard", trust_remote_code=True)
for t in ev.text.head(10).astype(str):          # warm-up, not timed
    g.score_fresh([t])

per_event, per_session = [], []
for sid, d in ev.groupby("sid"):
    tot = 0.0
    for t in d.text.astype(str):
        t0 = time.perf_counter()
        g.score_fresh([t])
        dt = (time.perf_counter() - t0) * 1000
        per_event.append(dt); tot += dt
    per_session.append(tot)

pe, ps = np.array(per_event), np.array(per_session)
print(f"device: {g.device}")
print(f"PIGuard alone, ms per event  : mean {pe.mean():.1f}  p95 {np.percentile(pe,95):.1f}")
print(f"PIGuard alone, ms per session: mean {ps.mean():.0f}  p95 {np.percentile(ps,95):.0f}")
