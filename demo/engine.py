"""engine.py : runs the REAL proposed system (full + B5(ext) = "+FT-C+block") one event at a time and returns a
complete trace of what every stage computed.  It reuses the experiment code (followup/run_ft.py), so the system in
the demo is built exactly as in the final run, for one seed.

First start: loads the datasets, fits the system and fine-tunes FT-C on the seed's dev split (a few minutes on a GPU).
FT-C weights are cached in demo/cache/, so later starts skip the fine-tuning.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "followup", ROOT / "experiments"):
    sys.path.insert(0, str(p))

OURS_FPR_SCALE = 0.4         # validation FPR target of the FT-C systems = 0.4 x 5% = 2% (PREREG_ft.md)


class Engine:
    def __init__(self, seed: int = 20, raw_dir: str | None = None, cache_dir: str | None = None):
        self.seed = seed
        self.raw_dir = raw_dir
        self.cache = Path(cache_dir) if cache_dir else HERE / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.sessions = {}
        self._build()

    # ------------------------------------------------------------------ build the real system
    def _build(self):
        import torch
        import yaml
        import run_ft
        from pidetect.attackbank import HAND_ATTACKS  # noqa: F401
        from pidetect.baselines import ExternalContentScore, ext_pipe_overrides
        import ft_scorer

        run_ft.OURS_FPR_SCALE = OURS_FPR_SCALE
        cfg_path = run_ft.make_config([self.seed])
        if self.raw_dir:                                      # point the data folder somewhere else (e.g. a shared copy)
            c = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            c["data"]["raw_dir"] = str(Path(self.raw_dir).resolve())
            cfg_path.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
        far = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["h1"]["target_session_fpr"]
        h1 = run_ft.load_h1()
        ft, taus = run_ft.install(h1, far)

        # cache the fine-tuned weights: fine-tuning is the slow part and is deterministic given the seed
        wfile = self.cache / f"ftc_seed{self.seed}.pt"
        orig_fit = ft.fit_for_seed

        def fit_or_load(seed, sp, hand):
            man = Path(__file__).resolve().parent / "weights" / f"ftc_seed{self.seed}.json"
            if not wfile.exists() and man.exists():        # weights shipped in git as parts: join them (hash-checked)
                try:
                    import weights_pack
                    weights_pack.join(man, wfile)
                    print(f"[demo] FT-C weights joined from {man.parent} (hash verified)")
                except Exception as e:  # noqa: BLE001
                    print(f"[demo] could not use the shipped weights ({e}); fine-tuning from scratch instead (about 70 s on a GPU).\n"
                          "       To see which part is bad and how to repair it:  python demo/weights_pack.py check")
            if wfile.exists():
                model, tok = ft_scorer._build(ft.device)
                model.load_state_dict(torch.load(wfile, map_location=ft.device))
                model.eval()
                ft.model, ft.tok, ft.seed, ft._cache = model, tok, seed, {}
                print(f"[demo] FT-C weights loaded from {wfile}")
            else:
                orig_fit(seed, sp, hand)
                torch.save(ft.model.state_dict(), wfile)
                print(f"[demo] FT-C weights saved to {wfile}")
        ft.fit_for_seed = fit_or_load

        args = SimpleNamespace(config=str(cfg_path), smoke=False, data=None, embedder=None, seeds=[self.seed])
        cfg, world, emb = h1.setup(args, ("bipia", "deepset", "gandalf", "notinject", "tasktracker"))
        bls = h1.build_baselines(cfg, emb, ["B4", "B5"])
        b5 = bls["B5-OursFT2path"]
        h1.build_full_system(cfg, world, emb, self.seed)               # fits our Stage 2 and trains / loads FT-C
        det, sp, bank, va, vb, info = h1.build_full_system(
            cfg, world, emb, self.seed, ext=ExternalContentScore.from_mode(b5, "ext"), **ext_pipe_overrides("ext"))
        det.name = "full+B5(ext)"
        val_s = h1.h1_sessions(sp, "val", self.seed, cfg["h1"]["n_test_per_kind"])
        benign_val = [s for s in val_s if not s.is_attack]
        ci = h1.calibrate_session_threshold(det, benign_val, cfg["h1"]["target_session_fpr"])
        print(f"[demo] ready: t_blk={ci['t_blk']:.4f} val session FPR={ci['val_session_fpr']:.3f}")
        import sampler
        self.sp, self.pools = sp, sampler.build_pools(sp)
        self.det, self.ft, self.tau_ft = det, ft, float(taus[self.seed]["tau_ft"])
        self.calib = ci

    # ------------------------------------------------------------------ public API
    def config(self):
        d, s2 = self.det, self.det.s2
        c = d.cfg
        return dict(seed=self.seed, system="full+B5(ext)  (+FT-C+block, the proposed system)",
                    thresholds=dict(t_rev=c.t_rev, t_blk=c.t_blk, tau_ft=self.tau_ft, tau_f=float(s2.tau_f),
                                    tau_o=c.tau_o, rho=c.rho, tau_same=c.tau_same, tau_a=c.tau_a, rbar=c.rbar, W=c.W,
                                    k_min=c.k_min),
                    stage2=dict(w_q=float(s2.w_q), w_m=float(s2.w_m), lam=float(s2.lam), lam_g=float(s2.lam_g),
                                eta=float(s2.cfg.eta), b0=float(s2.b0)),
                    features=list(c.features), stub=False)

    def sample(self, kind="any"):
        import sampler
        return sampler.sample(self.pools, kind)

    def reset(self, sid):
        self.sessions[sid] = dict(st=self.det.new_state(), turn=0, n=0, log=[])

    def scenario(self, kind):
        import sampler
        return sampler.make_scenario(self.sp, kind)

    def step(self, sid, text, channel, source=None):
        from pidetect.rules import explain
        from pidetect.schema import ACTION_NAMES, Event
        with self.lock:
            if sid not in self.sessions:
                self.reset(sid)
            S = self.sessions[sid]
            det, st = self.det, S["st"]
            direct = channel == "direct"
            if direct:
                S["turn"] += 1
            S["n"] += 1
            ev = Event(text=text, channel=channel, turn=max(1, S["turn"]),
                       source="user" if direct else (source or f"doc-{S['n']}"))
            before = dict(has_R=st.R is not None, trust=bool(st.trust), a=float(st.a), blocked=dict(st.B))
            t0 = time.perf_counter()
            inner = det.s2._inner                               # Stage 2 without the content block path
            o_in = inner.process(text, channel)
            ft_score = float(self.ft.score([text])[0])
            rec = det.step(ev, st)
            ms = (time.perf_counter() - t0) * 1000
            views = [dict(name=v.name, text=v.text, f=v.f, g=int(v.g), m=float(v.m), q=float(v.q),
                          rule_term=float(v.rule_term), s=float(v.s), nearest_bank=v.nearest_bank)
                     for v in o_in.views]
            raw = o_in.raw
            block_path = (not o_in.fast) and (not o_in.hard) and ft_score > self.tau_ft
            stage = rec.stage
            x = {k: float(v) for k, v in rec.x.items()}
            contrib = {}
            if det.model is not None and stage == "lr":
                sc, lr = det.model.named_steps["scaler"], det.model.named_steps["lr"]
                v = det.vector(rec.x)
                z = lr.coef_[0] * (v - sc.mean_) / np.where(sc.scale_ > 0, sc.scale_, 1.0)
                for i, f in enumerate(det.feature_names):
                    contrib[f] = float(z[i])
                contrib["_intercept"] = float(lr.intercept_[0])
            S["log"].append(rec.action)
            return dict(
                session=sid, event_no=S["n"], channel=channel, text=text, turn=ev.turn, ms=ms,
                stage1=dict(rule_flag=float(raw.f), obf_flag=int(o_in.g), changes=list(o_in.changes),
                            canonical=(o_in.views[1].text if len(o_in.views) > 1 else None),
                            matched_patterns=explain(text)),
                stage2=dict(views=views, fired=o_in.fired.name, bank_fast_path=bool(o_in.fast),
                            nearest_bank=raw.nearest_bank, s_ours=float(o_in.s), ftc_score=ft_score,
                            tau_ft=self.tau_ft, content_block_path=bool(block_path), s=float(rec.s),
                            n_windows=int(raw.n_windows)),
                stage3=dict(ran=(not direct) and stage == "lr", r=float(x["r"]), top_window=raw.top_window,
                            has_reference=before["has_R"], trusted_reference=before["trust"]),
                stage4=dict(a=x["a"], k=x["k"], u=x["u"], o=x["o"], fbar=x["fbar"], delta=x["delta"], h=x["h"],
                            z=x["z"], ran=stage == "lr"),
                decision=dict(stage=stage, p=float(rec.p), action=int(rec.action),
                              action_name=ACTION_NAMES[rec.action], reasons=list(rec.reasons),
                              new_task=bool(rec.new_task), reference_updated=bool(rec.reference_updated),
                              features=x, contributions=contrib, t_rev=det.cfg.t_rev, t_blk=det.cfg.t_blk),
                state_after=dict(has_R=st.R is not None, trust=bool(st.trust), a=float(st.a),
                                 blocked_sources=dict(st.B), window=len(st.H)),
            )
