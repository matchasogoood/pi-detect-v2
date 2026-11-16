"""runtime.py : everything the experiment scripts share
(reading the config, building the embedder, loading the data, splitting it per seed, building the attack bank, fitting the full system)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .attackbank import HAND_ATTACKS, AttackBank, build_bank
from .data.lang import backend as lang_backend
from .data.world import World, load_world
from .embedder import CachedEmbedder, get_embedder
from .fit import FitConfig, fit_system
from .pipeline import PipeConfig, ProposedDetector
from .sessions import (KBRetriever, Splits, attach_heldout, fitting_sessions, head_training_data, make_splits,
                       set_retriever, split_sessions)
from .stage2 import Stage2Config

ROOT = Path(__file__).resolve().parents[1]


def ensure_utf8_stdio() -> None:
    """Lets print() handle non-ASCII characters on Windows"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


ensure_utf8_stdio()


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


REQUIRED = ["embedder.kind", "embedder.model_name", "data.source", "data.raw_dir", "data.local_dir", "seeds",
            "chunk.max_chars", "chunk.min_chars", "bank.n_clusters", "bank.n_holdout", "stage2.eta", "pipeline.W",
            "pipeline.k_min", "fit.nu", "fit.nu_r", "fit.n_fit_sessions"]


def validate_config(cfg: dict) -> List[str]:
    problems = []
    for key in REQUIRED:
        cur = cfg
        for k in key.split("."):
            if not isinstance(cur, dict) or k not in cur:
                problems.append(f"'{key}' is missing from config.yaml (compare with the original file, or see the README)")
                break
            cur = cur[k]
    if problems:
        return problems
    if cfg["embedder"]["kind"] not in ("sbert", "hashing"):
        problems.append("embedder.kind must be sbert or hashing")
    if cfg["data"]["source"] not in ("real", "synthetic"):
        problems.append("data.source must be real or synthetic")
    if not (isinstance(cfg["seeds"], list) and cfg["seeds"] and all(isinstance(x, int) for x in cfg["seeds"])):
        problems.append("seeds must be a list of integers, e.g. [0, 1, 2, 3, 4]")
    if not (0 < cfg["fit"]["nu"] < cfg["fit"]["nu_r"] < 1):
        problems.append("0 < fit.nu < fit.nu_r < 1 is required")
    if not (0 <= cfg["bank"]["n_holdout"] < cfg["bank"]["n_clusters"]):
        problems.append("bank.n_holdout must be smaller than bank.n_clusters")
    return problems


def load_config(path: str | None = None, smoke: bool = False) -> dict:
    path = path or str(ROOT / "config.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SystemExit(f"could not read {path} (invalid YAML -- usually indentation or a tab character):\n{e}")
    cfg["_path"] = path
    if smoke:
        deep_update(cfg, cfg.get("smoke", {}))
        cfg["_smoke"] = True
    problems = validate_config(cfg)
    if problems:
        raise SystemExit("config.yaml has problems:\n  - " + "\n  - ".join(problems))
    return cfg


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--config", default=None, help="path to config.yaml (default: the one in the project folder)")
    ap.add_argument("--smoke", action="store_true", help="quick test mode (little data, 1 seed) to check the setup")
    ap.add_argument("--data", choices=["real", "synthetic"], default=None, help="data source (overrides the config)")
    ap.add_argument("--embedder", choices=["sbert", "hashing"], default=None,
                    help="sbert = the real one | hashing = fallback for testing the code (never for the paper)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="e.g. --seeds 0 1 2")
    ap.add_argument("--out", default=None, help="output folder (default: results/final/<experiment name>)")
    ap.add_argument("--skip-baselines", nargs="*", default=[], help="e.g. --skip-baselines B3a B3b")
    return ap


def _sub(cls, d: dict):
    names = {f.name for f in fields(cls)}
    kw = {}
    for k, v in (d or {}).items():
        if k in names:
            kw[k] = tuple(v) if isinstance(v, list) else v
    return cls(**kw)


def stage2_config(cfg: dict) -> Stage2Config:
    return _sub(Stage2Config, cfg.get("stage2", {}))


def pipe_config(cfg: dict) -> PipeConfig:
    return _sub(PipeConfig, cfg.get("pipeline", {}))


def fit_config(cfg: dict) -> FitConfig:
    return _sub(FitConfig, cfg.get("fit", {}))


def setup(args, need: Tuple[str, ...]) -> Tuple[dict, World, CachedEmbedder]:
    cfg = load_config(args.config, smoke=args.smoke)
    if args.data:
        cfg["data"]["source"] = args.data
    if args.embedder:
        cfg["embedder"]["kind"] = args.embedder
    if args.seeds is not None and len(args.seeds):
        cfg["seeds"] = args.seeds
    if not cfg["data"].get("use_tasktracker", True):
        need = tuple(n for n in need if n != "tasktracker")
    base = Path(cfg["_path"]).parent
    raw = Path(cfg["data"]["raw_dir"])
    loc = Path(cfg["data"]["local_dir"])
    raw = raw if raw.is_absolute() else base / raw
    loc = loc if loc.is_absolute() else base / loc
    log(f"data source: {cfg['data']['source']} | embedder: {cfg['embedder']['kind']} | seeds: {cfg['seeds']}"
        f"{' | SMOKE' if cfg.get('_smoke') else ''}")
    world = load_world(str(raw), str(loc), cfg["data"]["source"], need=need, tasks=cfg["data"]["bipia_tasks"],
                       tasktracker_max_rows=cfg["data"].get("tasktracker_max_rows", 4000),
                       english_only=cfg["data"].get("english_only", True))
    log(world.summary())
    if cfg["data"].get("english_only", True) and lang_backend() != "lingua":
        log("**WARNING: lingua-language-detector not found -> the language filter runs on rules only and does not "
            "drop all the German in deepset. Results must not go into the paper.**  Fix: pip install -r requirements.txt")
    emb = get_embedder(cfg["embedder"]["kind"], cfg["embedder"]["model_name"],
                       cfg["embedder"].get("allow_fallback", False))
    log(f"embedder = {emb.name} (dim {emb.dim})" + ("  **fallback: results must not go into the paper**" if emb.is_fallback else ""))
    return cfg, world, emb


def prepare_seed(cfg: dict, world: World, emb: CachedEmbedder, seed: int) -> Tuple[Splits, AttackBank]:
    """Splits the data for this seed -> attack bank from dev only -> held-out family examples go to val/test"""
    dedup = bool(cfg.get("data", {}).get("dedup", False))
    sp = make_splits(world, seed, dedup=dedup)
    if dedup:
        from .dedup_splits import leak_gate
        tx = lambda L: [x["text"] if isinstance(x, dict) and "text" in x else (x["context"] if isinstance(x, dict) else x) for x in L]  # noqa: E731
        strict = {"att dev/val": (tx(sp.att["dev"]), tx(sp.att["val"])), "att dev/test": (tx(sp.att["dev"]), tx(sp.att["test"])),
                  "ben dev/val": (tx(sp.ben["dev"]), tx(sp.ben["val"])), "ben dev/test": (tx(sp.ben["dev"]), tx(sp.ben["test"])),
                  "att(dev) vs ben(test)": (tx(sp.att["dev"]), tx(sp.ben["test"])), "ben(dev) vs att(test)": (tx(sp.ben["dev"]), tx(sp.att["test"]))}
        lenient = {"bipia ctx dev/test": (tx(sp.bipia["dev"]["contexts"]), tx(sp.bipia["test"]["contexts"])),
                   "bipia ctx val/test": (tx(sp.bipia["val"]["contexts"]), tx(sp.bipia["test"]["contexts"])),
                   "tasktracker dev/test": (tx(sp.tt.get("dev", [])), tx(sp.tt.get("test", [])))}
        rep = leak_gate(strict)                                   # exact AND near duplicates stop the run
        rep.update(leak_gate(lenient, max_exact=0, max_near=10 ** 9))   # exact duplicates stop the run; near ones are logged
        log(f"seed {seed}: leak gate passed | " + "; ".join(f"{k}: exact={v['exact']} near={v['near']}/{v['n_eval']}" for k, v in rep.items()))
    retr = KBRetriever(emb)                      # the real top-k retriever over the mock KB, one index per split
    for split in ("val", "test"):
        retr.index(sp.kb[split])
    set_retriever(retr)
    b = cfg["bank"]
    bank, held = build_bank([d["text"] for d in sp.att["dev"]], emb, n_clusters=b["n_clusters"],
                            n_holdout=b["n_holdout"], seed=seed, extra=HAND_ATTACKS)
    attach_heldout(sp, held)
    log(f"seed {seed}: {sp.describe()} | bank={len(bank)} (held out {len(held)} examples from {b['n_holdout']} families)")
    return sp, bank


def build_full_system(cfg: dict, world: World, emb: CachedEmbedder, seed: int, ext=None, **pipe_overrides):
    """Fits the full system (Stages 1-4 + LR) on this seed's dev/val -> (detector, splits, bank, val_a, val_b, info)"""
    sp, bank = prepare_seed(cfg, world, emb, seed)
    fit_sess = fitting_sessions(sp, "val", seed, cfg["fit"]["n_fit_sessions"])
    val_a, val_b = split_sessions(fit_sess, 0.5, seed)
    pc = pipe_config(cfg)
    if pipe_overrides:
        from dataclasses import replace
        pc = replace(pc, **pipe_overrides)
    det, info = fit_system(emb, bank, head_training_data(sp, HAND_ATTACKS, seed), val_a, val_b, stage2_config(cfg),
                           pc, fit_config(cfg), seed, ext=ext)
    log(f"seed {seed}: full system fitted | t_blk={det.cfg.t_blk:.3f} t_rev={det.cfg.t_rev:.3f} "
        f"τ_f={det.s2.tau_f:.3f} τ_same={det.cfg.tau_same:.3f} τ_a={det.cfg.tau_a:.3f} ρ={det.cfg.rho:.3f}")
    return det, sp, bank, val_a, val_b, info


def out_dir(args, name: str) -> str:
    d = args.out or str(ROOT / "results" / "final" / name)
    os.makedirs(d, exist_ok=True)
    return d


def write_run_info(d: str, cfg: dict, world: World, emb, extra: Dict | None = None) -> None:
    info = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), config_path=cfg.get("_path"), smoke=bool(cfg.get("_smoke")),
                data_source=cfg["data"]["source"], embedder=emb.name, embedder_is_fallback=bool(emb.is_fallback),
                seeds=cfg["seeds"], world=world.summary(), english_only=bool(cfg["data"].get("english_only", True)),
                language_filter=getattr(world, "language_filter", {}), language_filter_backend=lang_backend(),
                python=sys.version.split()[0])
    if extra:
        info.update(extra)
    with open(os.path.join(d, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    bad_lang = cfg["data"].get("english_only", True) and lang_backend() != "lingua"
    if emb.is_fallback or cfg["data"]["source"] == "synthetic" or bad_lang:
        with open(os.path.join(d, "WARNING_NOT_FOR_PAPER.txt"), "w", encoding="utf-8") as f:
            f.write("These results come from synthetic data and/or the fallback embedder -> they only show that the code runs. Not for the paper.\n")
