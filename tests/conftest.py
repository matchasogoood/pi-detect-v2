"""Fixtures shared by every test file -- synthetic data and the fallback embedder (nothing is downloaded)"""
import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def emb():
    from pidetect.embedder import get_embedder
    return get_embedder("hashing")


@pytest.fixture(scope="session")
def world():
    from pidetect.data.world import default_dirs, load_world
    raw, local = default_dirs(ROOT)
    return load_world(raw, local, source="synthetic", need=("bipia", "deepset", "gandalf", "notinject", "tasktracker"))


@pytest.fixture(scope="session")
def system(world, emb):
    """The full system, fitted on synthetic data (shared by several tests)"""
    from pidetect.runtime import build_full_system, load_config
    cfg = load_config(os.path.join(ROOT, "config.yaml"), smoke=True)
    det, sp, bank, va, vb, info = build_full_system(cfg, world, emb, 0)
    return dict(cfg=cfg, det=det, sp=sp, bank=bank, va=va, vb=vb, info=info)
