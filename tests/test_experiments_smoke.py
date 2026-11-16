"""Runs every experiment script at its smallest size on synthetic data (~1 minute). Skip with  pytest -m "not slow" """
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["experiments/h1_behavior_vs_memoryless.py", "experiments/h2_instruction_relevance.py",
           "experiments/h3_authority_principle.py", "experiments/s_component_checks.py",
           "experiments/prelim_decision_model.py"]


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script, tmp_path):
    extra = ["--no-permutation"] if "prelim" in script else []
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), "--smoke", "--data", "synthetic", "--embedder",
                        "hashing", "--out", str(tmp_path), *extra], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert os.path.exists(os.path.join(tmp_path, "summary.md"))
    assert os.path.exists(os.path.join(tmp_path, "WARNING_NOT_FOR_PAPER.txt"))
