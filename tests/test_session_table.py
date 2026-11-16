"""The grouped session_table must give exactly what the per-session loop gives (the loop is kept as the reference)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pidetect.evaluate import _session_table_slow, session_table  # noqa: E402


def _events(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method in ("a", "b"):
        for sd in (0, 1):
            for sid in range(25):
                n = int(rng.integers(2, 9))
                attack = bool(rng.random() < 0.5)
                start = int(rng.integers(0, n))
                for i in range(n):
                    lab = int(attack and i >= start)
                    rows.append(dict(method=method, seed=sd, sid=f"s{sid}", idx=i, label=lab,
                                     blocked=int(rng.random() < (0.4 if lab else 0.1)),
                                     phase="attack" if (attack and i >= start and rng.random() < 0.8) else "setup",
                                     se_attack_kind="k1" if attack else np.nan, se_subset="x"))
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed)      # shuffled: the order must not matter


def _assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Every column equal.  Numeric columns are compared as floats (NaN counts as equal to NaN), everything else
    as text -- which dtype pandas picks for a string column differs between versions, so we ask what the column
    holds rather than what its dtype object is."""
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    a = a[list(b.columns)]
    for c in b.columns:
        if is_numeric_dtype(a[c]) and is_numeric_dtype(b[c]):
            assert np.allclose(a[c].astype(float).values, b[c].astype(float).values, equal_nan=True), c
        else:
            # object, not str: converting an Arrow-backed string column with astype(str) turns a missing value
            # into a real NaN, and NaN != NaN would call two equally-missing cells different
            x, y = a[c].astype(object).values, b[c].astype(object).values
            both_missing = a[c].isna().values & b[c].isna().values
            assert ((x == y) | both_missing).all(), c


def test_grouped_equals_loop():
    df = _events()
    _assert_same(session_table(df), _session_table_slow(df))


def test_grouped_equals_loop_on_a_real_run():
    """The same check on real events, if a run is present: the synthetic table above cannot cover every column
    combination a real experiment writes.  Skipped when no results folder exists yet."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = [os.path.join(r, f) for d in os.listdir(root) if d.startswith("results")
             for r, _, fs in os.walk(os.path.join(root, d)) for f in fs if f.endswith("events.csv.gz")]
    if not found:
        pytest.skip("no results/*/events.csv.gz to check against")
    df = pd.read_csv(sorted(found)[0], low_memory=False).head(60000)
    _assert_same(session_table(df), _session_table_slow(df))


if __name__ == "__main__":
    test_grouped_equals_loop()
    test_grouped_equals_loop_on_a_real_run()
    print("ok")
