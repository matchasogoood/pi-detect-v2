import numpy as np

from pidetect.stats import corrected_ttest, holm, mcnemar, paired_bootstrap


def test_paired_bootstrap_detects_clear_gain():
    a = np.array([1] * 80 + [0] * 20)
    b = np.array([1] * 40 + [0] * 60)
    r = paired_bootstrap(a, b)
    assert r["diff"] > 0 and r["excludes_zero"]


def test_paired_bootstrap_no_difference():
    a = np.array([1, 0] * 50)
    r = paired_bootstrap(a, a.copy())
    assert r["diff"] == 0 and not r["excludes_zero"]


def test_holm_is_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0


def test_mcnemar_and_ttest():
    assert mcnemar([1, 1, 1, 0], [1, 1, 1, 0]) == 1.0
    m, se, p = corrected_ttest([0.1, 0.12, 0.09, 0.11, 0.1], 500, 100)
    assert m > 0 and se > 0 and p < 0.05
