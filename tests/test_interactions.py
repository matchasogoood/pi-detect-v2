"""Channel interactions in the decision model, and the log-odds / single-channel options of the external score."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pidetect.baselines import EXT_MODES, ExternalContentScore, ext_pipe_overrides  # noqa: E402
from pidetect.pipeline import FEATURES, PipeConfig, ProposedDetector  # noqa: E402


def _x(**kw):
    x = {f: 0.0 for f in FEATURES}
    x.update(kw)
    return x


def test_vector_without_interactions_is_unchanged():
    d = ProposedDetector(None, PipeConfig(), "t")
    x = _x(s=1.5, c=1.0, n_imp=2.0, n_ai=3.0)
    assert d.feature_names == list(FEATURES)
    assert np.allclose(d.vector(x), [x[f] for f in FEATURES])


def test_interaction_columns_are_zero_on_direct_and_copy_on_indirect():
    ix = ("s", "n_imp", "n_ai")
    d = ProposedDetector(None, PipeConfig(interactions=ix), "t")
    assert d.feature_names == list(FEATURES) + ["s:ind", "n_imp:ind", "n_ai:ind"]
    direct = d.vector(_x(s=1.5, c=0.0, n_imp=2.0, n_ai=3.0))
    indirect = d.vector(_x(s=1.5, c=1.0, n_imp=2.0, n_ai=3.0))
    assert np.allclose(direct[-3:], [0.0, 0.0, 0.0])
    assert np.allclose(indirect[-3:], [1.5, 2.0, 3.0])
    assert np.allclose(direct[: len(FEATURES)], indirect[: len(FEATURES)] * 0 + direct[: len(FEATURES)])


def test_interaction_of_a_feature_the_variant_dropped_is_skipped():
    d = ProposedDetector(None, PipeConfig(features=("s", "c"), interactions=("s", "n_imp")), "t")
    assert d.feature_names == ["s", "c", "s:ind"]
    assert len(d.vector(_x(s=1.0, c=1.0))) == 3


class _Bl:
    name = "fake"

    def __init__(self, v):
        self.v = v

    def score(self, texts):
        return np.array([self.v] * len(texts), dtype=float)


def test_only_channel_leaves_the_other_channel_to_us():
    e = ExternalContentScore(_Bl(0.9), only_channel="indirect")
    assert e.fuse(0.123, "t", "direct") == 0.123
    assert abs(e.fuse(0.123, "t", "indirect") - 0.9) < 1e-9


def test_logit_is_monotone_and_bounded():
    e = ExternalContentScore(_Bl(0.0), logit=True)
    vals = [e._scale(v) for v in (0.0, 1e-9, 0.01, 0.5, 0.99, 0.999999, 1.0)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals          # never reorders two scores
    assert abs(vals[0]) <= ExternalContentScore.LOGIT_CLIP + 1e-6
    assert abs(vals[-1]) <= ExternalContentScore.LOGIT_CLIP + 1e-6
    # strictly increasing over everything a float32 softmax can distinguish, so no tie is created at the top,
    # which is the region a low false-positive threshold reads
    fine = [e._scale(v) for v in (1e-6, 1e-3, 0.5, 0.999, 1 - 1e-5, 1 - 1e-6)]
    assert all(b > a for a, b in zip(fine, fine[1:])), fine
    assert abs(e._scale(0.5)) < 1e-9
    assert abs(e._scale(0.9) - math.log(9.0)) < 1e-9


def test_mode_table_and_pipe_overrides():
    assert ext_pipe_overrides("ind_win") == {}
    assert ext_pipe_overrides("ind_win_logit")["interactions"] == ("s", "n_imp", "n_ai")
    for tag in EXT_MODES:
        ExternalContentScore.from_mode(_Bl(0.5), tag)     # every listed mode must build


def test_both_channel_modes_use_the_guard_model_on_direct_too():
    e = ExternalContentScore.from_mode(_Bl(0.9), "ext_win_logit")
    for ch in ("direct", "indirect"):
        assert abs(e.fuse(0.123, "t", ch) - math.log(9.0)) < 1e-9


def test_ensemble_is_the_mean_of_the_two_log_odds():
    e = ExternalContentScore.from_mode(_Bl(0.9), "ens_win_logit")
    # our score 1.0 with bias -2.0 is log-odds -1.0; the guard model's is log 9
    assert abs(e.fuse(1.0, "t", "direct", ours_offset=-2.0) - 0.5 * (-1.0 + math.log(9.0))) < 1e-9
    assert e.fuse(1.5, "t", "indirect") > e.fuse(0.5, "t", "indirect")      # monotone in our own score
    assert abs(e.fuse(1e6, "t", "direct")) <= ExternalContentScore.LOGIT_CLIP + 1e-9   # clipped


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
