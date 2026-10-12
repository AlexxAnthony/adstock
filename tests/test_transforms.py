import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adstock.transforms import (adstock_weights, geometric_adstock_np,  # noqa: E402
                                half_life, logistic_saturation_np)


def test_weights_sum_to_one():
    assert np.allclose(adstock_weights(np.array([0.1, 0.5, 0.9])).sum(-1), 1)


def test_adstock_conserves_spend_and_shifts_forward():
    x = np.zeros((20, 1)); x[2, 0] = 100
    out = geometric_adstock_np(x, np.array([0.5]))
    assert np.isclose(out.sum(), 100)        # nothing created or lost
    assert out[:2].sum() == 0                # no effect before the spend
    assert out[2, 0] > out[3, 0] > out[4, 0]  # decays afterwards


def test_adstock_zero_alpha_is_identity():
    x = np.random.default_rng(0).random((30, 3))
    assert np.allclose(geometric_adstock_np(x, np.zeros(3)), x)


def test_adstock_draw_axis_matches_single():
    x = np.random.default_rng(1).random((15, 2))
    a = np.array([[0.2, 0.7], [0.5, 0.1]])
    batched = geometric_adstock_np(x, a)
    for d in range(2):
        assert np.allclose(batched[d], geometric_adstock_np(x, a[d]))


def test_saturation_is_concave_and_bounded():
    x = np.linspace(0, 10, 50)
    s = logistic_saturation_np(x, 1.5)
    assert s[0] == 0 and s.max() < 1
    assert np.all(np.diff(s, 2) <= 1e-12)    # diminishing returns


def test_half_life():
    assert np.isclose(half_life(0.5), 1.0)
