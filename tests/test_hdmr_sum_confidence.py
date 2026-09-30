from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special

from SALib.analyze import enhanced_hdmr, hdmr
from SALib.util import ResultDict


def _bootstrap_indices(count):
    sa = np.array([[0.20, 0.30, 0.40], [0.10, 0.15, 0.20]])[:, :count]
    sb = sa / 10
    return sa, sb, sa + sb


def _confidence(values, alpha=0.95):
    multiplier = -np.sqrt(2) * special.erfcinv((alpha + (1 - alpha) / 2) * 2)
    return multiplier * np.std(np.sum(values, axis=0))


@pytest.mark.parametrize("bootstrap", [1, 3])
def test_enhanced_hdmr_sum_confidence_varies_across_bootstraps(bootstrap):
    sa, sb, total = _bootstrap_indices(bootstrap)
    si = ResultDict(
        {
            "Sa": sa.copy(),
            "Sb": sb.copy(),
            "S": total.copy(),
            "Sa_conf": np.zeros(2),
            "Sb_conf": np.zeros(2),
            "S_conf": np.zeros(2),
            "Sa_sum": 0.0,
            "Sb_sum": 0.0,
            "S_sum": 0.0,
            "Sa_sum_conf": 0.0,
            "Sb_sum_conf": 0.0,
            "S_sum_conf": 0.0,
            "ST": np.zeros(2),
            "ST_conf": np.zeros(2),
            "Signf": np.ones((2, bootstrap)),
            "Term": ["x1", "x2"],
        }
    )
    model = SimpleNamespace(d=2, max_order=1, S=total)

    result = enhanced_hdmr._finalize(model, si, alpha=0.95, return_emulator=False)

    assert_allclose(result["Sa_sum_conf"], _confidence(sa))
    assert_allclose(result["Sb_sum_conf"], _confidence(sb))
    assert_allclose(result["S_sum_conf"], _confidence(total))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("bootstrap", [1, 3])
def test_hdmr_sum_confidence_varies_across_bootstraps(bootstrap):
    sa, sb, total = _bootstrap_indices(bootstrap)
    sensitivity = {"Sa": sa.copy(), "Sb": sb.copy(), "S": total.copy()}
    emulator = {
        "n": 2,
        "n1": 2,
        "n2": 0,
        "n3": 0,
        "select": np.ones((2, bootstrap)),
    }

    result = hdmr._finalize(
        {"names": ["x1", "x2"]},
        sensitivity,
        emulator,
        d=2,
        alpha=0.95,
        maxorder=1,
        RT=None,
        Y_em=None,
        bootstrap_idx=None,
        X=None,
        Y=None,
    )

    assert_allclose(result["Sa_sum_conf"], _confidence(sa))
    assert_allclose(result["Sb_sum_conf"], _confidence(sb))
    assert_allclose(result["S_sum_conf"], _confidence(total))
