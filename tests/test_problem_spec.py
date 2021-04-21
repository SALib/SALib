import pytest

import numpy as np

from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def test_sp_bounds():
    """Ensure incorrect user-defined bounds raises AssertionErrors."""
    with pytest.raises(AssertionError) as e_info:
        ProblemSpec()

    with pytest.raises(AssertionError) as e_info:
        ProblemSpec({
            'names': ['x1', 'x2', 'x3'],
            'groups': None,
            'outputs': ['Y']
        })

    with pytest.raises(AssertionError) as e_info:
        ProblemSpec({
            'names': ['x1', 'x2', 'x3'],
            'groups': None,
            'bounds': [[-np.pi, np.pi]*3],
            'outputs': ['Y']
        })


def test_sp():
    """Ensure basic usage works without raising any errors."""

    sp = ProblemSpec({
        'names': ['x1', 'x2', 'x3'],
        'groups': None,
        'bounds': [[-np.pi, np.pi]]*3,
        'outputs': ['Y']
    })

    (sp.sample_saltelli(128, calc_second_order=True)
       .evaluate(Ishigami.evaluate)
       .analyze_sobol(calc_second_order=True, conf_level=0.95))


def test_sp_setters():
    """Test sample and result setters."""

    sp = ProblemSpec({
        'names': ['x1', 'x2', 'x3'],
        'groups': None,
        'bounds': [[-np.pi, np.pi]]*3,
        'outputs': ['Y']
    })

    nvars = sp['num_vars']
    N = 128
    X1 = sp.sample_saltelli(N, calc_second_order=True).samples

    # Saltelli produces `N*(2p+2)` samples when producing samples for 2nd order analysis
    expected_samples = 128*(2*nvars+2)

    assert X1.shape[0] == expected_samples, "Number of samples is not as expected"

    # Get another 100 samples
    X2 = sp.sample_saltelli(128, calc_second_order=True).samples
    assert np.all(sp.samples == X2), "Stored sample and extracted samples are not identical!"

    # Test property setter
    # this approach actually overrides the underlying sp._samples attribute
    sp.samples = X1
    assert np.all(sp.samples == X1), "Reverting samples to original set did not work!"

    # Setting samples should clear results attribute
    sp.set_samples(X2)
    Y2 = sp.evaluate(Ishigami.evaluate).results
    sp.set_samples(X1)
    assert sp.results is None, "Results were not cleared after new sample values were set!"

    # Entire process should not raise errors
    (sp.sample_saltelli(128, calc_second_order=True)
       .set_samples(X2)
       .set_results(Y2)
       .analyze_sobol(calc_second_order=True, conf_level=0.95))


if __name__ == '__main__':
    test_sp_setters()

