"""Test that evaluate() returns consumable results when p_tqdm is available.

Regression test for https://github.com/SALib/SALib/issues/676
Without the fix, p_imap returns a generator instead of a list, causing
TypeError when _collect_results() tries to consume it.
"""

import numpy as np
import pytest
from unittest import mock

from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def _make_spec():
    """Create a basic ProblemSpec with samples."""
    sp = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]],
            "outputs": ["Y"],
        }
    )
    sp.sample_saltelli(16, calc_second_order=False)
    return sp


def test_evaluate_with_ptqdm_returns_consumable_results():
    """p_imap result should be list-wrapped for consistent consumption.

    Before the fix, p_imap returned a generator that couldn't be properly
    consumed by _collect_results(), causing TypeError.
    """
    sp = _make_spec()

    def fake_p_imap(func, chunks, num_cpus=1):
        """Mimic p_imap: apply func to each chunk, yield results."""
        for chunk in chunks:
            yield func(chunk)

    # Inject p_imap into the module namespace (it may not exist if p_tqdm is not installed)
    import SALib.util.problem as problem_mod
    problem_mod.p_imap = fake_p_imap
    problem_mod.ptqdm_available = True

    try:
        result = sp.evaluate(Ishigami.evaluate, nprocs=2)
        assert result.results is not None
    finally:
        # Restore original state
        problem_mod.ptqdm_available = False
        if hasattr(problem_mod, 'p_imap'):
            delattr(problem_mod, 'p_imap')


def test_evaluate_without_ptqdm_still_works():
    """Verify evaluate() still works when p_tqdm is not available."""
    sp = _make_spec()

    with mock.patch("SALib.util.problem.ptqdm_available", False):
        result = sp.evaluate(Ishigami.evaluate, nprocs=2)

    assert result.results is not None
    assert result.results.size > 0


def test_evaluate_single_core_no_ptqdm():
    """Single-core evaluate should always work regardless of p_tqdm."""
    sp = _make_spec()

    with mock.patch("SALib.util.problem.ptqdm_available", False):
        result = sp.evaluate(Ishigami.evaluate, nprocs=1)

    assert result.results is not None
