=============
API Reference
=============

This page documents the sensitivity analysis methods supported by SALib.

FAST - Fourier Amplitude Sensitivity Test
-----------------------------------------

.. autofunction:: SALib.sample.fast_sampler.sample

.. autofunction:: SALib.analyze.fast.analyze

Method of Morris
----------------

.. autofunction:: SALib.sample.morris.sample

.. autofunction:: SALib.analyze.morris.analyze

Sobol Sensitivity Analysis
--------------------------

.. autofunction:: SALib.sample.saltelli.sample

.. autofunction:: SALib.analyze.sobol.analyze

Delta Moment-Independent Measure
--------------------------------

.. autofunction:: SALib.sample.latin.sample

.. autofunction:: SALib.analyze.delta.analyze

Derivative-based Global Sensitivity Measure (DGSM)
--------------------------------------------------

.. autofunction:: SALib.analyze.dgsm.analyze
