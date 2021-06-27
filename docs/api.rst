=====================
Concise API Reference
=====================

This page documents the sensitivity analysis methods supported by SALib.

FAST - Fourier Amplitude Sensitivity Test
-----------------------------------------

.. autofunction:: SALib.sample.fast_sampler.sample
   :noindex:

.. autofunction:: SALib.analyze.fast.analyze
   :noindex:

RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test
--------------------------------------------------------------------

.. autofunction:: SALib.sample.latin.sample
   :noindex:

.. autofunction:: SALib.analyze.rbd_fast.analyze
   :noindex:

Method of Morris
----------------

.. autofunction:: SALib.sample.morris.sample
   :noindex:

.. autofunction:: SALib.analyze.morris.analyze
   :noindex:

Sobol Sensitivity Analysis
--------------------------

.. autofunction:: SALib.sample.saltelli.sample
   :noindex:

.. autofunction:: SALib.analyze.sobol.analyze
   :noindex:

Delta Moment-Independent Measure
--------------------------------

.. autofunction:: SALib.sample.latin.sample
   :noindex:

.. autofunction:: SALib.analyze.delta.analyze
   :noindex:

Derivative-based Global Sensitivity Measure (DGSM)
--------------------------------------------------

.. autofunction:: SALib.analyze.dgsm.analyze
   :noindex:

Fractional Factorial
--------------------

.. autofunction:: SALib.sample.ff.sample
   :noindex:

.. autofunction:: SALib.analyze.ff.analyze
   :noindex:

High-Dimensional Model Representation
-------------------------------------

.. autofunction:: SALib.analyze.ff.hdmr
   :noindex:
