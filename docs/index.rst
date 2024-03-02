==============================================
SALib - Sensitivity Analysis Library in Python
==============================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.160164.svg
   :target: https://doi.org/10.5281/zenodo.160164

.. image:: https://img.shields.io/badge/DOI-10.18174%2Fsesmo.18155-blue
  :target: https://sesmo.org/article/view/18155

.. image:: https://joss.theoj.org/papers/10.21105/joss.00097/status.svg
  :target: https://doi.org/10.21105/joss.00097


Python implementations of commonly used sensitivity analysis methods, including
Sobol, Morris, and FAST methods. Useful in systems modeling to calculate the
effects of model inputs or exogenous factors on outputs of interest.

Supported Methods
-----------------
* Sobol Sensitivity Analysis
  (`Sobol 2001 <http://www.sciencedirect.com/science/article/pii/S0378475400002706>`_, `Saltelli 2002 <http://www.sciencedirect.com/science/article/pii/S0010465502002801>`_, `Saltelli et al. 2010 <http://www.sciencedirect.com/science/article/pii/S0010465509003087>`_)
* Method of Morris, including groups and optimal trajectories
  (`Morris 1991 <http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804>`_, `Campolongo et al. 2007 <http://www.sciencedirect.com/science/article/pii/S1364815206002805>`_)
* Fourier Amplitude Sensitivity Test (FAST)
  (`Cukier et al. 1973 <http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571>`_, `Saltelli et al. 1999 <http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594>`_)
* Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST)
  (`Tarantola et al. 2006 <https://hal.archives-ouvertes.fr/hal-01065897/file/Tarantola06RESS_HAL.pdf>`_, `Elmar Plischke 2010 <https://doi.org/10.1016/j.ress.2009.11.005>`_, `Tissot et al. 2012 <https://doi.org/10.1016/j.ress.2012.06.010>`_)
* Delta Moment-Independent Measure
  (`Borgonovo 2007 <http://www.sciencedirect.com/science/article/pii/S0951832006000883>`_, `Plischke et al. 2013 <http://www.sciencedirect.com/science/article/pii/S0377221712008995>`_)
* Derivative-based Global Sensitivity Measure (DGSM)
  (`Sobol and Kucherenko 2009 <http://www.sciencedirect.com/science/article/pii/S0378475409000354>`_)
* Fractional Factorial Sensitivity Analysis
  (`Saltelli et al. 2008 <http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html>`_)
* High Dimensional Model Representation
  (`Li et al. 2010 <https://pubs.acs.org/doi/pdf/10.1021/jp9096919>`_)
* PAWN
  (`Pianosi and Wagener 2018 <https://dx.doi.org/10.1016/j.envsoft.2018.07.019>`__, `Pianosi and Wagener 2015 <https://doi.org/10.1016/j.envsoft.2015.01.004>`__)
* Regional Sensitivity Analysis
  (based on `Hornberger and Spear, 1981 <https://www.osti.gov/biblio/6396608-approach-preliminary-analysis-environmental-systems>`__, `Saltelli et al. 2008 <https://dx.doi.org/10.1002/9780470725184>`__, `Pianosi et al., 2016 <https://dx.doi.org/10.1016/j.envsoft.2016.02.008>`__)


Getting Started
---------------

.. toctree::
   :maxdepth: 2

   Getting started <user_guide/getting-started>
   Basics <user_guide/basics>
   SALib Interface <user_guide/basics_with_interface>
   Advanced <user_guide/advanced>
   Wrappers <user_guide/wrappers>
   FAQ <user_guide/faq>


For Developers
--------------

.. toctree::
    :maxdepth: 2

    API <api>
    Developers Guide <developers_guide>
    Changes <changelog>
    Complete Module Reference <api/modules>


Other Info
----------

.. toctree::
    :maxdepth: 2

    License <license>
    Authors <authors>
    Projects that use SALib <citations>
