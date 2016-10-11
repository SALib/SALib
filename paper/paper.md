---
title: 'SALib: An open-source Python library for Sensitivity Analysis'
tags:
  - sensitivity analysis
  - Python
  - uncertainty
  - variance-based
  - global sensitivity analysis
  - fractional factorial
  - Method of Morris
authors:
 - name: Jon Herman
   orcid: 0000-0002-4081-3175
   affiliation: 1
 - name: Will Usher
   orcid: 0000-0001-9367-1791
   affiliation: 2
affiliations:
 - name: UCDavis
   index: 1
 - name: University of Oxford
   index: 2
date: 11th October 2016
bibliography: paper.bib
---

# Summary

<!--Statement of need-->

SALib contains Python implementations of commonly used global sensitivity
analysis methods, including Sobol [@Sobol2001,@Saltelli2002a,@Saltelli2010a],
Morris [@Morris1991,@Campolongo2007], FAST [@Cukier1973,@Saltelli1999],
Delta Moment-Independent Measure [@Borgonovo2007,@Plischke2013]
Derivative-based Global Sensitivity Measure (DGSM) [@Sobol2009]
, and Fractional Factorial Sensitivity Analysis [@Saltelli2008b]
 methods.
SALib is useful in simulation, optimisation and systems modelling to calculate
the influence of model inputs or exogenous factors on outputs of interest.

<!--Target Audience-->

SALib exposes a range of global sensitivity analysis techniques to the
scientist, researcher and modeller, making it very easy to easily implement
the range of techniques into typical modelling workflows.

The library facilitates the generation of samples associated with a model's
inputs, and then provides functions to analyse the outputs from a model and
visualise those results.

# References
