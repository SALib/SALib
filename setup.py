# -*- coding: utf-8 -*-
"""
    Setup file for SALib.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import os
import sys

from setuptools import setup

scripts = ['src/SALib/scripts/salib.py']
if os.name == 'nt':
    scripts.append('src/SALib/scripts/salib.bat')

if __name__ == "__main__":
    try:
        setup(use_scm_version={"version_scheme": "no-guess-dev"},
              scripts=scripts)
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
