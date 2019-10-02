# -*- coding: utf-8 -*-
"""
    Setup file for salib.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import os
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

scripts = ['src/SALib/scripts/salib.py']
if os.name == 'nt':
    scripts.append('src/SALib/scripts/salib.bat')

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(use_pyscaffold=True, scripts=scripts)
