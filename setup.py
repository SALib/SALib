#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for SALib.

    This file was generated with PyScaffold 3.0.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: http://pyscaffold.org/
"""

import os
import sys
from setuptools import setup

scripts = ['src/SALib/scripts/salib.py']
if os.name == 'nt':
    scripts.append('src/SALib/scripts/salib.bat')

# Add here console scripts and other entry points in ini-style format
entry_points = """
# Add here console scripts such as:
[console_scripts]
salib=SALib.scripts.salib:main
"""


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['pyscaffold>=3.0a0,<3.1a0'] + sphinx,
          entry_points=entry_points,
          scripts=scripts,
          use_pyscaffold=True)


if __name__ == "__main__":
    setup_package()
