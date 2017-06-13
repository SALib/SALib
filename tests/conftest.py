#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for salib.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import print_function, absolute_import, division
import tempfile
import pytest


def make_temporary_file():
    """ Returns a temporary file name

    Returns
    =========
    openfile.name : str
        Name of the temporary file
    """
    with tempfile.NamedTemporaryFile() as openfile:
        return openfile.name


@pytest.fixture(scope='function')
def setup_param_file():
    filename = make_temporary_file()
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0\n")
        ofile.write("Test 2,0,1.0\n")
        ofile.write("Test 3,0,1.0\n")
    return filename


@pytest.fixture(scope='function')
def setup_param_file_with_groups():
    filename = make_temporary_file()
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0,Group 1\n")
        ofile.write("Test 2,0,1.0,Group 1\n")
        ofile.write("Test 3,0,1.0,Group 2\n")
    return filename


@pytest.fixture(scope='function')
def setup_param_file_with_groups_prime():
    filename = make_temporary_file()
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0,Group 1\n")
        ofile.write("Test 2,0,1.0,Group 2\n")
        ofile.write("Test 3,0,1.0,Group 2\n")
        ofile.write("Test 4,0,1.0,Group 3\n")
        ofile.write("Test 5,0,1.0,Group 3\n")
        ofile.write("Test 6,0,1.0,Group 3\n")
        ofile.write("Test 7,0,1.0,Group 3\n")
    return filename
