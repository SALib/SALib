import pytest
import os
from os.path import join as pth_join
import subprocess


salib_cli = "salib"
ishigami_fp = "./src/SALib/test_functions/params/Ishigami.txt"
test_file = 'test.txt'
test_data = pth_join('./tests', 'data', test_file)


def teardown_function(func):
    # Removes the test file if it was created.
    files = os.listdir('./tests/data')
    if test_file in files:
        os.remove(test_data)


def test_cli_entry():
    cmd = '{cli} -h'.format(cli=salib_cli).split()
    result = subprocess.check_output(cmd)
    assert 'Errno' not in str(result), "Error occurred when trying to use CLI!"


def test_ff():
    cmd = "{cli} sample ff -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_fast():
    cmd = "{cli} sample fast_sampler -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_finite_diff():
    cmd = "{cli} sample finite_diff -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_latin():
    cmd = "{cli} sample latin -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_saltelli():
    cmd = "{cli} sample saltelli -p {fn} -o {test_data} -n 512".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_saltelli_warning():
    # Ensure error is raised when n_samples not a power of 2
    cmd = f"salib sample saltelli -p {ishigami_fp} -o {test_data} -n 511".split()
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    assert "Warning" in str(result)

    # Ensure warning is raised when skip_values < n_samples
    cmd = f"salib sample saltelli -p {ishigami_fp} -o {test_data} -n 2048 --skip-values 1024".split()
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    assert "Warning" in str(result)

    # Ensure error is raised when skip_values not a power of 2
    cmd = f"salib sample saltelli -p {ishigami_fp} -o {test_data} -n 512 --skip-values 1025".split()
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    assert "Warning" in str(result)


def test_sobol_corr():
    cmd = "python {cli} sample sobol_corr -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_shapley():
    cmd = """python {cli} sample shapley -p {fn} -o {test_data}
        -n 1000
        --n-outer 100
        --n-inner 100
        """.format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


if __name__ == '__main__':
    test_cli_entry()
    test_ff()
    test_fast()
    test_finite_diff()
    test_latin()
    test_saltelli()
    test_sobol_corr()
    test_shapley()
