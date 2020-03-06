import os
from os.path import join as pth_join
import subprocess
import pytest


salib_cli = "./src/SALib/scripts/salib.py"
ishigami_fp = "./src/SALib/test_functions/params/Ishigami.txt"
test_data = pth_join('tests', 'data', 'test.txt')


def test_cli_entry():
    cmd = 'python {cli} -h'.format(cli=salib_cli).split()
    result = subprocess.check_output(cmd)
    assert 'Errno' not in str(result), "Error occurred when trying to use CLI!"


def test_ff():
    cmd = "python {cli} sample ff -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    
    assert len(result) == 0, "Error occurred!"


def test_fast():
    cmd = "python {cli} sample fast_sampler -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    
    assert len(result) == 0, "Error occurred!"


def test_finite_diff():
    cmd = "python {cli} sample finite_diff -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    
    assert len(result) == 0, "Error occurred!"


def test_latin():
    cmd = "python {cli} sample latin -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    
    assert len(result) == 0, "Error occurred!"


# This test identical to the one above...
# def test_saltelli():
#     cmd = "python {cli} sample latin -p {fn} -o {test_data} -n 100".format(
#         cli=salib_cli,
#         fn=ishigami_fp,
#         test_data=test_data).split()
#     result = subprocess.check_output(cmd)
#     assert len(result) == 0, "Error occurred!"


def test_radial_sobol():
    """Only Sobol-based radial sample is offered from the CLI for now."""
    cmd = "python {cli} sample radial -p {fn} -o {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data).split()
    result = subprocess.check_output(cmd)
    
    assert len(result) == 0, "Error occurred!"
    

@pytest.fixture(scope="session", autouse=True)
def cleanup():
    try:
        os.remove(test_data)
    except:
        pass


if __name__ == '__main__':
    test_cli_entry()
    test_ff()
    test_fast()
    test_finite_diff()
    test_latin()
    test_radial_sobol()
