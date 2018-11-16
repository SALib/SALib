from os.path import join as pth_join
import subprocess

salib_cli = "./SAlib/scripts/salib.py"
ishigami_fp = "./SALib/test_functions/params/Ishigami.txt"
test_data = pth_join('tests', 'data', 'test.txt')


def test_cli_entry():
    result = subprocess.check_output('python {cli} -h'.format(cli=salib_cli))
    assert 'Errno' not in str(result), "Error occurred when trying to use CLI!"


def test_ff():
    cmd = "python {cli} sample ff -p {fn} -r {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data)
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_fast():
    cmd = "python {cli} sample fast_sampler -p {fn} -r {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data)
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_finite_diff():
    cmd = "python {cli} sample finite_diff -p {fn} -r {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data)
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_latin():
    cmd = "python {cli} sample latin -p {fn} -r {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data)
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


def test_saltelli():
    cmd = "python {cli} sample latin -p {fn} -r {test_data} -n 100".format(
        cli=salib_cli,
        fn=ishigami_fp,
        test_data=test_data)
    result = subprocess.check_output(cmd)
    assert len(result) == 0, "Error occurred!"


if __name__ == '__main__':
    test_cli_entry()
    test_ff()
    test_fast()
    test_finite_diff()
    test_latin()
    test_saltelli()
