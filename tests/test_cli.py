import subprocess
import importlib
from SALib.util import avail_approaches


def test_cli_usage():
    cmd = ["salib"]
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    out = out.decode()
    assert len(out) > 0 and "usage" in out.lower(), \
        "Incorrect message returned from utility"


def test_cli_setup():
    cmd = ["salib", "sample", "morris"]
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    out = out.decode()
    assert len(out) > 0 and "error" not in out.lower(), \
        "Could not use salib as command line utility!"

    cmd = ["salib", "sample", "unknown_method"]
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    out = out.decode()
    assert len(str(out)) > 0 and "invalid choice" in out.lower(), \
        "Unimplemented method selected but no error outputted!"


def test_cli_avail_methods():
    method_types = ['sample', 'analyze']

    for method in method_types:
        module = importlib.import_module('.'.join(['SALib', method]))
        actions = avail_approaches(module)
        for act in actions:
            approach = importlib.import_module('.'.join(
                ['SALib', method, act]))

            # Just try to access the functions - raises error on failure
            approach.cli_parse
            approach.cli_action


if __name__ == '__main__':
    test_cli_usage()
    test_cli_setup()
    test_cli_avail_methods()
