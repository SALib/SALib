import subprocess
import importlib
from SALib.util import avail_approaches


def test_cli_usage():
    cmd = ["salib"]
    try:
        out = subprocess.check_output(cmd,
                                      stderr=subprocess.STDOUT,
                                      shell=True,
                                      universal_newlines=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Call failed:\n{e}")
    else:
        # if no error raised, check the returned string
        assert len(out) > 0 and "usage" in out.lower(), \
            "Incorrect message returned from utility"


def test_cli_avail_methods():
    method_types = ['sample', 'analyze']

    for method in method_types:
        module = importlib.import_module(f'SALib.{method}')
        actions = avail_approaches(module)
        for act in actions:
            approach = importlib.import_module(f'SALib.{method}.{act}')

            # Just try to access the functions - raises error on failure
            approach.cli_parse
            approach.cli_action


if __name__ == '__main__':
    test_cli_usage()
    test_cli_avail_methods()
