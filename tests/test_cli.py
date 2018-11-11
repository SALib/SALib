import subprocess


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


if __name__ == '__main__':
    test_cli_setup()
