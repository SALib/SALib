import sys
import subprocess
import os

from SALib.test_functions import Ishigami
import numpy as np
import re
import pytest


salib_cli = "./src/SALib/scripts/salib.py"
ishigami_fp = "./src/SALib/test_functions/params/Ishigami.txt"

if sys.version_info[0] == 2:
    subprocess.run = subprocess.call


def test_delta():
    cmd = "python {cli} sample saltelli -p {fn} -o model_input.txt -n 1000"\
          .format(cli=salib_cli, fn=ishigami_fp) +\
          " --precision 8 --max-order 2 --seed=100"
    subprocess.run(cmd.split())

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze delta -p {fn} -X model_input.txt \
    -Y model_output.txt -c 0 -r 10 --seed=100".format(cli=salib_cli,
                                                      fn=ishigami_fp).split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = 'Parameterdeltadelta_confS1S1_confx10.2104780.0060910.3113620.012291x20.3540230.0062380.4283650.017972x30.1609860.0047180.0011110.002995'
    assert len(result) > 0 and result in expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


def test_dgsm():
    # Generate inputs
    cmd = "python {cli} sample finite_diff -p {fn} -o model_input.txt -d 0.001\
    --precision=8 -n 1000 --seed=100".format(cli=salib_cli,
                                             fn=ishigami_fp).split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze dgsm -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 -r 1000 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp).split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "Parametervivi_stddgsmdgsm_confx17.62237816.1981232.2075541.034173x224.48775717.3385567.0920191.090835x311.18125824.0621273.2382591.477114"

    assert len(result) > 0 and result == expected, \
        "Unexpected DGSM results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_fast():
    # Generate inputs
    cmd = "python {cli} sample fast_sampler -p {fn} -o model_input.txt \
    --precision=8 -n 1000 -M 4 --seed=100".format(cli=salib_cli,
                                                  fn=ishigami_fp).split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze fast -p {fn} \
    -Y model_output.txt -c 0 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp).split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterFirstTotalx10.3104030.555603x20.4425530.469546x30.0000000.239155"

    assert len(result) > 0 and result == expected, \
        "Unexpected FAST results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_ff():
    # Generate inputs
    cmd = "python {cli} sample ff -p {fn} -o model_input.txt \
    --precision=8 -n 1000 --seed=100".format(cli=salib_cli,
                                             fn=ishigami_fp).split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze ff -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp).split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterMEx10.000000x20.000000x30.000000dummy_00.000000('x1','x2')0.000000('x1','x3')0.000000('x2','x3')0.000000('x1','dummy_0')0.000000('x2','dummy_0')0.000000('x3','dummy_0')0.000000"

    assert len(result) > 0 and result == expected, \
        "Unexpected FF results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_morris():
    # Generate inputs
    cmd = "python {cli} sample morris -p {fn} -o model_input.txt -n 100\
    --precision=8 --levels=10 --seed=100 -lo False"\
    .format(cli=salib_cli, fn=ishigami_fp).split()

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    # run analysis
    analyze_cmd = "python {cli} analyze morris -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 -r 1000 -l 10 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp).split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = """ParameterMu_StarMuMu_Star_ConfSigmax17.4997.4991.8019.330x22.215-0.4700.3482.776x35.4240.8641.1487.862"""

    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


def test_rbd_fast():
    # Generate inputs
    cmd = "python {cli} sample ff -p {fn} -o model_input.txt \
    --precision=8 --seed=100".format(cli=salib_cli, fn=ishigami_fp).split()

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', 
               Ishigami.evaluate(np.loadtxt('model_input.txt'))
    )

    analyze_cmd = "python {cli} analyze rbd_fast -p {fn} -X model_input.txt\
    -Y model_output.txt --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp).split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterFirstx10.39223x20.299578x30.0342307"

    assert len(result) > 0 and result == expected, \
        "Unexpected RBD-FAST results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_sobol():
    # Generate inputs
    cmd = "python {cli} sample saltelli -p {fn} -o model_input.txt -n 1000\
    --precision 8 --max-order 2 --seed=100".format(cli=salib_cli,
                                                   fn=ishigami_fp)
    cmd = cmd.split()

    result = subprocess.check_output(cmd, universal_newlines=True)
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze sobol -p {fn}\
    -Y model_output.txt -c 0 --max-order 2\
    -r 1000 --seed=100".format(cli=salib_cli, fn=ishigami_fp).split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = 'ParameterS1S1_confSTST_confx10.3079750.0630470.5601370.091908x20.4477670.0533230.4387220.040634x3-0.0042550.0596670.2428450.026578Parameter_1Parameter_2S2S2_confx1x20.0122050.086177x1x30.2515260.108147x2x3-0.0099540.065569'
    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    try:
        os.remove("model_input.txt")
        os.remove("model_output.txt")
    except:
        pass


if __name__ == '__main__':
    test_delta()
    test_dgsm()
    test_fast()
    test_ff()
    test_morris()
    test_rbd_fast()
    test_sobol()
