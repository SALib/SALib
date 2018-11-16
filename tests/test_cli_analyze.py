import sys
import subprocess
from SALib.test_functions import Ishigami
import numpy as np
import re

salib_cli = "./SALib/scripts/salib.py"
ishigami_fp = "./SALib/test_functions/params/Ishigami.txt"

if sys.version_info[0] == 2:
    subprocess.run = subprocess.call


def test_delta():
    cmd = "python {cli} sample saltelli -p {fn} -r model_input.txt -n 1000\
    --delimiter \" \" --precision 8 --max-order 2 --seed 100".format(cli=salib_cli,
                                                                     fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze delta -p {fn} -X model_input.txt \
    -Y model_output.txt -c 0 -r 10 --seed {seed}".format(
        cli=salib_cli, fn=ishigami_fp, seed=100)

    result = subprocess.check_output(analyze_cmd)
    expected_output = b'Parameter delta delta_conf S1 S1_conf\r\nx1 0.210478 0.006091 0.311362 0.012291\r\nx2 0.354023 0.006238 0.428365 0.017972\r\nx3 0.160986 0.004718 0.001111 0.002995\r\n'
    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


def test_dgsm():
    # Generate inputs
    cmd = "python {cli} sample finite_diff -p {fn} -r model_input.txt -d 0.001\
    --delimiter \" \" --precision=8 -n 1000 --seed=100"\
    .format(cli=salib_cli, fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze dgsm -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 -r 1000 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp)

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd)
    result = re.sub(r'[\s+]', '', result.decode())

    expected = "Parametervivi_stddgsmdgsm_confx17.62237816.1981232.2075541.034173x224.48775717.3385567.0920191.090835x311.18125824.0621273.2382591.477114"

    assert len(result) > 0 and result == expected, \
        "Unexpected DGSM results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_fast():
    # Generate inputs
    cmd = "python {cli} sample fast_sampler -p {fn} -r model_input.txt \
    --delimiter \" \" --precision=8 -n 1000 -M 4 --seed=100"\
    .format(cli=salib_cli, fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze fast -p {fn} \
    -Y model_output.txt -c 0 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp)

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd)
    result = re.sub(r'[\s+]', '', result.decode())

    expected = "ParameterFirstTotalx10.3104030.555603x20.4425530.469546x30.0000000.239155"

    assert len(result) > 0 and result == expected, \
        "Unexpected FAST results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_ff():
    # Generate inputs
    cmd = "python {cli} sample ff -p {fn} -r model_input.txt \
    --delimiter \" \" --precision=8 -n 1000 --seed=100"\
    .format(cli=salib_cli, fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze ff -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp)

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd)
    result = re.sub(r'[\s+]', '', result.decode())

    expected = "ParameterMEx10.000000x20.000000x30.000000dummy_00.000000('x1','x2')0.000000('x1','x3')0.000000('x2','x3')0.000000('x1','dummy_0')0.000000('x2','dummy_0')0.000000('x3','dummy_0')0.000000"

    assert len(result) > 0 and result == expected, \
        "Unexpected FF results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_morris():

    # Generate inputs
    cmd = "python {cli} sample morris -p {fn} -r model_input.txt -n 100\
    --delimiter \" \" --precision=8 --levels=10 --seed=100 -o False"\
    .format(cli=salib_cli, fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    # run analysis
    analyze_cmd = "python {cli} analyze morris -p {fn} -X model_input.txt\
    -Y model_output.txt -c 0 -r 1000 -l 10 --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp)

    result = subprocess.check_output(analyze_cmd)

    expected_output = """ParameterMu_StarMuMu_Star_ConfSigmax13.3753.3750.5903.003x21.4740.1180.0001.477x32.6980.4200.5954.020"""

    # using regex to strip all whitespace
    result = re.sub(r'[\s+]', '', result.decode())

    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


def test_rbd_fast():
    pass
    # Generate inputs
    cmd = "python {cli} sample ff -p {fn} -r model_input.txt \
    --delimiter \" \" --precision=8 --seed=100"\
    .format(cli=salib_cli, fn=ishigami_fp)

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze rbd_fast -p {fn} -X model_input.txt\
    -Y model_output.txt --seed=100"\
     .format(cli=salib_cli, fn=ishigami_fp)

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd)
    result = re.sub(r'[\s+]', '', result.decode())

    expected = "ParameterFirstx10.437313x20.129825x30.000573789"

    assert len(result) > 0 and result == expected, \
        "Unexpected RBD-FAST results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)


def test_sobol():
    # Generate inputs
    cmd = "python {cli} sample saltelli -p {fn} -r model_input.txt -n 1000\
    --delimiter \" \" --precision 8 --max-order 2 --seed 100".format(cli=salib_cli,
                                                                     fn=ishigami_fp)
    result = subprocess.check_output(cmd)
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze sobol -p {fn} -Y model_output.txt\
     -c 0 --max-order 2 -r 1000 --seed=100".format(cli=salib_cli, fn=ishigami_fp)

    result = subprocess.check_output(analyze_cmd)
    expected_output = b'Parameter S1 S1_conf ST ST_conf\r\nx1 0.307975 0.063047 0.560137 0.091908\r\nx2 0.447767 0.053323 0.438722 0.040634\r\nx3 -0.004255 0.059667 0.242845 0.026578\r\n\r\nParameter_1 Parameter_2 S2 S2_conf\r\nx1 x2 0.012205 0.086177\r\nx1 x3 0.251526 0.108147\r\nx2 x3 -0.009954 0.065569\r\n'
    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


if __name__ == '__main__':
    test_delta()
    test_dgsm()
    test_fast()
    test_ff()
    test_morris()
    test_rbd_fast()
    test_sobol()
