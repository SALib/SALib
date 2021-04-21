import sys
import subprocess
from SALib.test_functions import Ishigami
import numpy as np
import re

salib_cli = "./src/SALib/scripts/salib.py"
ishigami_fp = "./src/SALib/test_functions/params/Ishigami.txt"

if sys.version_info[0] == 2:
    subprocess.run = subprocess.call


def test_delta():
    cmd = "python {cli} sample saltelli -p {fn} -o model_input.txt -n 1024"\
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

    expected_output = 'Parameterdeltadelta_confS1S1_confx10.2122850.0074810.3123190.011463x20.3530150.0061840.4306860.013135x30.1613440.0057540.0013880.001545'
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

    expected = "Parametervivi_stddgsmdgsm_confx17.69803416.3731482.2331100.986061x224.48770117.3199737.1035971.092944x311.05754523.7851003.2076651.488346"

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
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

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
    cmd = "python {cli} sample saltelli -p {fn} -o model_input.txt -n 1024\
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

    expected_output = 'ParameterS1S1_confSTST_confx10.3168320.0622410.5558600.085972x20.4437630.0560470.4418980.041596x30.0122030.0559540.2446750.025332Parameter_1Parameter_2S2S2_confx1x20.0092540.083829x1x30.2381720.101764x2x3-0.0048880.067819'
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
