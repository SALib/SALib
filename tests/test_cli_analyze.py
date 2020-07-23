"""Test CLI usage.

These tests only ensure input/outputs from the CLI.
Results are compared against previously obtained results.
"""

import sys
import subprocess
from SALib.test_functions import Ishigami
import numpy as np
import re
import os
from os.path import join as pth_join

salib_cli = "./src/SALib/scripts/salib.py"
ishigami_fp = "./src/SALib/test_functions/params/Ishigami.txt"
test_dir = 'tests/data'
input_file = 'model_input.txt'
output_file = 'model_output.txt'
input_path = pth_join(test_dir, input_file) 
output_path =  pth_join(test_dir, output_file)

if sys.version_info[0] == 2:
    subprocess.run = subprocess.call


def teardown_function(func):
    # Removes the test file if it was created.
    files = os.listdir(test_dir)
    if input_file in files:
        os.remove(input_path)

    if output_file in files:
        os.remove(output_path)


def test_delta():
    cmd = f"python {salib_cli} sample saltelli -p {ishigami_fp} -o {input_path} -n 1000 \
    --precision 8 --max-order 2 --seed=100"
    subprocess.run(cmd.split())

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze delta -p {ishigami_fp} -X {input_path} \
    -Y {output_path} -c 0 -r 10 --seed=100".split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = 'Parameterdeltadelta_confS1S1_confx10.2104780.0060910.3113620.012291x20.3540230.0062380.4283650.017972x30.1609860.0047180.0011110.002995'
    assert len(result) > 0 and result in expected_output, \
        f"Results did not match expected values:\n\n Expected: \n{expected_output} \n\n Got: \n{result}"


def test_dgsm():
    # Generate inputs
    cmd = f"python {salib_cli} sample finite_diff -p {ishigami_fp} -o {input_path} -d 0.001\
    --precision=8 -n 1000 --seed=100".split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze dgsm -p {ishigami_fp} -X {input_path}\
    -Y {output_path} -c 0 -r 1000 --seed=100".split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "Parametervivi_stddgsmdgsm_confx17.62237816.1981232.2075541.034173x224.48775717.3385567.0920191.090835x311.18125824.0621273.2382591.477114"

    assert len(result) > 0 and result == expected, \
        f"Unexpected DGSM results.\n\nExpected:\n{expected}\n\nGot:{result}"


def test_fast():
    # Generate inputs
    cmd = f"python {salib_cli} sample fast_sampler -p {ishigami_fp} -o {input_path} \
    --precision=8 -n 1000 -M 4 --seed=100".split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze fast -p {ishigami_fp} \
    -Y {output_path} -c 0 --seed=100".split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterFirstTotalx10.3104030.555603x20.4425530.469546x30.0000000.239155"

    assert len(result) > 0 and result == expected, \
        f"Unexpected FAST results.\n\nExpected:\n{expected}\n\nGot:{result}"


def test_ff():
    # Generate inputs
    cmd = f"python {salib_cli} sample ff -p {ishigami_fp} -o {input_path} \
    --precision=8 -n 1000 --seed=100".split()
    subprocess.run(cmd)

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze ff -p {ishigami_fp} -X {input_path} \
    -Y {output_path} -c 0 --seed=100".split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterMEx10.000000x20.000000x30.000000dummy_00.000000('x1','x2')0.000000('x1','x3')0.000000('x2','x3')0.000000('x1','dummy_0')0.000000('x2','dummy_0')0.000000('x3','dummy_0')0.000000"

    assert len(result) > 0 and result == expected, \
        f"Unexpected FF results.\n\nExpected:\n{expected}\n\nGot:{result}"


def test_morris():

    # Generate inputs
    cmd = f"python {salib_cli} sample morris -p {ishigami_fp} -o {input_path} -n 100\
    --precision=8 --levels=10 --seed=100 -lo False".split()

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(np.loadtxt(input_path)))

    # run analysis
    analyze_cmd = f"python {salib_cli} analyze morris -p {ishigami_fp} -X {input_path}\
    -Y {output_path} -c 0 -r 1000 -l 10 --seed=100".split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = """ParameterMu_StarMuMu_Star_ConfSigmax17.4997.4991.8019.330x22.215-0.4700.3482.776x35.4240.8641.1487.862"""

    assert len(result) > 0 and result == expected_output, \
        "Results did not match expected values:\n\n Expected: \n{} \n\n Got: \n{}".format(
            expected_output, result)


def test_rbd_fast():
    # Generate inputs
    cmd = f"python {salib_cli} sample latin -p {ishigami_fp} -o {input_path} \
    --precision=8 -n 1000 --seed=100".split()

    subprocess.run(cmd)

    # Run model and save output
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze rbd_fast -p {ishigami_fp} -X {input_path}\
    -Y {output_path} --seed=100".split()

    # run analysis and use regex to strip all whitespace from result
    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected = "ParameterFirstx10.298085x20.46924x3-0.0105166"

    assert len(result) > 0 and result == expected, \
        f"Unexpected RBD-FAST results.\n\nExpected:\n{expected}\n\nGot:{result}"


def test_sobol():
    # Generate inputs
    cmd = f"python {salib_cli} sample saltelli -p {ishigami_fp} -o {input_path} -n 1000\
    --precision 8 --max-order 2 --seed=100"
    cmd = cmd.split()

    result = subprocess.check_output(cmd, universal_newlines=True)
    np.savetxt(output_path, Ishigami.evaluate(
        np.loadtxt(input_path)))

    analyze_cmd = f"python {salib_cli} analyze sobol -p {ishigami_fp}\
    -Y {output_path} -c 0 --max-order 2\
    -r 1000 --seed=100".split()

    result = subprocess.check_output(analyze_cmd, universal_newlines=True)
    result = re.sub(r'[\n\t\s]*', '', result)

    expected_output = 'ParameterS1S1_confSTST_confx10.3079750.0630470.5601370.091908x20.4477670.0533230.4387220.040634x3-0.0042550.0596670.2428450.026578Parameter_1Parameter_2S2S2_confx1x20.0122050.086177x1x30.2515260.108147x2x3-0.0099540.065569'
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
