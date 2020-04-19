import sys
import subprocess
from SALib.test_functions import Ishigami
import numpy as np
import pandas as pd
from io import StringIO

import re

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

    expected = "vivi_stddgsmdgsm_confx17.62237816.1981232.2075541.034173x224.48775717.3385567.0920191.090835x311.18125824.0621273.2382591.477114"

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

    expected = """              S1        ST
x1  3.104027e-01  0.555603
x2  4.425532e-01  0.469546
x3  6.340153e-34  0.239155"""

    data = StringIO(expected)
    df1 = pd.read_csv(data, sep="\t")

    df1 = df1.iloc[:, 0].str.split(expand=True)
    df1.columns = ["Name", "S1", "ST"]
    df1 = df1.iloc[:, 1:].values.astype('float64')

    data = StringIO(result)
    df2 = pd.read_csv(data, sep="\t")

    df2 = df2.iloc[:, 0].str.split(expand=True)
    df2.columns = ["Name", "S1", "ST"]
    df2 = df2.iloc[:, 1:].values.astype('float64')

    assert np.allclose(df1, df2), \
        "Unexpected FAST results.\n\nExpected:\n{}\n\nGot:{}"\
        .format(expected, result)
    
    # expected = "S1STx13.104027e-010.555603x24.425532e-010.469546x36.340153e-340.239155"
    # expected = "S1STx13.104027e-010.555603x24.425532e-010.469546x35.105568e-340.239155"

    # assert len(result) > 0 and result == expected, \
    #     "Unexpected FAST results.\n\nExpected:\n{}\n\nGot:{}"\
    #     .format(expected, result)


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

    expected = "MEx13.855764e-08x20.000000e+00x30.000000e+00dummy_00.000000e+00IE(x1,x2)0.0(x1,x3)0.0(x2,x3)0.0(x1,dummy_0)0.0(x2,dummy_0)0.0(x3,dummy_0)0.0"

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

    expected_output = "mumu_starsigmamu_star_confx17.4989307.4989309.3304601.801208x2-0.4703942.2152432.7759250.347972x30.8640155.4238337.8621281.147559"

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

    expected = "S1x10.392230x20.299578x30.034231"

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

    expected_output = 'STST_confx10.5601370.091908x20.4387220.040634x30.2428450.026578S1S1_confx10.3079750.063047x20.4477670.053323x3-0.0042550.059667S2S2_conf(x1,x2)0.0122050.086177(x1,x3)0.2515260.108147(x2,x3)-0.0099540.065569'
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
