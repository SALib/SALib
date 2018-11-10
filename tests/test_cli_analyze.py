import subprocess
from SALib.test_functions import Ishigami
import numpy as np


np.random.seed(100)

salib_cli = "../SAlib/scripts/salib.py"
ishigami_fp = "../SALib/test_functions/params/Ishigami.txt"


def test_sobol():

    # Generate inputs
    cmd = "python {cli} sample saltelli -p {fn} -r model_input.txt -n 1000\
    --delimiter \" \" --precision 8 --max-order 2 --seed 100".format(cli=salib_cli,
                                                                     fn=ishigami_fp)
    result = subprocess.check_output(cmd)
    np.savetxt('model_output.txt', Ishigami.evaluate(
        np.loadtxt('model_input.txt')))

    analyze_cmd = "python {cli} analyze sobol -p {fn} -Y model_output.txt\
     -c 0 --max-order 2 -r 1000".format(cli=salib_cli, fn=ishigami_fp)

    result = subprocess.check_output(analyze_cmd)
    expected_output = b'Parameter S1 S1_conf ST ST_conf\r\nx1 0.307975 0.062560 0.560137 0.084924\r\nx2 0.447767 0.052750 0.438722 0.040011\r\nx3 -0.004255 0.061140 0.242845 0.025259\r\n\r\nParameter_1 Parameter_2 S2 S2_conf\r\nx1 x2 0.012205 0.082933\r\nx1 x3 0.251526 0.106556\r\nx2 x3 -0.009954 0.067336\r\n'
    assert len(result) > 0 and result == expected_output, \
        "Error occurred in Sobol Analysis:\n\n {}".format(
            result)


if __name__ == '__main__':
    test_sobol()
