import subprocess

import numpy as np

from SALib.test_functions import Ishigami


def test_shapley_cli_round_trip(tmp_path):
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    input_file = tmp_path / "model_input.txt"
    output_file = tmp_path / "model_output.txt"

    subprocess.run(
        [
            "salib",
            "sample",
            "shapley",
            "-n",
            "256",
            "-p",
            param_file,
            "-o",
            str(input_file),
            "--seed",
            "123",
        ],
        check=True,
    )

    inputs = np.loadtxt(input_file)
    assert inputs.shape == (1_024, 3)
    np.savetxt(output_file, Ishigami.evaluate(inputs))

    result = subprocess.run(
        [
            "salib",
            "analyze",
            "shapley",
            "-p",
            param_file,
            "-X",
            str(input_file),
            "-Y",
            str(output_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Shapley" in result.stdout
    assert "Shapley_conf" in result.stdout
