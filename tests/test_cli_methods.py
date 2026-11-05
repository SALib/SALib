"""Exercise discovered method CLIs with real subprocesses and valid model data."""

import importlib
import subprocess

import numpy as np
import pytest

from SALib import analyze, sample
from SALib.test_functions import Ishigami
from SALib.util import avail_approaches, read_param_file


@pytest.mark.parametrize("action", ["sample", "analyze"])
def test_method_cli_help(action):
    package = sample if action == "sample" else analyze
    for method in avail_approaches(package):
        result = subprocess.run(
            ["salib", action, method, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (method, result.stderr)
        assert "usage:" in result.stdout.lower(), method


@pytest.mark.parametrize("method", avail_approaches(analyze))
def test_analyze_cli(method, tmp_path):
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    samplers = {
        "fast": "fast_sampler",
        "dgsm": "finite_diff",
        "ff": "ff",
        "morris": "morris",
        "sobol": "sobol",
        "shapley": "shapley",
        "radial_ee": "radial.radial_sobol",
        "sobol_jansen": "radial.radial_sobol",
    }
    if method in samplers:
        sampler = importlib.import_module(f"SALib.sample.{samplers[method]}")
        kwargs = {} if method == "ff" else {"N": 512}
        X = sampler.sample(problem, seed=123, **kwargs)
    else:
        X = np.random.default_rng(123).uniform(-np.pi, np.pi, (512, 3))
    input_file = tmp_path / "X.txt"
    output_file = tmp_path / "Y.txt"
    np.savetxt(input_file, X)
    np.savetxt(output_file, Ishigami.evaluate(X))
    options = {
        "radial_ee": ["-n", "512"],
        "sobol_jansen": ["-n", "512"],
        # Keep the HDMR smoke tests small; numerical accuracy has separate tests.
        "hdmr": ["-K", "1", "-mit", "1"],
        "enhanced_hdmr": [
            "-mor",
            "1",
            "-por",
            "2",
            "-K",
            "1",
            "-mit",
            "1",
            "-ext",
            "true",
        ],
    }
    cmd = [
        "salib",
        "analyze",
        method,
        "-p",
        param_file,
        "-Y",
        str(output_file),
        "--seed",
        "123",
    ]
    if method not in {"fast", "sobol", "sobol_jansen"}:
        cmd += ["-X", str(input_file)]
    result = subprocess.run(
        cmd + options.get(method, []),
        capture_output=True,
        text=True,
        timeout=120,
    )
    known_failures = {
        "pawn": "TypeError: 'NoneType' object cannot be interpreted as an integer",
        "rsa": "TypeError: unsupported operand type(s) for /: 'int' and 'NoneType'",
        "radial_ee": "AttributeError: 'Namespace' object has no attribute 'num_resamples'",
    }
    if method in known_failures and result.returncode != 0:
        assert known_failures[method] in result.stderr
        pytest.xfail("Known CLI crash reported in #712")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("method", avail_approaches(sample))
def test_sample_cli(method, tmp_path):
    output_file = tmp_path / "X.txt"
    cmd = [
        "salib",
        "sample",
        method,
        "-p",
        "src/SALib/test_functions/params/Ishigami.txt",
        "-o",
        str(output_file),
        "-n",
        "512",
    ]
    if method != "saltelli":
        cmd += ["--seed", "123"]
    if method == "morris":
        cmd += ["--local", "False"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    X = np.loadtxt(output_file, ndmin=2)
    assert X.shape[0] > 0
    assert X.shape[1] >= 3
    assert np.isfinite(X).all()
