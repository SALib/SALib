"""The analyze CLI checks X and Y the way `ProblemSpec` does (#70)."""

import sys

import numpy as np
import pytest

from SALib.scripts.salib import main

# Methods that read a model input file, with any other required options.
X_METHODS = {
    "delta": [],
    "dgsm": [],
    "discrepancy": [],
    "enhanced_hdmr": ["-mor", "1", "-por", "2", "-ext", "false"],
    "ff": [],
    "hdmr": [],
    "morris": [],
    "pawn": [],
    "radial_ee": ["-n", "32"],
    "rbd_fast": [],
    "rsa": [],
    "shapley": [],
}


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(1)
    n = 128
    paths = {
        "p": tmp_path / "params.txt",
        "X_cols": tmp_path / "X_cols.txt",
        "X_rows": tmp_path / "X_rows.txt",
        "Y": tmp_path / "Y.txt",
    }
    paths["p"].write_text("x1 -3.14 3.14\nx2 -3.14 3.14\nx3 -3.14 3.14\n")
    np.savetxt(paths["X_cols"], rng.random((n, 2)))
    np.savetxt(paths["X_rows"], rng.random((n - 8, 3)))
    np.savetxt(paths["Y"], rng.random(n))
    return paths


def run(monkeypatch, method, X, files):
    argv = ["salib", "analyze", method, "-p", str(files["p"]), "-X", str(X)]
    argv += ["-Y", str(files["Y"]), *X_METHODS[method]]
    monkeypatch.setattr(sys, "argv", argv)
    main()


@pytest.mark.parametrize("method", sorted(set(X_METHODS) - {"ff"}))
def test_cli_rejects_wrong_number_of_columns(monkeypatch, files, method):
    with pytest.raises(ValueError, match="Expected 3 cols"):
        run(monkeypatch, method, files["X_cols"], files)


@pytest.mark.parametrize("method", sorted(X_METHODS))
def test_cli_rejects_mismatched_rows(monkeypatch, files, method):
    with pytest.raises(ValueError, match="Mismatched number of rows"):
        run(monkeypatch, method, files["X_rows"], files)
