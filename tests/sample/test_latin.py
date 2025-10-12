from SALib.sample.latin import sample
from pytest import approx
import numpy as np


class TestLatinSample:
    def test_latin_sample_trivial(self):
        problem = {"num_vars": 1, "bounds": [[0, 1]], "names": ["var1"]}

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.77860643],
                [0.0773956 ],
                [0.40941773],
                [0.28585979],
                [0.3697368 ],
                [0.59756224],
                [0.14388784],
                [0.81281136],
                [0.67611397],
                [0.94503859]
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_trivial_group(self):
        problem = {
            "num_vars": 1,
            "bounds": [[0, 1]],
            "names": ["var1"],
            "groups": ["group1"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.77860643],
                [0.0773956 ],
                [0.40941773],
                [0.28585979],
                [0.3697368 ],
                [0.59756224],
                [0.14388784],
                [0.81281136],
                [0.67611397],
                [0.94503859]
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_one_group(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": ["group1", "group1"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.77860643, 0.77860643],
                [0.0773956,  0.0773956 ],
                [0.40941773, 0.40941773],
                [0.28585979, 0.28585979],
                [0.3697368,  0.3697368 ],
                [0.59756224, 0.59756224],
                [0.14388784, 0.14388784],
                [0.81281136, 0.81281136],
                [0.67611397, 0.67611397],
                [0.94503859, 0.94503859]
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_sample_no_groups(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": None,
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.86011150, 0.15247564],
                [0.27319939, 0.84560700],
                [0.03745401, 0.73663618],
                [0.60580836, 0.51394939],
                [0.78661761, 0.46118529],
                [0.97080726, 0.97851760],
                [0.35986585, 0.03042422],
                [0.19507143, 0.32912291],
                [0.41560186, 0.62921446],
                [0.51559945, 0.24319450],
            ]
        )
        approx(actual, expected)

    def test_latin_sample_two_groups(self):
        problem = {
            "num_vars": 2,
            "bounds": [[0, 1], [0, 1]],
            "names": ["var1", "var2"],
            "groups": ["group1", "group2"],
        }

        actual = sample(problem, 10, seed=42)
        expected = np.array(
            [
                [0.20941773, 0.68227616],
                [0.98276312, 0.37860643],
                [0.0773956,  0.80638173],
                [0.85545848, 0.29756224],
                [0.41281136, 0.96316644],
                [0.18585979, 0.5926765 ],
                [0.66438651, 0.1697368 ],
                [0.74434142, 0.72272387],
                [0.37611397, 0.44503859],
                [0.5370798,  0.04388784]
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_latin_group_constant(self):
        """Ensure grouped parameters have identical values."""
        problem = {
            "num_vars": 6,
            "names": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "bounds": [[-100.0, 100.0] * 6],
            "groups": ["A", "B"] * 3,
        }
        samples = sample(problem, 10, seed=42)

        # Group samples should have the same values
        # Get (max - min) with the `ptp()` method, the result of which should be
        # an array of zeros
        diff = np.ptp(samples[:, ::2], axis=1)
        assert np.all(diff == 0), "Grouped samples do not have the same values"

        diff = np.ptp(samples[:, 1::2], axis=1)
        assert np.all(diff == 0), "Grouped samples do not have the same values"
