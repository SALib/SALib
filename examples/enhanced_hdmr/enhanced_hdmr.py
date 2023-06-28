import sys
sys.path.append("../../src")
import numpy as np
from SALib.analyze import enhanced_hdmr
from SALib.sample import latin
from SALib.test_functions import Ishigami

# Define SALib problem specification.
problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3}

# Generate samples
X = latin.sample(problem, 1000, seed=101)

# Run the "model" and save the output in a text file
Y = Ishigami.evaluate(X)

# Add random error
Y += np.var(Y) / 100

# SALib-HDMR options
options = {
    "max_order": 2,
    "poly_order": 6,
    "bootstrap": 50,
    "subset": 500,
    "max_iter": 100,
    "l2_penalty": 0.01,
    "alpha": 0.95,
    "extended_base": True,
    "print_to_console": True,
    "return_emulator": True,
    "seed": 101
}
# Run SALib-HDMR
Si = enhanced_hdmr.analyze(problem, X, Y, **options)

# Displays sensitivity results and HDMR training results
Si.plot()

# Generate samples  int(np.random.random() * 100)
X_new = latin.sample(problem, 1000, seed=int(np.random.random() * 100))
# Run the "model"
Y_new = Ishigami.evaluate(X_new)

# Test emulator with new Y values.
# Can call `emulate()` without Y values, which will run the emulator
# Emulation results will be in Si["emulated"]
Y_em = Si.emulate(X_new) + np.mean(Y_new)

print(f"\nSum of squared residual is: {sum((Y_em - Y_new)**2)}")
