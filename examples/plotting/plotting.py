import sys
sys.path.append('../..')

from SALib import ProblemSpec
from SALib.test_functions import Ishigami

import matplotlib.pyplot as plt
import numpy as np


# By convention, we assign to "sp" (for "SALib Problem")
sp = ProblemSpec({
    'names': ['x1', 'x2', 'x3'],   # Name of each parameter
    'bounds': [[-np.pi, np.pi]]*3,  # bounds of each parameter
    'outputs': ['Y']               # name of outputs in expected order
})

(sp.sample_saltelli(512, calc_second_order=True)
    .evaluate(Ishigami.evaluate)
    .analyze_sobol())

# Display results in table format
print(sp)

# First-order indices expected with Saltelli sampling:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0

# Basic plotting of results
sp.plot()

plt.title("Basic example plot")


# More advanced plotting

# Plot functions actually return matplotlib axes objects
# In the case of the Sobol' method if `calc_second_order=True`, there will be 
# 3 axes (one each for Total, First, and Second order indices)
axes = sp.plot()

# These can be modified as desired.
# Here, for example, we set the Y-axis to log scale
for ax in axes:
    ax.set_yscale('log')

axes[0].set_title("Example custom plot with log scale")

# Other custom layouts can be created in the usual matplotlib style
# with the basic bar plotter.

from SALib.plotting.bar import plot as barplot

# Example: Direct control of plot elements
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 16))

# Get result DataFrames
total, first, second = sp.to_df()

ax1 = barplot(total, ax=ax1)
ax2 = barplot(first, ax=ax2)
ax3 = barplot(second, ax=ax3)

ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.set_title("Customized matplotlib plot")
plt.show()
