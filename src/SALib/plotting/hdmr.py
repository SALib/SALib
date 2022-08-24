"""
Created on Dec 20, 2019

@author: @sahin-abdullah

This submodule produces two diffent figures: (1) emulator vs simulator,
(2) regression lines of first order component functions
"""
import matplotlib.pyplot as plt
import numpy as np


def plot(Si):
    # Close all figures
    plt.close("all")

    # Extract necessary data from Si
    problem = Si.problem
    Em = Si["Em"]
    RT = Si["RT"]
    Y_em = Si["Y_em"]
    idx = Si["idx"]
    X = Si["X"]
    Y = Si["Y"]

    # Get number of bootstrap from Runtime and sample size N
    K = RT.shape[1]
    N = Y.shape[0]
    row = 2
    col = 5

    try:
        ax = Si._plot()
    except AttributeError:
        # basic bar plot not found or is not available
        pass

    # Plot emulator performance
    Y_p = np.linspace(np.min(Y), np.max(Y), N, endpoint=True)

    start = max(0, K - 10)
    for i in range(start, K):
        # Only showing the last 10
        if (i % (row * col) == 0) or (i == 0):
            fig = plt.figure(frameon=False)
            fig.suptitle("Showing last 10 Bootstrap Trials")
            it = 1

        title_str = f"Trial {K - i}"
        ax = fig.add_subplot(row, col, it, frameon=True, title=title_str)
        ax.plot(Em["Y_e"][:, i], Y[idx[:, i]], "r+", label="Emulator")
        ax.plot(Y_p, Y_p, "darkgray", label="1:1 Line")
        ax.axis("square")
        ax.set_xlim(np.min(Y), np.max(Y))
        ax.set_ylim(np.min(Y), np.max(Y))
        it += 1

        if i == (K - 1):
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()

    # Now plot regression lines of component functions
    row = 3
    col = 1
    last_bootstrap = idx[:, -1]
    for i in range(problem["num_vars"]):
        if (i % (row * col) == 0) or (i == 0):
            fig = plt.figure(frameon=False)
            it = 1
        title_str = (
            "Regression of parameter " + problem["names"][i] + r"$^{(Last Trial)}$"
        )
        ax = fig.add_subplot(row, col, it, frameon=True, title=title_str)
        ax.plot(X[last_bootstrap, i], Y[last_bootstrap], "r.")
        ax.plot(X[last_bootstrap, i], np.mean(Em["f0"]) + Y_em[:, i], "k.")
        ax.legend(
            [r"$\widetilde{Y}$", "$f_" + str(i + 1) + "$"],
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
        )
        it += 1

    fig.tight_layout()
    plt.show()

    if "emulated" in Si:
        emulated = Si["emulated"]
        # Sum of squared residuals
        Y_test = Si["Y_test"]
        ssr = np.sum((emulated - Y_test) ** 2)

        # Plot testing results
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.plot(emulated, Y_test, "r+", label="Emulator")
        ax.plot(Y_test, Y_test, "darkgray", label="1:1 Line")
        ax.axis("square")
        ax.set_xlabel("Emulator")
        ax.set_ylabel("New Observation")
        ax.set_xlim(np.min(Y_test), np.max(Y_test))
        ax.set_ylim(np.min(Y_test), np.max(Y_test))
        ax.legend(loc="lower right")
        plt.title(f"Testing results\nSSR = {ssr:.2f}")
        fig.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    pass
