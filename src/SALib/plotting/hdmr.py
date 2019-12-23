'''
Created on Dec 20, 2019

@author: @sahin-abdullah

This submodule produces two diffent figures: (1) emulator vs simulator,
(2) regression lines of first order component functions
'''
import matplotlib.pyplot as plt
import numpy as np


def figures(problem, Si, Em, RT, X, Y, Y_em, idx):
    # Close all figures
    plt.close('all')
    # Get number of bootstrap from Runtime and sample size N
    K = RT.shape[1]
    N = Y.shape[0]
    row = 2
    col = 5
    # Plot emulator performance
    Y_p = np.linspace(np.min(Y), np.max(Y), N, endpoint=True)
    for i in range(K):
        if (i % (row * col) == 0) or (i == 0):
            fig = plt.figure(frameon=False)
            it = 1
        title_str = 'Bootstrap Trial of ' + str(i + 1)
        ax = fig.add_subplot(row, col, it, frameon=True, title=title_str)
        ax.plot(Em['Y_e'][:, i], Y[idx[:, i]], 'r+')
        ax.plot(Y_p, Y_p, 'darkgray')
        ax.axis('tight')
        ax.axis('square')
        ax.set_xlim(np.min(Y), np.max(Y))
        ax.set_ylim(np.min(Y), np.max(Y))
        ax.legend(['Emulator', '1:1 Line'], loc='upper left')
        it += 1
    # Now plot regression lines of component functions
    row = 3
    col = 1
    for i in range(problem['num_vars']):
        if (i % (row * col) == 0) or (i == 0):
            fig = plt.figure(frameon=False)
            it = 1
        title_str = 'Regression of parameter ' + \
            problem['names'][i] + r'$^{(Last Trial)}$'
        ax = fig.add_subplot(row, col, it, frameon=True, title=title_str)
        ax.plot(X[idx[:, -1], i], Y[idx], 'r.')
        ax.plot(X[idx[:, -1], i], np.mean(Em['f0']) + Y_em[:, i], 'k.')
        ax.legend([r'$\widetilde{Y}$', '$f_' +
                   str(i + 1) + '$'], loc='upper left')
        it += 1
    plt.show()

    return


if __name__ == '__main__':
    pass
