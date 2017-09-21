'''Finds optimal trajectories using a global optimisation method

Example
-------
Run using

>>> optimal_trajectories.py -n=10 \
    -p=esme_param.txt -o=test_op.txt \
    -s=12892 --num-levels=4 --grid-jump=2 \
    --k-choices=4

'''
from __future__ import division

import re
from datetime import datetime as dt

import numpy as np
from scipy.misc import comb as nchoosek

from . strategy import Strategy

from SALib.util import requires_gurobipy

try:
    from gurobipy import Model, quicksum, GRB
except ImportError:
    _has_gurobi = False
else:
    _has_gurobi = True


@requires_gurobipy(_has_gurobi)
class GlobalOptimisation(Strategy):
    """Implements the global optimisation algorithm
    """

    def _sample(self, input_sample, num_samples,
                num_params, k_choices, num_groups=None):
        return self.return_max_combo(input_sample, num_samples,
                                     num_params, k_choices, num_groups)

    @staticmethod
    def global_model(N, k_choices, distance_matrix):

        if k_choices >= N:
            raise ValueError("k_choices must be less than N")

        model = Model("distance1")
        I = range(N)

        distance_matrix = np.array(
            distance_matrix / distance_matrix.max(), dtype=np.float64)

        dm = distance_matrix ** 2

        y, x = {}, {}
        for i in I:
            y[i] = model.addVar(vtype="B", obj=0, name="y[%s]" % i)
            for j in range(i + 1, N):
                x[i, j] = model.addVar(
                    vtype="B", obj=1.0, name="x[%s,%s]" % (i, j))
        model.update()

        model.setObjective(quicksum([x[i, j] * dm[j][i]
                                     for i in I for j in range(i + 1, N)]))

        # Add constraints to the model
        model.addConstr(quicksum([y[i] for i in I]) <= k_choices, "27")

        for i in I:
            for j in range(i + 1, N):
                model.addConstr(x[i, j] <= y[i], "28-%s-%s" % (i, j))
                model.addConstr(x[i, j] <= y[j], "29-%s-%s" % (i, j))
                model.addConstr(y[i] + y[j] <= 1 + x[i, j],
                                "30-%s-%s" % (i, j))

        model.addConstr(quicksum([x[i, j] for i in I
                                  for j in range(i + 1, N)])
                        <= nchoosek(k_choices, 2), "Cut_1")
        model.update()
        return model

    def return_max_combo(self, input_data, N, num_params,
                         k_choices, num_groups=None):
        """Find the optimal combination of most different trajectories

        Arguments
        ---------
        input_data : numpy.ndarray
        N : int
            The number of samples
        num_params : int
            The number of factors
        k_choices : int
            The number of optimal trajectories to select
        num_groups: int, default=None
            The number of groups

        Returns
        -------
        list
        """

        distance_matrix = self.compute_distance_matrix(
            input_data, N, num_params, num_groups)

        model = self.global_model(N, k_choices, distance_matrix)
        model.params.Threads = 0

        model.ModelSense = GRB.MAXIMIZE
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            print(model.objval)

        variables = list(model.getVars())
        x_vars = []
        for v in variables:
            if (v.X > 0) & (v.VarName[0] == 'x'):
                x_vars.append(v.VarName)
        b = [re.findall(r"\d{1,}", str(v)) for v in x_vars]
        maximum_combo = list(set([int(y) for z in b for y in z]))

        print(maximum_combo)

        return sorted(maximum_combo)


def timestamp(num_params, p_levels, grid_step, k_choices, N):
    """
    Returns a uniform timestamp with parameter values for file identification
    """
    string = "_v%s_l%s_gs%s_k%s_N%s_%s.txt" % (num_params,
                                               p_levels,
                                               grid_step,
                                               k_choices,
                                               N,
                                               dt.strftime(dt.now(),
                                                           "%d%m%y%H%M%S"))
    return string
