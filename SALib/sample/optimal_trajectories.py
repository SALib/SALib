from __future__ import division
from gurobipy import *
import morris_optimal, morris_oat
from morris_optimal import compile_output
import common_args
import numpy as np
import random as rd
from ..util import read_param_file
from scipy.misc import comb as nchoosek
import re
from datetime import datetime as dt

'''
Run using
optimal_trajectories.py -n=10 -p=esme_param.txt -o=test_op.txt -s=12892 --num-levels=4 --grid-jump=2 --k-choices=4

'''

def model(N, k_choices, distance_matrix):
    
    if k_choices >= N:
        raise ValueError("k_choices must be less than N")
    
    m = Model("distance1")
    I = range(N)
    big_M = k_choices + 1

    dm=distance_matrix**2

    y,x = {},{}
    for i in I:
        y[i] = m.addVar(vtype="B", obj=0, name="y[%s]" % i)
        for j in range(i+1, N):
            x[i,j] = m.addVar(vtype="B", obj=1.0, name="x[%s,%s]" % (i, j))
    m.update()

    m.setObjective(quicksum([x[i, j] * dm[j][i] for i in I for j in range(i+1,N)]))

    m.addConstr(quicksum([x[i, j] for i in I for j in range(i+1,N)]) == nchoosek(k_choices, 2), "All")

    # Finally, each combination may only appear three times in the combination list
    for i in I:
        m.addConstr(quicksum(x[i, j] for j in range(i+1,N)) + quicksum(x[k, i] for k in range(0,i)) - (y[i] * big_M),
                    '<=',
                    (k_choices - 1),
                    "a:Only k-1 scores in any row/column for %s" % i)
        m.addConstr(quicksum(x[i, j] for j in range(i+1,N)) + quicksum(x[k, i] for k in range(0,i)) + (y[i] * big_M),
                    '>=',
                    (k_choices - 1),
                    "b:Only k-1 scores in any row/column for %s" % i)

    m.addConstr( quicksum( y[i] for i in I ), "==", N - k_choices, name="Only %s hold" % (N-k_choices))
    m.update()
    return m


def return_max_combo(input_data, N, param_file, p_levels, grid_step, k_choices, param_delim=" "):

    pf = read_param_file(param_file)
    num_params = pf['num_vars']

    distance_matrix = morris_optimal.compute_distance_matrix(input_data,
                                                             N,
                                                             num_params)

    m = model(N, k_choices, distance_matrix)
    #m.params.MIPFocus=1 # Focus on feasibility over optimality
    m.params.IntFeasTol=min(0.1,1./(k_choices+1))

    #m.write("model.lp")
    m.ModelSense = GRB.MAXIMIZE
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.objval)

    variables = list(m.getVars())
    x_vars = []
    for v in variables:
        if (v.X > 0 ) & (v.VarName[0] == 'x'):
                x_vars.append(v.VarName)
    b = [re.findall("\d{1,}",str(v)) for v in x_vars]
    maximum_combo = set([int(y) for z in b for y in z])
    return tuple(maximum_combo)


def optimised_trajectories(input_sample,
                           N,
                           param_file,
                           p_levels,
                           grid_step,
                           k_choices,
                           param_delim=" "):
    """
    """
    pf = read_param_file(param_file)
    num_params = pf['num_vars']

    maximum_combo = return_max_combo(input_sample,
                                     N,
                                     param_file,
                                     p_levels,
                                     grid_step,
                                     k_choices,
                                     param_delim)

    output = compile_output(input_sample,
                            N,
                            num_params,
                            k_choices,
                            maximum_combo)
    return output


def timestamp():
    return dt.strftime(dt.now(),"%d%m%y%H%M%S")


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('-g','--grid-jump', type=int, required=False, default=2, help='Grid jump size (Morris only)')
    parser.add_argument('-k', '--k-choices', type=int, required=False, default=4, help='Number of desired optimised trajectories')
    parser.add_argument('-pd','--p-delim', type=str, required=False, default=" ", help='Delimeter for parameter file')

    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    param_file = args.paramfile
    param_delim = args.p_delim
    pf = read_param_file(param_file,param_delim)
    N = args.samples
    num_params = pf['num_vars']
    bounds = pf['bounds']
    k_choices = args.k_choices
    p_levels = int(args.num_levels)
    grid_step = int(args.grid_jump)

    input_sample = morris_oat.sample(N,
                                     param_file,
                                     num_levels=p_levels,
                                     grid_jump=grid_step)

    output = optimised_trajectories(input_sample,
                                    N,
                                    param_file,
                                    p_levels,
                                    grid_step,
                                    k_choices,
                                    param_delim)


    filename = args.output + "_v%s_l%s_gs%s_k%s_N%s_%s.txt" % (num_params, p_levels, grid_step, k_choices, N, timestamp())
    if (len(list(maximum_combo)) == k_choices) & (m.status == GRB.OPTIMAL):
        np.savetxt(filename, output, delimiter=args.delim)
    else:
        raise RuntimeError("Solution not legal, so file not saved")
