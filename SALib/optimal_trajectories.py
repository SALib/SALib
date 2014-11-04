from gurobipy import *
import esme as mo
from sample import common_args
import numpy as np
import random as rd
from util import scale_samples, read_param_file
from scipy.misc import comb as nchoosek

'''
Run using
optimal_trajectories.py -n=10
                        -p=esme_param.txt
                        -o=test_op.txt
                        -s=12892
                        --num-levels=4
                        --grid-jump=2

'''

def model(N, k_choices, distance_matrix):
    m = Model("distance1")
    I = range(N)
    big_M = 1e3

    dm=distance_matrix**2

    y,x = {},{}
    for i in I:
        y[i] = m.addVar(vtype="B", obj=0, name="y[%s]" % i)
        for j in range(i+1, N):
            x[i,j] = m.addVar(vtype="B", obj=1.0, name="x[%s,%s]" % (i, j))
    m.update()

    m.setObjective(quicksum([x[i, j] * dm[j][i] for i in I for j in range(i+1,N)]))

    m.addConstr(quicksum([x[i, j] for i in I for j in range(i+1,N)]) == nchoosek(k_choices, 2), "All")

    for i in I:
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                m.addConstr( x[j, k] + 1 >= (x[i, j] + x[i, k]),
                            "Equiv[%s%s]<-[%s%s]&[%s%s]"%(j, k, i, j, i, k))
                m.addConstr( x[i, k] + 1 >= (x[i, j] + x[j, k]),
                            "Equiv[%s%s]<-[%s%s]&[%s%s]"%(i, k, i, j, j, k))
                m.addConstr( x[i, j] + 1 >= (x[i, k] + x[j, k]),
                            "Equiv[%s%s]<-[%s%s]&[%s%s]"%(i, k, i, j, j, k))

    m.update()
    # Use triangles through pairs of (x[i,j], x[i,k]) forcing a third x[j,k]
    for i in I:
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                m.addConstr(x[j, k] + 1 >= (x[i, j] + x[i, k]), "Equiv[%s,%s]<-[%s,%s]&[%s,%s]"%(j,k,i,j,i,k))
                m.addConstr(x[i, k] + 1 >= (x[i, j] + x[j, k]), "Equiv[%s,%s]<-[%s,%s]&[%s,%s]"%(i,k,i,j,j,k))
                m.addConstr(x[i, j] + 1 >= (x[i, k] + x[j, k]), "Equiv[%s,%s]<-[%s,%s]&[%s,%s]"%(i,j,i,k,j,k))

    # Finally, each combination may only appear three times in the combination list
    for i in I:
        m.addConstr( quicksum(x[i, j] for j in range(i+1,N)) + quicksum(x[k, i] for k in range(0,i)),
                    '<=',
                    (k_choices - 1) + (y[i] * big_M),
                    "Only three[%s]"%i)
        m.addConstr( quicksum(x[i, j] for j in range(i+1,N)) + quicksum(x[k, i] for k in range(0,i)) + (y[i] * big_M),
                    '>=',
                    (k_choices - 1),
                    "Only three[%s]"%i)

    m.addConstr( quicksum( y[i] for i in I ), "==", N - k_choices, name="Only %s hold" % (N-k_choices))
    m.update()
    return m


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False, default=2, help='Grid jump size (Morris only)')
    parser.add_argument('--k-choices', type=int, required=False, default=4, help='Number of choices (optimised trajectory)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)


    param_file = args.paramfile
    pf = read_param_file(param_file)
    N = args.samples
    num_params = pf['num_vars']
    bounds = pf['bounds']
    k_choices = args.k_choices
    p_levels = int(args.num_levels)
    grid_step = int(args.grid_jump)
    # Generates N(D + 1) x D matrix of samples
    input_data = mo.morris_sample(N,
                        num_params,
                        bounds,
                        num_levels=p_levels,
                        grid_jump=grid_step)
    distance_matrix = mo.compute_distance_matrix(input_data,
                                              N,
                                              num_params)

    m = model(N, k_choices, distance_matrix)
    m.params.MIPFocus=1 # Focus on feasibility over optimality

    # m.write("file.lp")
    m.ModelSense = GRB.MAXIMIZE
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print m.objval

    import re

    variables = list(m.getVars())
    x_vars = []
    for v in variables:
        if (v.X > 0 ) & (v.VarName[0] == 'x'):
                x_vars.append(v.VarName)
    b = [re.findall("\d",str(v)) for v in x_vars]
    maximum_combo = set([int(y) for z in b for y in z])

    index_list = []
    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    output = np.zeros((len(maximum_combo) * (num_params + 1), num_params))
    for counter, x in enumerate(maximum_combo):
        output[index_list[counter]] = np.array(input_data[index_list[x]])
    np.savetxt(args.output, output, delimiter=' ')
