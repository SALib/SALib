from gurobipy import *

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

    m.addConstr(quicksum([x[i, j] for i in I for j in range(i+1,N)]) == choose(k_choices, 2), "All")

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

    # m.params.MarkowitzTol=1e-04
    # m.params.MIPGapAbs=0
    # m.params.MIPGap=0.05
    # m.params.FeasibilityTol=1e-09
    # m.params.OptimalityTol=1e-09
    # m.params.IntFeasTol=1e-9
    m.params.MIPFocus=1 # Focus on feasibility over optimality
    # m.params.cuts=0

    # m.write("file.lp")
    m.ModelSense = GRB.MAXIMIZE
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print m.objval()
