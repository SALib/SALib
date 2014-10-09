from __future__ import division
from __future__ import print_function
import numpy as np
from ..util import read_param_file
from ..sample import saltelli
import common_args, sobol
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Build a metamodel with Support Vector Regression (SVR),
# then run Sobol analysis on the metamodel.
# Returns a dictionary with keys 'S1', 'ST', 'S2', 'R2_cv', and 'R2_fullset'
# Where 'R2_cv' is the mean R^2 value from k-fold cross validation,
# 'R2_fullset' is the R^2 value when the metamodel is applied to all observed data,
# and the other entries are lists of size D (the number of parameters)
# containing the indices in the same order as the parameter file
def analyze(pfile, input_file, output_file, N_rbf=10000, column = 0, n_folds = 10,
            delim = ' ', print_to_console=False, training_sample=None):

    param_file = read_param_file(pfile)
    y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    if len(X.shape) == 1: X = X.reshape((len(X),1))
    D = param_file['num_vars']
    mms = MinMaxScaler()
    X = mms.fit_transform(X)

    # Cross-validation to choose C and epsilon parameters
    C_range = [0.01, 0.1, 1, 10, 100]
    gamma_range = [0.001, 0.1, 1, 10]
    eps_range = [0.01, 0.1, 1]
    param_grid = dict(C=C_range, gamma=gamma_range, epsilon=eps_range)
    reg = GridSearchCV(svm.SVR(), param_grid=param_grid, cv=n_folds)

    if training_sample is None: reg.fit(X, y) # will be very slow for large N
    else: 
        if training_sample > y.size: raise ValueError("training_sample is greater than size of dataset.")
        ix = np.random.randint(y.size, size=training_sample)
        reg.fit(X[ix,:], y[ix])

    X_rbf = saltelli.sample(N_rbf, pfile)
    X_rbf = mms.transform(X_rbf)
    y_rbf = reg.predict(X_rbf)

    np.savetxt("y_rbf.txt", y_rbf, delimiter=' ')

    # not using the bootstrap intervals here. For large enough N, they will go to zero.
    # (this doesn't mean the indices are accurate -- check the metamodel R^2)
    S = sobol.analyze(pfile, "y_rbf.txt", print_to_console=False, num_resamples=2)
    S.pop("S1_conf", None)
    S.pop("ST_conf", None)
    S.pop("S2_conf", None)
    S["R2_cv"] = reg.best_score_
    S["R2_fullset"] = reg.score(X,y)

    if print_to_console:
        print("# Cross-Validation Mean R^2: %f" % S["R2_cv"])
        print("# Full dataset R^2: %f" % S["R2_fullset"])
        print("\nParameter S1 ST")
        for j in range(D):
            print("%s %f %f" % (param_file['names'][j], S['S1'][j], S['ST'][j]))

        print("\nParameter_1 Parameter_2 S2")
        for j in range(D):
            for k in range(j+1, D):
                print("%s %s %f" % (param_file['names'][j], param_file['names'][k], S['S2'][j,k]))
    
    return S

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str, required=True, default=None, help='Model input file')
    parser.add_argument('-N', '--N-rbf', type=int, required=False, default=10000, help='Number of sample points on the RBF surface')
    parser.add_argument('-k', '--n-folds', type=int, required=False, default=10, help='Number of folds in SVR cross-validation')
    parser.add_argument('-t', '--training-sample', type=int, required=False, default=None, help='Subsample size to train SVR. Default uses all points in dataset.')

    args = parser.parse_args()
    analyze(args.paramfile, args.model_input_file, args.model_output_file, args.N_rbf, args.column, 
        delim=args.delimiter, n_folds=args.n_folds, print_to_console=True, 
        training_sample=args.training_sample)
