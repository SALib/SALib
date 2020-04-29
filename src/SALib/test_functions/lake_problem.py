from typing import Union
import numpy as np
from numpy import log, sqrt
from scipy.optimize import brentq


def lake_problem(X: float, a: float = 0.1, q: float = 2.0, b: float = 0.42,
                 eps: float = 0.02) -> float:
    """Lake Problem as given in [1] and [2] modified for use as a test function.

    The `mean` and `stdev` parameters control the log normal distribution of
    natural inflows (`epsilon` in [1] and [2]).

    .. [1] Hadka, D., Herman, J., Reed, P., Keller, K., (2015).
           "An open source framework for many-objective robust decision making."
           Environmental Modelling & Software 74, 114–129. 
           doi:10.1016/j.envsoft.2015.07.014
    
    .. [2] Kwakkel, J.H, (2017). "The Exploratory Modeling Workbench: An open 
           source toolkit for exploratory modeling, scenario discovery, and 
           (multi-objective) robust decision making."
           Environmental Modelling & Software 96, 239–250. 
           doi:10.1016/j.envsoft.2017.06.054

    .. [3] Singh, R., Reed, P., Keller, K., (2015). "Many-objective robust 
           decision making for managing an ecosystem with a deeply uncertain 
           threshold response."
           Ecology and Society 20. 
           doi:10.5751/ES-07687-200312
    
    Parameters
    ----------
    X : float or np.ndarray
        normalized concentration of Phosphorus at point in time
    a : float or np.ndarray
        rate of anthropogenic pollution (0.0 to 0.1)
    q : float or np.ndarray
        exponent controlling recycling rate (2.0 to 4.5).
    b : float or np.ndarray
        decay rate for phosphorus
        (0.1 to 0.45, where default 0.42 is irreversible, as described in [1])
    eps : float or np.ndarray
          natural inflows of phosphorus (pollution), see [3]

    Returns
    -------
    * float, phosphorus pollution for a point in time
    """
    Xq = X**q
    X_t1 = X + a + (Xq / (1.0 + Xq)) - (b*X) + eps

    return X_t1


def evaluate_lake(values: np.ndarray) -> np.ndarray:
    """Evaluate the Lake Problem with an array of parameter values.

    .. [1] Hadka, D., Herman, J., Reed, P., Keller, K., (2015).
           "An open source framework for many-objective robust decision making."
           Environmental Modelling & Software 74, 114–129. 
           doi:10.1016/j.envsoft.2015.07.014

    .. [2] Singh, R., Reed, P., Keller, K., (2015). "Many-objective robust 
           decision making for managing an ecosystem with a deeply uncertain 
           threshold response."
           Ecology and Society 20. 
           doi:10.5751/ES-07687-200312

    Parameters
    ----------
    values : np.ndarray, of model inputs in the (column) order of
             a, q, b, mean, stdev

             where 
             * `a` is rate of anthropogenic pollution
             * `q` is exponent controlling recycling rate
             * `b` is decay rate for phosphorus
             * `mean` and
             * `stdev` set the log normal distribution of `eps`, see [2]

    Returns
    -------
    * np.ndarray, of Phosphorus pollution over time `t`
    """
    nvars = values.shape[0]

    a, q, b, mean, stdev = values.T

    sq_mean = mean**2
    sq_std = stdev**2
    eps = np.random.lognormal(log(sq_mean / sqrt(sq_std + sq_mean)),
                              sqrt(log(1.0 + sq_std / sq_mean)),
                              size=nvars)

    Y = np.zeros((nvars, nvars))
    for t in range(nvars):
        # First X value will be last Y value (should be 0 as we are filling an array of zeros)
        Y[t] = lake_problem(Y[t-1], a, q, b, eps)

    return Y


def evaluate(values: np.ndarray, nvars=100):
    """Evaluate the Lake Problem with an array of parameter values.

    Parameters
    ----------
    values : np.ndarray, of model inputs in the (column) order of
             a, q, b, mean, stdev, delta, alpha

    nvars : int, 
            number of decision variables to simulate (default: 100)
    

    # delta : float or np.ndarray, 
    #         discount rate (0.93 to 0.99, default: 0.98)
    

    Returns
    -------
    * np.ndarray : max_P, utility, inertia, reliability
    """
    a, q, b, _, __, delta, alpha = values.T

    nsamples = len(a)
    Y = np.empty((nsamples, 4))
    for i in range(nsamples):
        tmp = evaluate_lake(values[i, :5])

        a_i = a[i]
        q_i = q[i]

        Pcrit = brentq(lambda x: x**q_i/(1.0+x**q_i) - b[i]*x, 0.01, 1.5)

        reliability = len(tmp[tmp < Pcrit]) / nvars

        max_P = np.max(tmp)
        utility = np.sum(alpha[i]*a_i*np.power(delta[i], np.arange(nvars)))

        # In practice, `a` will be set by a separate decision model
        # See [2] in `lake_problem`
        # Here, it is a constant for a given scenario.
        # The value for `tau` (0.02) is taken from [2].
        inertia = len(a_i[a_i < 0.02]) / nvars

        Y[i, :] = max_P, utility, inertia, reliability

    return Y



if __name__ == '__main__':
    from SALib.sample import latin
    from SALib.analyze import delta

    SEED_VAL = 101

    LAKE_SPEC = {
        'num_vars': 7,
        'names': ['a', 'q', 'b', 'mean', 'stdev', 'delta', 'alpha'],
        'bounds': [[0.0, 0.1], [2.0, 4.5], [0.1, 0.45], [0.01, 0.05], [0.001, 0.005], [0.93, 0.99], [0.2, 0.5]],
        'outputs': ['max_P', 'Utility', 'Inertia', 'Reliability']
    }

    latin_samples = latin.sample(LAKE_SPEC, 2000, seed=SEED_VAL)
    Y = evaluate(latin_samples)

    for i, name in enumerate(LAKE_SPEC['outputs']):
        print(name)
        Si = delta.analyze(LAKE_SPEC, latin_samples, Y[:, i])
        print(Si.to_df())