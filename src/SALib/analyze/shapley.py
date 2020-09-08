import random
import numpy as np
import openturns as ot


from ..util import read_param_file

def analyze(problem, y, n_perms, n_var, n_outer, n_inner, n_boot=1):
    """Computes the Shapley indices.
    
    The Shapley indices are computed from the computed samples. In addition
    to the Shapley indices, the first-order and total Sobol' indices are
    also computed.
    
    Parameters
    ----------
    n_boot : int
        The number of bootstrap samples.
        
    Returns
    -------
    indice_results : dict
        The sensitivity results of the estimation.
    
    """
    y = np.array(y)

    assert isinstance(n_boot, int), "n_boot should be an integer."
    assert n_boot > 0, "n_boot should be positive."
    
    dim = problem['num_vars']
    seed = problem.get('seed', 42)

    if n_perms is None:
        estimation_method = 'exact'
        ot.RandomGenerator_SetSeed(seed)
        np.random.seed(seed)
        random.seed(seed)
        perms = list(ot.KPermutations(dim, dim).generate())
        n_perms = len(perms)
    else:
        estimation_method = 'random'
        ot.RandomGenerator_SetSeed(seed)
        np.random.seed(seed)
        random.seed(seed)
        perms = [np.random.permutation(dim) for i in range(n_perms)]

    n_realization = 1
    output_sample_1_full = y[:n_var]
    output_sample_2_full = y[n_var:].reshape((n_perms, dim-1, n_outer, n_inner, n_realization))

    # Initialize Shapley, main and total Sobol effects for all players
    shapley_indices = np.zeros((dim, n_boot, n_realization))
    first_indices = np.zeros((dim, n_boot, n_realization))
    total_indices = np.zeros((dim, n_boot, n_realization))
    shapley_indices_2 = np.zeros((dim, n_realization))
    first_indices_2 = np.zeros((dim, n_realization))
    total_indices_2 = np.zeros((dim, n_realization))

    n_first = np.zeros((dim, n_boot, n_realization))
    n_total = np.zeros((dim, n_boot, n_realization))
    c_hat = np.zeros((n_perms, dim, n_boot, n_realization))

    if estimation_method == 'random':
        boot_perms = np.zeros((n_perms, n_boot), dtype=int)
    
    # TODO: ugly... Do it better
    variance = np.zeros((n_boot, n_realization))
    perms = np.asarray(perms)


    for i in range(n_boot):
        # Bootstrap sample indexes
        # The first iteration is computed over the all sample.
        if i > 0:
            boot_var_idx = np.random.randint(0, n_var, size=(n_var, ))
            if estimation_method == 'exact':
                boot_No_idx = np.random.randint(0, n_outer, size=(n_outer, ))
            else:
                boot_n_perms_idx = np.random.randint(0, n_perms, size=(n_perms, ))
                boot_perms[:, i] = boot_n_perms_idx
        else:
            boot_var_idx = range(n_var)
            if estimation_method == 'exact':
                boot_No_idx = range(n_outer)
            else:
                boot_n_perms_idx = range(n_perms)
                boot_perms[:, i] = boot_n_perms_idx
            
        # Output variance
        var_y = output_sample_1_full[boot_var_idx].var(axis=0, ddof=1)

        variance[i] = var_y

        # Conditional variances
        if estimation_method == 'exact':
            output_sample_2 = output_sample_2_full[:, :, boot_No_idx]
        else:
            output_sample_2 = output_sample_2_full[boot_n_perms_idx]
        
        c_var = output_sample_2.var(axis=3, ddof=1)

        # Conditional exceptations
        c_mean_var = c_var.mean(axis=2)

        # Cost estimation
        c_hat[:, :, i] = np.concatenate((c_mean_var, [var_y.reshape(1, -1)]*n_perms), axis=1)

    # Cost variation
    delta_c = c_hat.copy()
    delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
    
    for i in range(n_boot):
        if estimation_method == 'random':
            boot_n_perms_idx = boot_perms[:, i]
            tmp_perms = perms[boot_n_perms_idx]
        else:
            tmp_perms = perms

        # Estimate Shapley, main and total Sobol effects
        for i_p, perm in enumerate(tmp_perms):
            # Shapley effect
            shapley_indices[perm, i] += delta_c[i_p, :, i]
            shapley_indices_2[perm] += delta_c[i_p, :, 0]**2

            # Total effect
            total_indices[perm[0], i] += c_hat[i_p, 0, i]
            total_indices_2[perm[0]] += c_hat[i_p, 0, 0]**2
            n_total[perm[0], i] += 1

            # First order effect
            first_indices[perm[-1], i] += c_hat[i_p, -2, i]
            first_indices_2[perm[-1]] += delta_c[i_p, -2, 0]**2
            n_first[perm[-1], i] += 1
        
    N_total = n_perms / dim if estimation_method == 'exact' else n_total
    N_first = n_perms / dim if estimation_method == 'exact' else n_first
    
    N_total_2 = n_perms / dim if estimation_method == 'exact' else n_total[:, 0]
    N_first_2 = n_perms / dim if estimation_method == 'exact' else n_first[:, 0]
    
    output_variance = variance[np.newaxis]
    shapley_indices = shapley_indices / n_perms / output_variance
    total_indices = total_indices / N_total / output_variance
    first_indices = first_indices / N_first / output_variance

    if estimation_method == 'random':
        output_variance_2 = output_variance[:, 0]
        shapley_indices_2 = shapley_indices_2 / n_perms / output_variance_2**2
        shapley_indices_SE = np.sqrt((shapley_indices_2 - shapley_indices[:, 0]**2) / n_perms)

        total_indices_2 = total_indices_2 / N_total_2 / output_variance_2**2
        total_indices_SE = np.sqrt((total_indices_2 - total_indices[:, 0]**2) / N_total_2)

        first_indices_2 = first_indices_2 / N_first_2 / output_variance_2**2
        first_indices_SE = np.sqrt((first_indices_2 - first_indices[:, 0]**2) / N_first_2)
    else:
        shapley_indices_SE = None
        total_indices_SE = None
        first_indices_SE = None
        
    first_indices = 1. - first_indices


    return {
        'S1': first_indices.reshape(dim, -1).mean(axis=1),
        'ST': total_indices.reshape(dim, -1).mean(axis=1),
        'Sh': shapley_indices.reshape(dim, -1).mean(axis=1),
    }


def cli_parse(parser):
    parser.add_argument('--n-perms', type=int, required=False, default=None,
                        help='Number of permutations. Should be less equal than factorial(dimension).')
    parser.add_argument('--n-outer', type=int, required=True,
                        help='Number of outer conditional samples.')
    parser.add_argument('--n-inner', type=int, required=True,
                        help='Number of inner conditional samples.')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=100,
                        help='Number of bootstrap resamples for Sobol '
                        'confidence intervals')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
                   usecols=(args.column,))

    analyze(problem, Y, 
        args.n_perms, 
        args.samples, 
        args.n_outer, 
        args.n_inner, 
        args.resamples)