import numpy as np
import openturns as ot


from ..util.distrs import make_input_distribution


def sample(problem, n_sample):
    """
    Creates the Monte-Carlo samples for correlated variables.
    
    A pick and freeze strategy is done considering the distribution of the 
    input sample. This method creates the input samples and evaluate them 
    through the model to create the output sample. Note that the variables
    should be considered independent.

    Parameters
    ----------
    model : callable
        The model function.
        
    n_sample : int
        The sampling size of the Monte-Carlo estimation.
        
    n_realization : int, optional (default=1)
        The number of realization of the meta-model.  
        
    References
    ----------
    .. [1] Thierry A Mara, Stefano Tarantola, Paola Annoni, Non-parametric
        methods for global sensitivity analysis of model output with dependent inputs
        https://hal.archives-ouvertes.fr/hal-01182302/file/Mara15EMS_HAL.pdf
    """
    assert isinstance(problem, dict), "problem should be of dict type"
    dim = problem['num_vars']
    seed = problem.get('seed', 42)
    n_realization = 1

    input_distribution = make_input_distribution(problem)
    ot.RandomGenerator.SetSeed(seed)
    np.random.seed(seed)

    assert isinstance(n_sample, int), \
        "The number of sample should be an integer"
    assert n_sample > 0, \
        "The number of sample should be positive: %d<0" % (n_sample)
    n_pairs = int(dim*(dim-1) / 2)

    # Gaussian distribution
    norm_dist = ot.Normal(dim)

    # Independent samples
    U_1 = np.asarray(norm_dist.getSample(n_sample))
    U_2 = np.asarray(norm_dist.getSample(n_sample))

    
    blocks = []
    for i in range(dim):
        # Copy of the input dstribution
        margins = [ot.Distribution(input_distribution.getMarginal(j)) for j in range(dim)]
        copula = input_distribution.getCopula()

        # 1) Pick and Freeze
        U_3_i = U_2.copy()
        U_3_i[:, 0] = U_1[:, 0]
        U_4_i = U_2.copy()
        U_4_i[:, -1] = U_1[:, -1]
        
        # 2) Permute the margins and the copula
        order_i = np.roll(range(dim), -i)
        order_i_inv = np.roll(range(dim), i)
        order_cop = np.roll(range(n_pairs), i)

        margins_i = [margins[j] for j in order_i]

        params_i = np.asarray(copula.getParameter())[order_cop]

        copula.setParameter(params_i)
        dist = ot.ComposedDistribution(margins_i, copula)

        # 3) Inverse Transformation
        tmp = dist.getInverseIsoProbabilisticTransformation()
        inv_rosenblatt_transform_i = lambda u: np.asarray(tmp(u))

        X_1_i = inv_rosenblatt_transform_i(U_1)
        X_2_i = inv_rosenblatt_transform_i(U_2)
        X_3_i = inv_rosenblatt_transform_i(U_3_i)
        X_4_i = inv_rosenblatt_transform_i(U_4_i)
        assert X_1_i.shape[1] == dim, "Wrong dimension"

        X_1_i = X_1_i[:, order_i_inv]
        X_2_i = X_2_i[:, order_i_inv]
        X_3_i = X_3_i[:, order_i_inv]
        X_4_i = X_4_i[:, order_i_inv]
        
        # 4) Model evaluations
        X = np.r_[X_1_i, X_2_i, X_3_i, X_4_i]
        blocks.append(X)
    
    return np.r_[tuple(blocks)]