import openturns as ot


def make_input_distribution(problem):
    """Make a distribution of input variables.
        Distribution may contain uniform and normal distributions 
        at the same time. Variables may be correlated - then 
        problem dict should have a 'corr' field for correlation matrix.
        

    Arguments
    ---------
    problem: dict

    Returns
    ---------
    distribution : openturns.ComposedDistribution

    """
    assert(not problem.get('distrs') or isinstance(problem['distrs'], list))

    D = problem['num_vars']
    bounds = problem['bounds']
    
    # set marginal distributions
    margins = []
    if problem.get('distrs'):
        for i in range(D):
            if problem['distrs'][i] == 'unif':
                margins.append(ot.Uniform(bounds[i][0], bounds[i][1]))

            elif problem['distrs'][i] == 'norm':
                margins.append(ot.Normal(bounds[i][0], bounds[i][1]))

            else:
                valid_dists = ['unif', 'norm']
                raise ValueError('Distributions: choose one of %s' %
                             ", ".join(valid_dists))

    else:
        # search for 'bounds' and make uniform mixture
        for i in range(D):
            margins.append(ot.Uniform(bounds[i][0], bounds[i][1]))


    # make copula
    corr_template = ot.CorrelationMatrix(D)
    if problem.get('corr'):
        for i in range(D):
            for j in range(D):
                corr_template[i, j] = problem['corr'][i][j]
    copula = ot.NormalCopula(corr_template)


    return ot.ComposedDistribution(margins, copula)
