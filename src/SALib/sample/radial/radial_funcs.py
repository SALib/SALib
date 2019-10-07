import numpy as np

def combine_samples(baseline, perturb):
    """Combine baseline and perturbation samples in order.

    The number of parameters will be inferred from the shape of `baseline`.

    Arguments
    ---------
    baseline : np.array
        baseline sample for one of `N` samples

    perturb : np.array
        perturbation value for a parameter

    Example
    -------
    >>> X = combine_samples(baseline, perturb)
    
    `X` will now hold:
    [[x_{1,1}, x_{1,2}, ..., x_{1,p}]
    [b_{1,1}, x_{1,2}, ..., x_{1,p}]
    [x_{1,1}, b_{1,2}, ..., x_{1,p}]
    [x_{1,1}, x_{1,2}, ..., b_{1,p}]
    ...
    [x_{N,1}, x_{N,2}, ..., x_{N,p}]
    [b_{N,1}, x_{N,2}, ..., x_{N,p}]
    [x_{N,1}, b_{N,2}, ..., x_{N,p}]
    [x_{N,1}, x_{N,2}, ..., b_{N,p}]]

    where `p` denotes the number of parameters as specified in
    `problem` and `N` is the number of samples.

    We can now run the model using the values in `X`. 
    The total number of model evaluations will be `N(p+1)`.

    Returns
    -------
    np.array
    """
    assert baseline.shape == perturb.shape, \
        "Baseline and perturbation data should be of same size"

    nrows, num_vars = baseline.shape
    group = num_vars+1
    sample_set = np.repeat(baseline, repeats=group, axis=0)

    grp_start = 0
    for i in range(nrows):
        mod = np.diag(perturb[i])

        first_ = grp_start+1

        np.copyto(sample_set[first_:first_+num_vars], mod, where=mod != 0.0)
        grp_start += group
    # End for

    return sample_set