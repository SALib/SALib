import numpy as np

__all__ = ["uniform", "normal", "latin_hypercube", "saltelli", "morris_oat", "scale_samples", "fast_sampler"]

def scale_samples(params, bounds):
    for i, b in enumerate(bounds):
        params[:,i] = params[:,i] * (b[1] - b[0]) + b[0]