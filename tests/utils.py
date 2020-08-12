import numpy as np


def get_sensitivity_stats(problem, si_fabric, n=100):
    results = {}
    for _ in range(n):
        Sis = si_fabric(problem)
        for k, v in Sis.items():
            if k not in results:
                results[k] = [v]
            else:
                results[k].append(v)
                
    for k in results:
        results[k] = np.array(results[k])
        results[k] = results[k].mean(axis=0)

    return results 