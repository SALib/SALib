from __future__ import division
from itertools import combinations
import numpy as np

def pl_mappable(combos, k_choices, distance_matrix):
    new_scores = np.zeros(len(combos), dtype=np.float32)
    pairwise = np.array([y for y in combinations(range(k_choices),2)])
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    combo_list = combos[pairwise[:,]]
    all_distances = distance_matrix[[combo_list[:,1], combo_list[:,0]]]
    new_scores = np.sqrt(np.einsum('i,i', all_distances, all_distances))
    return new_scores

if __name__ == '__main__':
    pass
