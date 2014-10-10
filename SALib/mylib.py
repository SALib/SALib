from __future__ import division
import numpy as np
from itertools import combinations

def pl_mappable(combos, k_choices, distance_matrix):
    pairwise = np.array([y for y in combinations(range(k_choices),2)])
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    print combos
    combo_list = combos[:,pairwise[:,]]
    all_distances = distance_matrix[[combo_list[:,:,1], combo_list[:,:,0]]]
    new_scores = np.sqrt(np.einsum('ij,ij->i', all_distances, all_distances))
    return new_scores
