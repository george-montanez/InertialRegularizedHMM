from __future__ import division
from itertools import permutations
import numpy as np
from GaussianHMM import GaussianHMM

segment_func = GaussianHMM(1,np.random.random(size=(2,2))).get_segments

def compute_cost_matrix(predicted_states, true_states):
    predicted_states -= predicted_states.min()
    N = len(true_states)
    num_states = max(true_states) + 1
    cost = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            ps = np.array(predicted_states)        
            ps[ps == j] = j + num_states
            ps[ps == i] = j
            ps[ps == j + num_states] = i
            cost[i][j] = N - np.equal(ps, true_states).sum()
    return cost            

def segment_measurements(predicted_states, true_states):
    predicted_states -= predicted_states.min()
    s1 = segment_func(predicted_states)
    s2 = segment_func(true_states)
    l1 = len(s1)
    l2 = len(s2)
    return l1/l2, max(l1, l2)/min(l1, l2), abs(l1-l2)

def max_correct(predicted_states, true_states):
    predicted_states -= predicted_states.min()
    mc = 0
    num_states = max(true_states) + 1
    best_perm = []
    for perm in permutations(range(num_states)):      
        ps = predicted_states.copy() + num_states
        for i, v in enumerate(perm):
            ps[ps == i + num_states] = v
        new_mc = np.equal(ps, true_states).sum()
        mc = max(mc, new_mc)     
    return mc, mc / true_states.shape[0], int(mc == len(true_states))

def entropies(predicted_states, true_states):
    predicted_states -= predicted_states.min()
    ps, ts = predicted_states, np.array(true_states, dtype="int8")
    K_p = np.max(ps) + 1
    K_t = np.max(ts) + 1
    N = len(ts)
    p_cluster_probs = [np.equal(ps, i).sum() / N for i in range(K_p)]
    t_cluster_probs = [np.equal(ts, i).sum() / N for i in range(K_t)]
    H_ps = sum([-p * np.log2(p) for p in p_cluster_probs if p])
    H_ts = sum([-p * np.log2(p) for p in t_cluster_probs if p])
    joint = np.zeros((K_p, K_t))
    for j in range(K_p):
        for k in range(K_t):
            joint[j][k] = np.logical_and(np.equal(ps, j), np.equal(ts, k)).sum() / N
    H_pt = -sum([joint[j][k] * np.log2(joint[j][k]) for j in range(K_p) for k in range(K_t) if joint[j][k]])
    I_pt = sum([joint[j][k] * np.log2(joint[j][k] / (p_cluster_probs[j] * t_cluster_probs[k])) for j in range(K_p) for k in range(K_t) if joint[j][k]])
    return H_ps, H_ts, H_pt, I_pt

def evaluate_prediction_set(predicted_states, true_states):
    predicted_states -= predicted_states.min()
    ents = entropies(predicted_states, true_states)
    normed_voi = (ents[0] + ents[1] - 2 * ents[3]) / ents[2]
    res = (max_correct(predicted_states, true_states), segment_measurements(predicted_states, true_states), normed_voi)
    return res
