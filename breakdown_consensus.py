import numpy as np
from numba import njit

from utils.numba_gossip import (
    borda_consensus_numba,
    copeland_consensus_numba,
    decentralized_footrule_numba,
    kendall_tau_distance_numba,
    run_borda_trials,
)
from utils.helper import consensus_to_array, list_dicts_to_array


@njit
def _borda_average_consensus_error(x, true_consensus, mask=None):
    """
    Compute consensus error by averaging scores across agents and comparing to true consensus.
    x: (n, m) array of scores
    true_consensus: (m,) array of true consensus ranks
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: consensus error (Kendall tau distance)
    """
    n, m = x.shape
    averaged_scores = np.empty(m, dtype=np.float64)
    for kk in range(m):
        total = 0.0
        count = 0
        for agent in range(n):
            if mask is None or (mask is not None and not mask[agent]):
                total += x[agent, kk]
                count += 1
        if count > 0:
            averaged_scores[kk] = total / count
        else:
            averaged_scores[kk] = 0.0

    ranking = 1 + np.argsort(np.argsort(averaged_scores))

    return kendall_tau_distance_numba(ranking, true_consensus)


@njit
def _copeland_average_consensus_error(x, true_consensus, mask=None):
    """
    Compute consensus error by averaging pairwise preferences across agents and comparing to true consensus.
    x: (n, m, m) array of pairwise preferences
    true_consensus: (m,) array of true consensus ranks
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: consensus error (Kendall tau distance)
    """

    n = x.shape[0]
    m = x.shape[1]

    averaged_pairwise = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            total = 0.0
            count = 0
            for agent in range(n):
                if mask is None or (mask is not None and not mask[agent]):
                    total += x[agent, i, j]
                    count += 1
            if count > 0:
                averaged_pairwise[i, j] = total / count
            else:
                averaged_pairwise[i, j] = 0.0

    scores = np.empty(m, dtype=np.float64)
    for i in range(m):
        cnt = 0.0
        for j in range(m):
            if averaged_pairwise[i, j] > 0.5:
                cnt += 1.0
        scores[i] = cnt

    ranking = 1 + np.argsort(np.argsort(-scores))
    error = kendall_tau_distance_numba(ranking, true_consensus)
    return error


@njit
def _borda_trial_numba_average_state(
    x,
    true_score,
    true_consensus,
    edges,
    rand_idx,
    iterations,
    consensus_every,
    mask=None,
):
    """
    Compute trial where at each step we average scores across the edge and then compute consensus error by averaging scores across agents and comparing to true consensus.
    x: (n, m) array of scores
    true_score: (m,) array of true average scores across agents
    true_consensus: (m,) array of true consensus ranks
    edges: (num_edges, 2) array of edge pairs
    rand_idx: (iterations,) array of random edge indices to sample at each iteration
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: score_errors, consensus_errors
    """
    n, m = x.shape
    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)
    last_consensus = 0.0

    for t in range(iterations):
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]
        for k in range(m):
            avg = 0.5 * (x[i, k] + x[j, k])
            x[i, k] = avg
            x[j, k] = avg

        err = 0.0
        for agent in range(n):
            row_err = 0.0
            for k in range(m):
                diff = x[agent, k] - true_score[k]
                row_err += diff * diff
            err += row_err / m
        score_errors[t] = err / n

        if t % consensus_every == 0:
            last_consensus = _borda_average_consensus_error(
                x, true_consensus, mask=mask
            )
        consensus_errors[t] = last_consensus

    return score_errors, consensus_errors


@njit
def _copeland_trial_numba_average_state(
    x,
    true_score,
    true_consensus,
    edges,
    rand_idx,
    iterations,
    consensus_every,
    mask=None,
):
    """
    Compute trial where at each step we average pairwise preferences across the edge and then compute consensus error by averaging pairwise preferences across agents and comparing to true consensus.
    x: (n, m, m) array of pairwise preferences
    true_score: (m, m) array of true average pairwise preferences across agents
    true_consensus: (m,) array of true consensus ranks
    edges: (num_edges, 2) array of edge pairs
    rand_idx: (iterations,) array of random edge indices to sample at each iteration
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: score_errors, consensus_errors
    """
    n = x.shape[0]
    m = x.shape[1]

    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)
    last_consensus = 0.0

    for t in range(iterations):
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]
        for a in range(m):
            for b in range(m):
                avg = 0.5 * (x[i, a, b] + x[j, a, b])
                x[i, a, b] = avg
                x[j, a, b] = avg

        err = 0.0
        for agent in range(n):
            row_err = 0.0
            for i2 in range(m):
                for j2 in range(m):
                    diff = x[agent, i2, j2] - true_score[i2, j2]
                    row_err += diff * diff
            err += row_err / (m * m)
        score_errors[t] = err / n

        if t % consensus_every == 0:
            last_consensus = _copeland_average_consensus_error(
                x, true_consensus, mask=mask
            )
        consensus_errors[t] = last_consensus

    return score_errors, consensus_errors


def run_borda_trial(
    data,
    edges,
    true_consensus,
    iterations=100,
    seed=42,
    consensus_every=10,
    **kwargs,
):
    """
    Compute trial where at each step we average scores across the edge and then compute consensus error by averaging scores across agents and comparing to true consensus.
    data: list of ranking dictionaries for each agent
    edges: (num_edges, 2) array of edge pairs
    true_consensus: (m,) array of true consensus ranks
    iterations: number of iterations to run
    seed: random seed for sampling edges
    consensus_every: how often to compute consensus error
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: score_errors, consensus_errors
    """
    rankings = np.asarray(data, dtype=np.int64)
    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(edges_arr), size=iterations, dtype=np.int64)

    x, true_score, _ = borda_consensus_numba(rankings)
    return _borda_trial_numba_average_state(
        x,
        true_score,
        true_consensus,
        edges_arr,
        rand_idx,
        iterations,
        consensus_every,
        **kwargs,
    )


def run_copeland_trial(
    data,
    edges,
    true_consensus,
    iterations=100,
    seed=42,
    consensus_every=10,
    **kwargs,
):
    """
    Compute trial where at each step we average pairwise preferences across the edge and then compute consensus error by averaging pairwise preferences across agents and comparing to true consensus.
    data: list of ranking dictionaries for each agent
    edges: (num_edges, 2) array of edge pairs
    true_consensus: (m,) array of true consensus ranks
    iterations: number of iterations to run
    seed: random seed for sampling edges
    consensus_every: how often to compute consensus error
    mask: (n,) boolean array indicating which agents are corrupted (True if corrupted)
    Returns: score_errors, consensus_errors
    """
    rankings = np.asarray(data, dtype=np.int64)
    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(edges_arr), size=iterations, dtype=np.int64)

    x, true_score, _ = copeland_consensus_numba(rankings)
    return _copeland_trial_numba_average_state(
        x,
        true_score,
        true_consensus,
        edges_arr,
        rand_idx,
        iterations,
        consensus_every,
        **kwargs,
    )


__all__ = [
    "borda_consensus_numba",
    "consensus_to_array",
    "copeland_consensus_numba",
    "decentralized_footrule_numba",
    "kendall_tau_distance_numba",
    "list_dicts_to_array",
    "run_borda_trial",
    "run_borda_trials",
    "run_copeland_trial",
]
