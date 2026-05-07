import numpy as np
from numba import njit


@njit
def kendall_tau_distance_numba(rank_a, rank_b):
    """
    Compute Kendall tau distance between two rankings.
    Expects rankings in format: rank[item_index] = position (1-indexed)
    Counts pairs of items with different relative orderings.
    """
    m = rank_a.shape[0]
    inv = 0
    for i in range(m):
        for j in range(i + 1, m):
            if (rank_a[i] < rank_a[j]) != (rank_b[i] < rank_b[j]):
                inv += 1
    return inv


@njit
def borda_consensus_numba(rankings):
    """
    Compute Borda consensus from rankings array.
    Input: rankings array where rankings[agent_id, item_id-1] = rank
    Returns: x (copy of rankings as float), true_score, true_consensus
    """
    _, m = rankings.shape
    x = rankings.astype(np.float64)
    true_score = np.empty(m, dtype=np.float64)
    for k in range(m):
        true_score[k] = np.mean(x[:, k])
    true_consensus = 1 + np.argsort(np.argsort(true_score))
    return x, true_score, true_consensus


@njit
def _borda_trial_numba(
    x,
    true_score,
    true_consensus,
    edges,
    rand_idx,
    iterations,
    consensus_every,
    topk=-1,
    mask=None,
):
    n, m = x.shape
    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)
    last_consensus = 0.0

    if topk >= 0:
        order = np.argsort(true_consensus)
        top_idx = order[:topk]
    else:
        top_idx = np.empty(0, dtype=np.int64)

    # Main loop of decentralized Borda consensus algorithm
    for t in range(iterations):
        # Sample a random edge (i, j) and update their local scores by averaging
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]
        for k in range(m):
            avg = 0.5 * (x[i, k] + x[j, k])
            x[i, k] = avg
            x[j, k] = avg

        # Compute score error (MSE) across all agents
        err = 0.0
        for a in range(n):
            row_err = 0.0
            for k in range(m):
                diff = x[a, k] - true_score[k]
                row_err += diff * diff
            err += row_err / m
        score_errors[t] = err / n

        if t % consensus_every == 0:
            mean_dist = 0.0
            for a in range(n):
                ranking = 1 + np.argsort(np.argsort(x[a]))
                if topk >= 0:
                    tmp_rank = np.empty(topk, dtype=np.int64)
                    tmp_true = np.empty(topk, dtype=np.int64)
                    for p in range(topk):
                        it = top_idx[p]
                        tmp_rank[p] = ranking[it]
                        tmp_true[p] = true_consensus[it]
                    mean_dist += kendall_tau_distance_numba(tmp_rank, tmp_true)
                elif mask is not None and not mask[a] or mask is None:
                    mean_dist += kendall_tau_distance_numba(ranking, true_consensus)
            last_consensus = mean_dist / n
        consensus_errors[t] = last_consensus

    return score_errors, consensus_errors


@njit
def copeland_consensus_numba(rankings):
    """
    Compute Copeland consensus from rankings array.
    Input: rankings array where rankings[agent_id, item_id-1] = rank
    Returns: x (pairwise preference matrix), true_score (average pairwise), true_consensus
    """
    n, m = rankings.shape
    x = np.zeros((n, m, m), dtype=np.float64)
    for agent_id in range(n):
        ranking = rankings[agent_id]
        for i in range(m):
            for j in range(m):
                if ranking[i] < ranking[j]:
                    x[agent_id, i, j] = 1.0
                elif ranking[i] == ranking[j]:
                    x[agent_id, i, j] = 0.5

    true_score = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            total = 0.0
            for a in range(n):
                total += x[a, i, j]
            true_score[i, j] = total / n

    si = np.zeros(m, dtype=np.float64)
    for i in range(m):
        count = 0.0
        for j in range(m):
            if true_score[i, j] > 0.5:
                count += 1.0
        si[i] = count
    true_consensus = 1 + np.argsort(np.argsort(-si))
    return x, true_score, true_consensus


@njit
def _copeland_trial_numba(
    x,
    true_score,
    true_consensus,
    edges,
    rand_idx,
    iterations,
    consensus_every,
    topk=None,
    mask=None,
):
    n = x.shape[0]
    m = x.shape[1]

    score_errors = np.empty(iterations, dtype=np.float64)
    consensus_errors = np.empty(iterations, dtype=np.float64)
    last_consensus = 0.0

    if topk >= 0:
        order = np.argsort(true_consensus)
        top_idx = order[:topk]
    else:
        top_idx = np.empty(0, dtype=np.int64)

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
        for a in range(n):
            row_err = 0.0
            for i2 in range(m):
                for j2 in range(m):
                    diff = x[a, i2, j2] - true_score[i2, j2]
                    row_err += diff * diff
            err += row_err / (m * m)
        score_errors[t] = err / n

        if t % consensus_every == 0:
            mean_dist = 0.0
            rankings = np.empty((n, m), dtype=np.int64)
            for a in range(n):
                scores = np.zeros(m, dtype=np.float64)
                for i2 in range(m):
                    count = 0.0
                    for j2 in range(m):
                        if x[a, i2, j2] > 0.5:
                            count += 1.0
                    scores[i2] = count
                ranking = 1 + np.argsort(np.argsort(-scores))
                if topk >= 0:
                    tmp_rank = np.empty(topk, dtype=np.int64)
                    tmp_true = np.empty(topk, dtype=np.int64)
                    for p in range(topk):
                        it = top_idx[p]
                        tmp_rank[p] = ranking[it]
                        tmp_true[p] = true_consensus[it]
                    mean_dist += kendall_tau_distance_numba(tmp_rank, tmp_true)
                elif mask is not None and not mask[a] or mask is None:
                    mean_dist += kendall_tau_distance_numba(ranking, true_consensus)
                rankings[a] = ranking

            last_consensus = mean_dist / n
        consensus_errors[t] = last_consensus

    return score_errors, consensus_errors


def run_borda_trial(
    data,
    edges,
    iterations=100,
    seed=42,
    consensus_every=10,
    topk=-1,
    true_consensus=None,
    **kwargs,
):
    rankings = np.asarray(data, dtype=np.int64)
    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(edges_arr), size=iterations, dtype=np.int64)

    x, true_score, consensus = borda_consensus_numba(rankings)
    if true_consensus is None:
        true_consensus = consensus
    return _borda_trial_numba(
        x,
        true_score,
        true_consensus,
        edges_arr,
        rand_idx,
        iterations,
        consensus_every,
        topk=topk,
        **kwargs,
    )


def run_copeland_trial(
    data,
    edges,
    iterations=100,
    seed=42,
    consensus_every=10,
    topk=-1,
    true_consensus=None,
    **kwargs,
):
    rankings = np.asarray(data, dtype=np.int64)
    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(0, len(edges_arr), size=iterations, dtype=np.int64)

    x, true_score, consensus = copeland_consensus_numba(rankings)
    if true_consensus is None:
        true_consensus = consensus
    return _copeland_trial_numba(
        x,
        true_score,
        true_consensus,
        edges_arr,
        rand_idx,
        iterations,
        consensus_every,
        topk=topk,
        **kwargs,
    )


def run_borda_trials(datasets, edges, iterations=100, seed=42, n_trials=10):

    edges_arr = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rand_idx = rng.integers(
        0, len(edges_arr), size=(n_trials, iterations), dtype=np.int64
    )
    final_consensuses = []
    for trial in range(n_trials):
        rankings = np.asarray(datasets[trial], dtype=np.int64)
        final_consensus = borda_consensus_at_T(
            rankings, edges_arr, rand_idx[trial], iterations
        )
        final_consensuses.append(final_consensus)
    return final_consensuses


from utils.asyladmm import asyladmm_update_numba


@njit
def decentralized_footrule_numba(
    rankings,
    edges,
    rand_idx,
    degrees,
    iterations,
    rho,
    beta,
    true_consensus,
):
    n, m = rankings.shape

    x = rankings.astype(np.float64)
    a = rankings.astype(np.float64)
    mu = np.zeros((n, m))

    errors = np.empty(iterations)

    for t in range(iterations):
        idx = rand_idx[t]
        i = edges[idx, 0]
        j = edges[idx, 1]

        asyladmm_update_numba(x, a, mu, degrees, i, j, rho, beta)

        mean_dist = 0.0
        for agent in range(n):
            ranking = 1 + np.argsort(np.argsort(x[agent]))

            mean_dist += kendall_tau_distance_numba(ranking, true_consensus)

        errors[t] = mean_dist / n

    return errors
