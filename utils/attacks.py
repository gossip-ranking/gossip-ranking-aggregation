import numpy as np
from numba import njit


@njit
def random_attack(n_contaminated, m, seed):
    """
    Generate random rankings for contaminated nodes.
    Returns: (n_contaminated, m) array of random rankings
    """
    np.random.seed(seed)
    rankings = np.empty((n_contaminated, m), dtype=np.int64)
    for i in range(n_contaminated):
        rankings[i] = np.random.permutation(m) + 1
    return rankings


@njit
def reversed_consensus_attack(true_consensus, n_contaminated, m):
    """
    All contaminated nodes report the exact reverse of true consensus.
    """
    reverse_ranking = m + 1 - true_consensus
    rankings = np.empty((n_contaminated, m), dtype=np.int64)
    for i in range(n_contaminated):
        rankings[i] = reverse_ranking
    return rankings


def create_corrupted_dataset(
    honest_rankings, true_consensus, epsilon, attack_type="random", seed=0
):
    """
    Replace a fraction epsilon of agents with contaminated ones.
    """
    n, m = honest_rankings.shape
    n_contaminated = round(epsilon * n)

    if n_contaminated <= 0:
        return honest_rankings.copy(), np.zeros(n, dtype=bool)

    if attack_type == "random":
        contaminated_rankings = random_attack(n_contaminated, m, seed)
    elif attack_type != "random":
        # Default to reversed consensus attack if not random
        contaminated_rankings = reversed_consensus_attack(
            true_consensus, n_contaminated, m
        )

    rng = np.random.default_rng(seed)
    contaminated_indices = rng.choice(n, n_contaminated, replace=False)

    corrupted_rankings = honest_rankings.copy()
    corrupted_rankings[contaminated_indices] = contaminated_rankings

    mask = np.zeros(n, dtype=bool)
    mask[contaminated_indices] = True

    return corrupted_rankings, mask
