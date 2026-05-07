import numpy as np


# create a dict of dicts for pairwise scores
def get_pairwise_scores(list_dicts):
    items = sorted(list_dicts[0].keys())
    dict_pairwise_scores = {a: {b: 0 for b in items} for a in items}
    for d in list_dicts:
        for a in items:
            for b in items:
                if a != b:
                    dict_pairwise_scores[a][b] += 1 if d[a] < d[b] else 0
    return dict_pairwise_scores


def borda_scores(list_dicts):
    items = sorted(list_dicts[0].keys())
    b_scores = {item: np.mean([d[item] for d in list_dicts]) for item in items}
    return b_scores


def borda_consensus(list_dicts):
    b_scores = borda_scores(list_dicts)
    consensus = sorted(b_scores.keys(), key=lambda x: b_scores[x], reverse=False)
    return consensus, b_scores


def copeland_scores(list_dicts):
    dict_pairwise_scores = get_pairwise_scores(list_dicts)
    items = sorted(list_dicts[0].keys())
    c_scores = {
        a: sum(
            1 if p / len(list_dicts) <= 0.5 else -1
            for p in dict_pairwise_scores[a].values()
        )
        for a in items
    }
    return c_scores


def copeland_consensus(list_dicts):
    c_scores = copeland_scores(list_dicts)
    consensus = sorted(c_scores.keys(), key=lambda x: c_scores[x], reverse=False)
    return consensus, c_scores


def footrule_scores(list_dicts):
    items = sorted(list_dicts[0].keys())
    f_scores = {item: np.median([d[item] for d in list_dicts]) for item in items}
    return f_scores


def footrule_consensus(list_dicts):
    f_scores = footrule_scores(list_dicts)
    consensus = sorted(f_scores.keys(), key=lambda x: f_scores[x], reverse=False)
    return consensus, f_scores


def local_kemenization(
    initial_ranking: list, pairwise_scores: dict[tuple, float], n: int, verbose=False
) -> list:
    """
    Apply local Kemenization to make an initial ranking locally Kemeny optimal.
    Algorithm: For every pair of adjacent elements (u, v): if p_{u,v} < 0.5, then swap (u, v).
    """
    current_ranking = list(initial_ranking)

    iteration = 0
    while True:
        swapped = False
        iteration += 1
        for i in range(len(current_ranking) - 1):
            u = current_ranking[i]
            v = current_ranking[i + 1]

            # If p_{u,v} < 0.5, then majority prefers v over u, so swap
            p_uv = pairwise_scores[u][v] / n
            if p_uv < 0.5:
                current_ranking[i], current_ranking[i + 1] = v, u
                swapped = True
                if verbose:
                    print(f"p_{{{u},{v}}} = {p_uv:.3f} -> SWAP!")
        swapped = False
        if not swapped:
            break
    return current_ranking
