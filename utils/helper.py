from preflibtools.instances import OrdinalInstance
import numpy as np


def instance_to_rankings(instance):
    """
    Convert a preflib instance to a list of rankings.
    Returns a list of tuples, where each tuple is a ranking in permutation format (ranking[pos] = item_id).
    """
    flat = instance.flatten_strict()
    rankings = []
    for ranking, count in flat:
        rankings.extend([tuple(map(int, ranking))] * count)
    return rankings


def rankings_to_instance(rankings):
    """
    Convert a list of rankings to a preflib instance.
    Returns an OrdinalInstance object.
    """
    instance = OrdinalInstance()
    instance.num_alternatives = len(rankings[0])
    for ranking in rankings:
        instance.append_order(ranking)
    return instance


def rankings_to_list_dicts(rankings):
    """Convert rankings to list of dicts {item_id: rank}.

    Supports two input formats for each ranking:
    1) permutation format: ranking[pos] = item_id (strict total order)
    2) rank-vector format: ranking[item_id-1] = rank (possibly with ties)
    Returns a list of dicts, where each dict maps item_id to its rank for that agent.
    """
    list_dicts = []
    for ranking in rankings:
        m = len(ranking)
        values = list(ranking)
        is_permutation = set(values) == set(range(1, m + 1))

        if is_permutation:
            d = {item: rank + 1 for rank, item in enumerate(values)}
        else:
            d = {item_id: int(values[item_id - 1]) for item_id in range(1, m + 1)}

        list_dicts.append(d)
    return list_dicts


def list_dicts_to_array(list_dicts):
    """
    Convert list of dicts {item: rank} to numpy array format.
    Returns array[agent_id, item_id-1] = rank
    """
    n = len(list_dicts)
    m = len(list_dicts[0])
    array = np.zeros((n, m), dtype=float)
    for agent_id, d in enumerate(list_dicts):
        for item, rank in d.items():
            array[agent_id, int(item) - 1] = rank
    return array


def consensus_to_array(consensus, m):
    """
    Convert consensus result (list of items) to array of ranks.
    Returns array where array[item-1] = rank of that item.
    """
    array = np.zeros(m, dtype=int)
    for rank, item in enumerate(consensus, start=1):
        array[item - 1] = rank
    return array
