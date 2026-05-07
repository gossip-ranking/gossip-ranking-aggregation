from utils.load_data import *
from utils.numba_gossip import *
from utils.mallows import generate_mallows_mixture
from utils.helper import list_dicts_to_array, consensus_to_array
from utils.graph import generate_graph


from tqdm import trange

import os
import time
import json
import pickle

from numba import set_num_threads
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np


def _resolve_n_jobs(n_jobs):
    if n_jobs is not None:
        return max(1, int(n_jobs))
    env_jobs = os.getenv("SLURM_CPUS_PER_TASK")
    if env_jobs is None:
        return 1
    try:
        return max(1, int(env_jobs))
    except ValueError:
        return 1


def save_results(results, filepath):
    """Save results dict preserving structure. Use .pkl for pickle or .json for JSON."""
    if filepath.endswith(".pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    elif filepath.endswith(".json"):

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (dict, list)):
                return obj
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        json_results = {}
        for method, graph_dict in results.items():
            json_results[method] = {}
            for graph_type, data_list in graph_dict.items():
                json_results[method][graph_type] = [
                    (convert_to_serializable(arr1), convert_to_serializable(arr2))
                    for arr1, arr2 in data_list
                ]
        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError("Use .pkl or .json extension")


def load_results(filepath):
    """Load results dict from pickle or JSON."""
    if filepath.endswith(".pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        results = {}
        for method, graph_dict in data.items():
            results[method] = {}
            for graph_type, data_list in graph_dict.items():
                results[method][graph_type] = [
                    (np.array(arr1), np.array(arr2)) for arr1, arr2 in data_list
                ]
        return results
    else:
        raise ValueError("Use .pkl or .json extension")


def _configure_worker_numba_threads():
    env_threads = os.getenv("NUMBA_NUM_THREADS")
    if env_threads is None:
        set_num_threads(1)
        return
    try:
        threads = max(1, int(env_threads))
    except ValueError:
        set_num_threads(1)
        return
    set_num_threads(threads)


def _trial_worker(args):
    (
        rankings,
        graph_type,
        trial,
        iterations,
        consensus_every,
        topk,
    ) = args
    _configure_worker_numba_threads()
    n = rankings.shape[0]
    G = generate_graph(n, type=graph_type, seed=trial)
    edges = np.asarray(list(G.edges()), dtype=np.int64)
    score_errors, consensus_errors = run_borda_trial(
        rankings,
        edges,
        iterations=iterations,
        seed=trial,
        consensus_every=consensus_every,
        topk=topk,
    )
    score_errors_c, consensus_errors_c = run_copeland_trial(
        rankings,
        edges,
        iterations=iterations,
        seed=trial,
        consensus_every=consensus_every,
        topk=topk,
    )
    return (score_errors, consensus_errors), (score_errors_c, consensus_errors_c)


def run_trials(
    data, n_trials=10, iterations=1000, consensus_every=20, topk=-1, n_jobs=None
):
    graph_types = ["Complete", "Watts-Strogatz", "2D Grid", "Geometric"]
    results = {
        method: {graph_type: [] for graph_type in graph_types}
        for method in ["Borda", "Copeland"]
    }
    rankings = list_dicts_to_array(data)
    n_jobs = _resolve_n_jobs(n_jobs)

    for graph_type in graph_types:
        print(f"Running trials for graph type: {graph_type}")
        start_time = time.perf_counter()
        errors = {
            "Borda": {"score": [], "consensus": []},
            "Copeland": {"score": [], "consensus": []},
        }

        if n_jobs == 1:
            for trial in trange(n_trials):
                borda_res, copeland_res = _trial_worker(
                    (rankings, graph_type, trial, iterations, consensus_every, topk)
                )
                errors["Borda"]["score"].append(borda_res[0])
                errors["Borda"]["consensus"].append(borda_res[1])
                errors["Copeland"]["score"].append(copeland_res[0])
                errors["Copeland"]["consensus"].append(copeland_res[1])
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
                args = [
                    (rankings, graph_type, trial, iterations, consensus_every, topk)
                    for trial in range(n_trials)
                ]
                for borda_res, copeland_res in executor.map(_trial_worker, args):
                    errors["Borda"]["score"].append(borda_res[0])
                    errors["Borda"]["consensus"].append(borda_res[1])
                    errors["Copeland"]["score"].append(copeland_res[0])
                    errors["Copeland"]["consensus"].append(copeland_res[1])
        for method in errors:
            all_err_scores = np.array(errors[method]["score"])
            all_err_consensus = np.array(errors[method]["consensus"])
            mean_err_scores = np.mean(all_err_scores, axis=0)
            mean_err_consensus = np.mean(all_err_consensus, axis=0)
            std_err_scores = np.std(all_err_scores, axis=0)
            std_err_consensus = np.std(all_err_consensus, axis=0)
            results[method][graph_type].extend(
                [
                    (mean_err_scores, std_err_scores),
                    (mean_err_consensus, std_err_consensus),
                ]
            )
        end_time = time.perf_counter() - start_time
        print(end_time)
    return results


if __name__ == "__main__":
    m = 10
    components = [
        {"ref_ranking": np.arange(1, m + 1), "phi": 0.4, "weight": 0.7},
        {"ref_ranking": np.arange(m, 0, -1), "phi": 0.6, "weight": 0.3},
    ]

    data, m, n = generate_mallows_mixture(n=1000, m=m, components=components, seed=0)
    results = run_trials(data, n_trials=500, iterations=50000)
    save_results(results, "mixture_07+03_mallows_m10_n1000.pkl")
    data, m, n = load_debian()
    results = run_trials(data, n_trials=250, iterations=50000)
    save_results(results, "debian.pkl")
    data, m, n = load_sushi()
    results = run_trials(data, n_trials=25, iterations=200000)
    save_results(results, "sushi.pkl")
