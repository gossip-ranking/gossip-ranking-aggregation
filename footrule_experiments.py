import numpy as np
from utils.graph import generate_graph
from utils.numba_gossip import (
    run_borda_trial,
    run_copeland_trial,
)
from utils.mallows import generate_mallows
from utils.results import save_results
from utils.helper import array_to_list_dicts
from methods import DecentralizedFootrule
from utils.plot import plot_multigraph_results
from tqdm import trange

import os
from numba import set_num_threads
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

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
        n_agents,
        num_alternatives,
        phi,
        references,
        graph_type,
        trial,
        iterations,
        consensus_every,
    ) = args
    _configure_worker_numba_threads()
    rankings = generate_mallows(
        n_agents, num_alternatives, references, phi=phi, seed=trial)

    G = generate_graph(n_agents, type=graph_type, seed=trial)
    edges_array = np.asarray(list(G.edges()), dtype=np.int64)
    edges_list = list(G.edges())

    score_errors, consensus_errors = run_borda_trial(
        rankings,
        edges_array,
        iterations=iterations,
        seed=trial,
        consensus_every=consensus_every,
    )
    score_errors_c, consensus_errors_c = run_copeland_trial(
        rankings,
        edges_array,
        iterations=iterations,
        seed=trial,
        consensus_every=consensus_every,
    )

    degrees = np.array([G.degree(i) for i in range(n_agents)])
    config = {
        "reference": references[0],
        "alpha": 0.5,
        "rho": 2,
        "verbose": False,
    }
    footrule_method = DecentralizedFootrule(array_to_list_dicts(rankings), config, degrees)
    score_errors_f, consensus_errors_f = footrule_method.run_trial(
        edges_list, iterations=iterations, seed=trial
    )

    return (
        (score_errors, consensus_errors),
        (score_errors_c, consensus_errors_c),
        (score_errors_f, consensus_errors_f),
    )


def run_mallows_trials(
    n_agents,
    num_alternatives,
    phi,
    references,
    n_trials=10,
    iterations=100,
    consensus_every=10,
    graph_types=None,
    n_jobs=None,
):
    """Run trials with Mallows data across multiple graph types."""
    if graph_types is None:
        graph_types = ["Complete", "Watts-Strogatz", "Geometric"]

    results = {
        method: {graph_type: [] for graph_type in graph_types}
        for method in ["Borda", "Copeland", "Footrule"]
    }

    eigenvalues_by_graph = {graph_type: [] for graph_type in graph_types}
    n_jobs = _resolve_n_jobs(n_jobs)

    for graph_type in graph_types:
        print(f"Running trials for graph type: {graph_type}")
        errors = {
            "Borda": {"score": [], "consensus": []},
            "Copeland": {"score": [], "consensus": []},
            "Footrule": {"score": [], "consensus": []},
        }

        if n_jobs == 1:
            for trial in trange(n_trials):
                (
                    borda_res,
                    copeland_res,
                    footrule_res,
                    lambda2,
                ) = _trial_worker(
                    (
                        n_agents,
                        num_alternatives,
                        phi,
                        references,
                        graph_type,
                        trial,
                        iterations,
                        consensus_every,
                    )
                )
                errors["Borda"]["score"].append(borda_res[0])
                errors["Borda"]["consensus"].append(borda_res[1])
                errors["Copeland"]["score"].append(copeland_res[0])
                errors["Copeland"]["consensus"].append(copeland_res[1])
                errors["Footrule"]["score"].append(footrule_res[0])
                errors["Footrule"]["consensus"].append(footrule_res[1])
                eigenvalues_by_graph[graph_type].append(lambda2)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
                args = [
                    (
                        n_agents,
                        num_alternatives,
                        phi,
                        references,
                        graph_type,
                        trial,
                        iterations,
                        consensus_every,
                    )
                    for trial in range(n_trials)
                ]
                for borda_res, copeland_res, footrule_res, lambda2 in executor.map(
                    _trial_worker, args
                ):
                    errors["Borda"]["score"].append(borda_res[0])
                    errors["Borda"]["consensus"].append(borda_res[1])
                    errors["Copeland"]["score"].append(copeland_res[0])
                    errors["Copeland"]["consensus"].append(copeland_res[1])
                    errors["Footrule"]["score"].append(footrule_res[0])
                    errors["Footrule"]["consensus"].append(footrule_res[1])
                    eigenvalues_by_graph[graph_type].append(lambda2)

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
    return results


if __name__ == "__main__":
    np.random.seed(42)
    n_agents = 151
    n_trials = 100
    iterations = 2000
    consensus_every = 3
    num_alternatives = 8
    phi = 0.5

    ref_ranking = np.random.permutation(num_alternatives) + 1
    reference = tuple((i,) for i in ref_ranking)
    references = [reference]

    results = run_mallows_trials(
        n_agents=n_agents,
        num_alternatives=num_alternatives,
        phi=phi,
        references=references,
        n_trials=n_trials,
        iterations=iterations,
        consensus_every=consensus_every,
    )
    save_results(results, "mallows_footrule.pkl")
    plot_multigraph_results(results, dataset_name="mallows", legend="both")
