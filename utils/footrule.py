from abc import ABC, abstractmethod

import numpy as np

from utils.asyladmm import AsylADMM
from utils.consensus import footrule_consensus
from utils.helper import consensus_to_array, list_dicts_to_array
from utils.numba_gossip import kendall_tau_distance_numba


class ConsensusMethod(ABC):
    def __init__(self, data, config):
        self.x = None
        self.n = len(data)
        self.m = len(data[0])
        self.ranking = [np.array(list(range(1, self.m + 1))) for _ in range(self.n)]
        self.config = config
        self.name = self.__class__.__name__
        self.score_errors = []
        self.consensus_errors = []
        self.true_score = None
        self.true_consensus = None
        self.reference = config.get("reference", None)
        self.bias = None
        self.flag = False

    @abstractmethod
    def init_true_values(self): 
        pass

    def init_x(self, data):
        self.x = np.array(data, dtype=float)

    def measure_convergence(self):
        """Measure how much agents' rankings differ."""
        mean_distance = 0
        for i in range(self.n):
            ranking = self.ranking[i]
            dist = kendall_tau_distance_numba(ranking, self.true_consensus)
            mean_distance += dist
        return mean_distance / self.n

    def get_local_ranking(self, i):
        """Get current ranking from agent's local scores."""
        if not self.flag:
            ranking = 1 + np.argsort(np.argsort(self.x[i]))
        else:
            scores = np.ceil(self.x[i] * 2 - 0.5) / 2
            ranking = 1 + np.argsort(np.argsort(scores))
        return ranking

    def update(self, i: int, j: int):

        if not self.flag:
            avg = (self.x[i] + self.x[j]) / 2
            self.x[i], self.x[j] = avg, avg
        else:
            for method in self.methods:
                method.update(i, j)
            self.x[i] = np.array([method.x[i] for method in self.methods])
            self.x[j] = np.array([method.x[j] for method in self.methods])
        self.ranking[i] = self.get_local_ranking(i)
        self.ranking[j] = self.get_local_ranking(j)
        self._compute_errors()

    def _compute_errors(self):
        error = np.mean((self.x - self.true_score) ** 2, axis=1).mean()
        self.score_errors.append(error)
        consensus_error = self.measure_convergence()
        self.consensus_errors.append(consensus_error)

    def run_trial(self, edges, iterations: int = 100, seed: int = 42):
        np.random.seed(seed)
        for _ in range(iterations):
            idx = np.random.randint(len(edges))
            i, j = edges[idx]
            self.update(i, j)
        return self.score_errors, self.consensus_errors


class DecentralizedFootrule(ConsensusMethod):
    def __init__(self, data, config, degrees):
        super().__init__(data, config)
        self.list_dicts = data
        array_data = list_dicts_to_array(data)
        self.init_x(array_data)
        self.init_true_values()
        self.flag = True
        self.methods = [
            AsylADMM([array_data[j][i] for j in range(self.n)], self.config, degrees)
            for i in range(self.m)
        ]
        if self.config.get("verbose", False):
            print("is correct?", np.all(self.true_consensus == range(1, self.m + 1)))
            print("Estimated footrule consensus", self.true_consensus)

    def init_true_values(self):
        consensus, f_scores = footrule_consensus(self.list_dicts)
        self.true_consensus = consensus_to_array(consensus, self.m)
        self.true_score = np.array([f_scores[item] for item in range(1, self.m + 1)])
        self.bias = kendall_tau_distance_numba(
            np.asarray(self.true_consensus), np.asarray(self.reference)
        )