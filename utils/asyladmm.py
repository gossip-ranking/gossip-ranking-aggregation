import numpy as np
from abc import ABC, abstractmethod


def prox_r(z, a_i, gamma, beta):
    """Proximal operator with closed-form solution."""
    if z < a_i - gamma * beta:
        return z + gamma * beta
    elif z > a_i + gamma:
        return z - gamma
    else:
        return a_i


class QuantileMethod(ABC):
    """
    Base class for quantile estimation methods.
    """

    def __init__(self, data, config):
        self.n = len(data)
        self.a = data.copy()
        self.x = data.copy()
        self.name = self.__class__.__name__
        self.errors = []
        self.alpha = config.get("alpha", 0.5)  # quantile level
        self.true_quantile = np.quantile(self.a, self.alpha)
        self.beta = self.alpha / (1 - self.alpha)

    @abstractmethod
    def update(self, i: int, j: int):
        pass

    def _average(self, i: int, j: int):
        avg = (self.x[i] + self.x[j]) / 2
        self.x[i], self.x[j] = avg, avg

    def _compute_errors(self):
        # consensus_error = np.std(self.x)
        quantile_error = np.mean(np.abs(self.x - self.true_quantile))
        self.errors.append(quantile_error)

    def print_info(self):
        print(f"{self.name} estimates:, ", self.x)


class AsylADMM(QuantileMethod):
    """
    AsylADMM: asynchronous lite ADMM for distributed quantile estimation.
    """

    def __init__(self, data, config, degrees):
        super().__init__(data, config)
        self.rho = config.get("rho", 1.0)  # penalty parameter
        self.degrees = degrees
        self.mu = np.zeros(self.n)

    def update(self, i: int, j: int):
        # 1. Take average
        z_e = (self.x[i] + self.x[j]) / 2
        # 2. Update dual variables
        self.mu[j] += self.rho * (z_e - self.x[j]) / self.degrees[j]
        self.mu[i] += self.rho * (z_e - self.x[i]) / self.degrees[i]
        # 3. Update primal variables
        rho_i = self.rho * self.degrees[i]
        rho_j = self.rho * self.degrees[j]
        z_i = z_e + self.mu[i] / self.rho
        z_j = z_e + self.mu[j] / self.rho
        self.x[i] = prox_r(z_i, self.a[i], 1 / (rho_i), self.beta)
        self.x[j] = prox_r(z_j, self.a[j], 1 / (rho_j), self.beta)
        self._compute_errors()


from numba import njit


@njit
def prox_r_numba(z, a_i, gamma, beta):
    if z < a_i - gamma * beta:
        return z + gamma * beta
    elif z > a_i + gamma:
        return z - gamma
    else:
        return a_i


@njit
def asyladmm_update_numba(x, a, mu, degrees, i, j, rho, beta):
    """
    x: (n, m)
    a: (n, m)
    mu: (n, m)
    degrees: (n,)
    """

    n, m = x.shape

    for k in range(m):
        # edge average
        z_e = 0.5 * (x[i, k] + x[j, k])

        # dual updates
        mu[j, k] += rho * (z_e - x[j, k]) / degrees[j]
        mu[i, k] += rho * (z_e - x[i, k]) / degrees[i]

        rho_i = rho * degrees[i]
        rho_j = rho * degrees[j]

        z_i = z_e + mu[i, k] / rho
        z_j = z_e + mu[j, k] / rho

        # proximal step
        x[i, k] = prox_r_numba(z_i, a[i, k], 1.0 / rho_i, beta)
        x[j, k] = prox_r_numba(z_j, a[j, k], 1.0 / rho_j, beta)
