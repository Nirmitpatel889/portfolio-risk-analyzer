"""src/simulation.py — Monte Carlo portfolio simulation."""

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    def __init__(self, returns, weights, trading_days=252):
        self.returns = returns.copy()
        self.weights = np.array(weights, dtype=float)
        self.td      = trading_days
        self.mu      = returns.mean().values
        self.L       = np.linalg.cholesky(returns.cov().values)

    def run(self, n=1000, horizon=252, start_value=10000, seed=42):
        rng = np.random.default_rng(seed)
        z   = rng.standard_normal((n, len(self.weights), horizon))
        cr  = np.einsum("ij,njt->nit", self.L, z) + self.mu[:, np.newaxis]
        pr  = np.einsum("nit,i->nt", cr, self.weights)
        cum = np.cumprod(1 + pr, axis=1)
        paths = np.hstack([np.ones((n, 1)), cum]) * start_value
        return pd.DataFrame(paths.T)
