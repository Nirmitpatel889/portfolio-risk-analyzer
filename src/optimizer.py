"""src/optimizer.py — Markowitz portfolio optimizer."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, returns, risk_free_rate=0.05, trading_days=252):
        self.returns  = returns.copy()
        self.tickers  = list(returns.columns)
        self.n        = len(self.tickers)
        self.rf       = risk_free_rate
        self.td       = trading_days
        self.mu       = returns.mean() * trading_days
        self.cov      = returns.cov()  * trading_days

    def _stats(self, w):
        ret = float(np.dot(w, self.mu))
        vol = float(np.sqrt(w @ self.cov.values @ w))
        return ret, vol

    def _neg_sharpe(self, w):
        r, v = self._stats(w)
        return -(r - self.rf) / v if v > 0 else 0.0

    def maximize_sharpe(self, max_w=1.0):
        w0  = np.ones(self.n) / self.n
        res = minimize(self._neg_sharpe, w0, method="SLSQP",
                       bounds=[(0, max_w)] * self.n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 1000, "ftol": 1e-9})
        r, v = self._stats(res.x)
        return {"weights": dict(zip(self.tickers, np.round(res.x, 4))),
                "return": round(r, 4), "volatility": round(v, 4),
                "sharpe": round((r - self.rf) / v, 4) if v > 0 else 0,
                "label": "Max Sharpe"}

    def minimize_variance(self, max_w=1.0):
        w0  = np.ones(self.n) / self.n
        res = minimize(lambda w: self._stats(w)[1]**2, w0, method="SLSQP",
                       bounds=[(0, max_w)] * self.n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 1000, "ftol": 1e-9})
        r, v = self._stats(res.x)
        return {"weights": dict(zip(self.tickers, np.round(res.x, 4))),
                "return": round(r, 4), "volatility": round(v, 4),
                "sharpe": round((r - self.rf) / v, 4) if v > 0 else 0,
                "label": "Min Variance"}

    def random_portfolios(self, n=4000):
        rng  = np.random.default_rng(42)
        rows = []
        for w in rng.dirichlet(np.ones(self.n), size=n):
            r, v = self._stats(w)
            rows.append({"return": r, "volatility": v,
                         "sharpe": (r - self.rf) / v if v > 0 else 0})
        return pd.DataFrame(rows)

    def efficient_frontier(self, n_points=60):
        lo, hi = float(self.mu.min()), float(self.mu.max())
        rows = []
        for target in np.linspace(lo, hi, n_points):
            res = minimize(
                lambda w: self._stats(w)[1]**2, np.ones(self.n) / self.n,
                method="SLSQP", bounds=[(0, 1)] * self.n,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1},
                    {"type": "eq", "fun": lambda w, t=target: self._stats(w)[0] - t},
                ],
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.success:
                r, v = self._stats(res.x)
                rows.append({"return": r, "volatility": v,
                             "sharpe": (r - self.rf) / v if v > 0 else 0})
        return pd.DataFrame(rows)

    def weights_df(self, current_weights=None):
        ms = self.maximize_sharpe()
        mv = self.minimize_variance()
        data = {"Max Sharpe": ms["weights"], "Min Variance": mv["weights"]}
        if current_weights is not None:
            data["Current"] = dict(zip(self.tickers,
                                       np.round(current_weights, 4)))
        return pd.DataFrame(data).applymap(lambda x: f"{x:.1%}")
