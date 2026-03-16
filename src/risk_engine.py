"""src/risk_engine.py — Portfolio risk metrics."""

import numpy as np
import pandas as pd
from scipy import stats


class RiskEngine:
    def __init__(self, returns, weights, risk_free_rate=0.05,
                 benchmark=None, trading_days=252):
        w = np.array(weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w.sum():.4f}")
        if len(w) != returns.shape[1]:
            raise ValueError("Weight count must match asset count")
        self.returns = returns.copy()
        self.weights = w
        self.rf = risk_free_rate
        self.benchmark = benchmark
        self.td = trading_days
        self.pr = returns.dot(w)          # portfolio daily returns

    # ── VaR ──────────────────────────────────
    def var_historical(self, conf=0.95):
        return float(np.percentile(self.pr, (1 - conf) * 100))

    def var_parametric(self, conf=0.95):
        z = stats.norm.ppf(1 - conf)
        return float(self.pr.mean() + z * self.pr.std())

    def var_monte_carlo(self, conf=0.95, n=10000, seed=42):
        rng = np.random.default_rng(seed)
        mu  = self.returns.mean().values
        cov = self.returns.cov().values
        L   = np.linalg.cholesky(cov)
        z   = rng.standard_normal((n, len(mu)))
        sim = (z @ L.T + mu) @ self.weights
        return float(np.percentile(sim, (1 - conf) * 100))

    def cvar(self, conf=0.95):
        var  = self.var_historical(conf)
        tail = self.pr[self.pr <= var]
        return float(tail.mean()) if not tail.empty else var

    # ── Ratios ───────────────────────────────
    def sharpe(self):
        daily_rf = self.rf / self.td
        ex = self.pr - daily_rf
        if ex.std() == 0:
            return 0.0
        return float(ex.mean() * self.td / (ex.std() * np.sqrt(self.td)))

    def sortino(self):
        daily_rf = self.rf / self.td
        ex = self.pr - daily_rf
        down = ex[ex < 0]
        if down.empty:
            return float("inf")
        dstd = np.sqrt((down**2).mean()) * np.sqrt(self.td)
        return float(ex.mean() * self.td / dstd)

    def calmar(self):
        mdd = abs(self.max_drawdown())
        return float("inf") if mdd == 0 else float(self._ann_return() / mdd)

    # ── Drawdown ─────────────────────────────
    def max_drawdown(self):
        cum = (1 + self.pr).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        return float(dd.min())

    def drawdown_series(self):
        cum = (1 + self.pr).cumprod()
        return (cum - cum.cummax()) / cum.cummax()

    def cumulative_returns(self):
        return (1 + self.pr).cumprod()

    def rolling_sharpe(self, window=63):
        daily_rf = self.rf / self.td
        ex = self.pr - daily_rf
        return (ex.rolling(window).mean() * self.td /
                (ex.rolling(window).std() * np.sqrt(self.td)))

    # ── Market ───────────────────────────────
    def beta(self):
        if self.benchmark is None:
            return None
        df = pd.concat([self.pr, self.benchmark], axis=1,
                       join="inner").dropna()
        df.columns = ["p", "b"]
        cov = np.cov(df["p"], df["b"])
        return float(cov[0, 1] / cov[1, 1])

    def alpha(self):
        if self.benchmark is None:
            return None
        b = self.beta()
        pa = self._ann_return()
        n  = len(self.benchmark.dropna())
        if n == 0:
            return 0.0
        bc = (1 + self.benchmark.dropna()).prod()
        ba = float(bc ** (self.td / n) - 1) if bc > 0 else 0.0
        return float(pa - (self.rf + b * (ba - self.rf)))

    def correlation_matrix(self):
        return self.returns.corr()

    def annualized_vol(self):
        return float(self.pr.std() * np.sqrt(self.td))

    # ── Summary ──────────────────────────────
    def compute_all(self, conf=0.95):
        c = int(conf * 100)
        out = {
            "ann_return":   self._ann_return(),
            "ann_vol":      self.annualized_vol(),
            "sharpe":       self.sharpe(),
            "sortino":      self.sortino(),
            "calmar":       self.calmar(),
            "max_drawdown": self.max_drawdown(),
            f"var_hist_{c}": self.var_historical(conf),
            f"var_para_{c}": self.var_parametric(conf),
            f"var_mc_{c}":   self.var_monte_carlo(conf),
            f"cvar_{c}":     self.cvar(conf),
        }
        if self.benchmark is not None:
            out["beta"]  = self.beta()
            out["alpha"] = self.alpha()
        return out

    def summary_df(self, conf=0.95):
        m = self.compute_all(conf)
        c = int(conf * 100)
        rows = [
            ("Returns",  "Annualized return",     f"{m['ann_return']:.2%}"),
            ("Returns",  "Annualized volatility",  f"{m['ann_vol']:.2%}"),
            ("Returns",  "Sharpe ratio",           f"{m['sharpe']:.2f}"),
            ("Returns",  "Sortino ratio",          f"{m['sortino']:.2f}"),
            ("Returns",  "Calmar ratio",           f"{m['calmar']:.2f}"),
            ("Drawdown", "Max drawdown",           f"{m['max_drawdown']:.2%}"),
            ("VaR",      f"Historical ({c}%)",     f"{m[f'var_hist_{c}']:.2%}"),
            ("VaR",      f"Parametric ({c}%)",     f"{m[f'var_para_{c}']:.2%}"),
            ("VaR",      f"Monte Carlo ({c}%)",    f"{m[f'var_mc_{c}']:.2%}"),
            ("VaR",      f"CVaR / ES ({c}%)",      f"{m[f'cvar_{c}']:.2%}"),
        ]
        if self.benchmark is not None:
            rows += [
                ("Market", "Beta",              f"{m['beta']:.2f}"),
                ("Market", "Alpha (ann.)",      f"{m['alpha']:.2%}"),
            ]
        df = pd.DataFrame(rows, columns=["Category", "Metric", "Value"])
        return df.set_index(["Category", "Metric"])

    def _ann_return(self):
        n = len(self.pr)
        if n == 0:
            return 0.0
        cum = (1 + self.pr).prod()
        if cum <= 0:
            return -1.0
        return float(cum ** (self.td / n) - 1)
