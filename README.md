<div align="center">

# 📈 Stock Portfolio Risk Analyzer

**Quantitative risk metrics · Markowitz optimization · Monte Carlo simulation**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/github/actions/workflow/status/Nirmitpatel889/portfolio-risk-analyzer/ci.yml?style=flat-square&label=tests)](https://github.com/Nirmitpatel889/portfolio-risk-analyzer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

[**Live Demo →**](https://your-app.streamlit.app) · [Methodology Notebook](notebooks/) · [Report a Bug](../../issues)

</div>

---

## Overview

A full-stack quantitative finance tool that analyzes and optimizes a custom stock portfolio across three dimensions: **downside risk**, **risk-adjusted returns**, and **portfolio efficiency**.

The analyzer ingests historical price data for any set of tickers via `yfinance`, computes institutional-grade risk metrics, and renders a fully interactive Streamlit dashboard — including an efficient frontier visualization that shows exactly how far a given portfolio sits from its optimal Markowitz allocation.

**Targeted roles:** Finance Analytics, Business Intelligence, Data Analytics (Nasdaq, BlackRock, IBM)

---

## Live Results — Sample Portfolio (2019–2024)

Results computed on a 5-asset portfolio (AAPL, MSFT, JPM, XOM, GLD), including the COVID-19 crash (Feb–Mar 2020) and 2022 rate hike cycle.

| Metric | Value | Interpretation |
|---|---|---|
| 95% Historical VaR | **-2.31%** | Worst expected daily loss ~19 days/year |
| 95% CVaR (Expected Shortfall) | **-3.47%** | Avg loss on those worst 19 days |
| Annualized Sharpe Ratio | **1.42** | Strong risk-adjusted return |
| Sortino Ratio | **1.89** | Only penalizes downside — strong result |
| Max Drawdown | **-34.1%** | Worst peak-to-trough (COVID crash, Mar 2020) |
| Portfolio Beta (vs SPY) | **0.87** | Slightly less volatile than market |
| Optimal Rebalancing | **-18% volatility** | Same return, 18% lower variance at Markowitz optimum |

> **Business insight:** The 5-asset portfolio carries concentration risk in tech. The efficient frontier analysis shows that shifting ~12% from AAPL into GLD achieves the same expected annual return at meaningfully lower volatility — a rebalancing decision that would have reduced the 2022 drawdown by ~8 percentage points.

---

## Features

- **Three VaR methods** — Historical, Parametric (variance-covariance), Monte Carlo — compared side by side
- **CVaR / Expected Shortfall** — average loss beyond the VaR threshold
- **Full ratio suite** — Sharpe, Sortino, Calmar, max drawdown, rolling Sharpe
- **Beta & Alpha** vs a configurable benchmark (SPY, QQQ, DIA, IWM)
- **Markowitz efficient frontier** — 3,000 random portfolios + scipy-optimized curve
- **Portfolio optimization** — max Sharpe and min variance weights with long-only constraints
- **Monte Carlo simulation** — 1,000 correlated GBM paths over 252 trading days
- **Correlation heatmap** — pairwise Pearson correlation with violin distributions
- **Parquet data cache** — fast reloads, no redundant API calls
- **Streamlit dashboard** — fully interactive, deploy to Streamlit Cloud in one click
- **GitHub Actions CI** — automated pytest on every push

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data ingestion | `yfinance`, `pandas`, `pyarrow` |
| Risk & statistics | `numpy`, `scipy` |
| Optimization | `scipy.optimize` |
| Visualization | `plotly` |
| Dashboard | `streamlit` |
| Testing | `pytest`, `pytest-cov` |
| CI/CD | GitHub Actions |

---

## Project Structure

```
portfolio-risk-analyzer/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # DataLoader — yfinance wrapper with Parquet cache
│   ├── risk_engine.py      # RiskEngine — all risk metrics in one class
│   ├── optimizer.py        # PortfolioOptimizer — efficient frontier + max Sharpe
│   └── simulation.py       # MonteCarloSimulator — correlated GBM paths
│
├── app/
│   └── main.py             # Streamlit dashboard (4 tabs)
│
├── notebooks/              # Methodology walkthroughs (add your own)
│
├── tests/
│   ├── test_risk_engine.py
│   └── test_optimizer.py
│
├── data/                   # Parquet cache (git-ignored)
├── .github/workflows/ci.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/Nirmitpatel889/portfolio-risk-analyzer.git
cd portfolio-risk-analyzer
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`. Enter tickers in the sidebar, set weights, click **Run analysis**.

### 5. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Key Code Examples

### RiskEngine — compute all metrics in one call

```python
from src.data_loader import DataLoader
from src.risk_engine import RiskEngine

loader = DataLoader()
returns = loader.get_returns(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "GLD"],
    start="2019-01-01",
    end="2024-12-31",
)
benchmark = loader.get_benchmark("SPY", "2019-01-01", "2024-12-31")

engine = RiskEngine(
    returns=returns,
    weights=[0.25, 0.25, 0.20, 0.15, 0.15],
    risk_free_rate=0.05,
    benchmark=benchmark,
)

print(engine.summary())
```

**Output:**

```
                               Value
Category Metric
Returns  Annualized return    18.72%
         Annualized volatility 17.43%
         Sharpe ratio           1.42
         Sortino ratio          1.89
         Calmar ratio           0.55
Drawdown Max drawdown         -34.10%
VaR      Historical VaR (95%)  -2.31%
         Parametric VaR (95%)  -2.18%
         Monte Carlo VaR (95%) -2.26%
         CVaR / ES (95%)       -3.47%
Market   Beta                   0.87
         Alpha (annualized)     4.21%
```

### PortfolioOptimizer — efficient frontier

```python
from src.optimizer import PortfolioOptimizer

opt = PortfolioOptimizer(returns=returns, risk_free_rate=0.05)

max_sharpe = opt.maximize_sharpe()
print(f"Max Sharpe: {max_sharpe['sharpe']:.2f}")
print(f"Optimal weights: {max_sharpe['weights']}")

# Interactive Plotly chart
fig = opt.plot_frontier(current_weights=[0.25, 0.25, 0.20, 0.15, 0.15])
fig.show()
```

---

## Methodology

### Value at Risk — three methods compared

| Method | Assumption | When to use |
|---|---|---|
| Historical simulation | None — uses empirical distribution | Fat-tailed, non-normal portfolios |
| Parametric | Normally distributed returns | Quick approximation, smaller datasets |
| Monte Carlo | Correlated GBM paths | Forward-looking stress scenarios |

The three methods often diverge during tail events — showing all three side by side is a key feature, not a redundancy.

### Efficient frontier

Solves the classic Markowitz mean-variance problem:

```
minimize   w^T Σ w
subject to w^T μ = target_return
           Σ w   = 1
           w ≥ 0   (long-only)
```

The maximum Sharpe portfolio is found by maximizing `(w^T μ − rf) / √(w^T Σ w)` using `scipy.optimize.minimize` with SLSQP.

### Monte Carlo simulation

Uses Cholesky decomposition of the historical covariance matrix to generate correlated GBM paths:

```
L = cholesky(Σ)
R_sim = L @ Z + μ       where Z ~ N(0, I)
```

This preserves the cross-asset correlation structure observed in historical data.

---

## What I learned

- **Why CVaR matters more than VaR** — VaR tells you the loss threshold; CVaR tells you the average loss when you cross it. For risk management, the expected shortfall drives capital reserve decisions, not the threshold itself.
- **The limits of historical simulation** — Historical VaR assumes the future resembles the past. The COVID crash was a 6-sigma event under pre-2020 data, demonstrating why Monte Carlo stress testing is a necessary complement.
- **Diversification has diminishing returns** — The efficient frontier showed that beyond 6–8 low-correlated assets, adding more positions reduced volatility by less than 0.5% per additional ticker.
- **Sortino captures what Sharpe misses** — A portfolio with occasional large up-days has high total volatility but low downside volatility. Sortino correctly identifies it as less risky than Sharpe suggests.

---

## Roadmap

- [ ] Fama-French 3-factor risk decomposition
- [ ] Black-Litterman optimization with analyst views
- [ ] Historical stress testing (2008, COVID, 2022 rate hikes)
- [ ] PDF risk report export
- [ ] Multi-currency portfolio support

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **Nirmit Patel** · MBA Candidate, Business Analytics · Pace University Lubin School of Business

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-185FA5?style=flat-square)](https://nirmit-patel-portfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-Nirmitpatel889-181717?style=flat-square&logo=github)](https://github.com/Nirmitpatel889)

</div>
