"""src/data_loader.py — Downloads and caches stock price data."""

import hashlib
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data"


class DataLoader:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_prices(self, tickers, start, end, use_cache=True):
        cache_path = self._cache_path(tickers, start, end)
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        frames = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                raise ValueError(f"No data for {ticker}")
            frames[ticker] = df["Close"]

        prices = pd.DataFrame(frames)
        prices.index = pd.to_datetime(prices.index)
        prices.dropna(how="all", inplace=True)
        if use_cache:
            prices.to_parquet(cache_path)
        return prices

    def get_returns(self, tickers, start, end, use_cache=True):
        prices = self.get_prices(tickers, start, end, use_cache)
        return prices.pct_change().dropna()

    def get_benchmark(self, ticker, start, end):
        prices = self.get_prices([ticker], start, end)
        return prices[ticker].pct_change().dropna()

    def _cache_path(self, tickers, start, end):
        key = "_".join(sorted(tickers)) + f"_{start}_{end}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"prices_{h}.parquet"
