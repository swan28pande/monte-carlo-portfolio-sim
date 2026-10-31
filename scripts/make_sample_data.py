"""Generate data/sample_portfolio.csv: ~2 years of synthetic daily prices for 5 tickers.

Correlated GBM random walk, seeded so the file is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "JPM"]
START = np.array([185.0, 375.0, 140.0, 150.0, 170.0])
ANNUAL_RET = np.array([0.18, 0.16, 0.14, 0.15, 0.10])
ANNUAL_VOL = np.array([0.27, 0.24, 0.29, 0.33, 0.22])
CORR = np.array([
    [1.00, 0.70, 0.62, 0.60, 0.35],
    [0.70, 1.00, 0.66, 0.63, 0.33],
    [0.62, 0.66, 1.00, 0.64, 0.30],
    [0.60, 0.63, 0.64, 1.00, 0.28],
    [0.35, 0.33, 0.30, 0.28, 1.00],
])

rng = np.random.default_rng(5)
dates = pd.bdate_range("2023-01-02", "2024-12-31")
sigma = ANNUAL_VOL / np.sqrt(252)
mu = ANNUAL_RET / 252 - sigma**2 / 2
z = rng.standard_normal((len(dates) - 1, len(TICKERS))) @ np.linalg.cholesky(CORR).T
log_paths = np.vstack([np.zeros(len(TICKERS)), np.cumsum(mu + sigma * z, axis=0)])
prices = START * np.exp(log_paths)

df = pd.DataFrame(prices, index=dates, columns=TICKERS).round(2)
df = df.rename_axis("Date").reset_index().melt(id_vars="Date", var_name="Ticker", value_name="AdjClose")
out = Path(__file__).resolve().parent.parent / "data" / "sample_portfolio.csv"
df.sort_values(["Date", "Ticker"]).to_csv(out, index=False, date_format="%Y-%m-%d")
print(f"Wrote {len(df)} rows to {out}")
