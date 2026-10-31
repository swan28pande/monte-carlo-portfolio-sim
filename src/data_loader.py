"""Load historical prices and turn them into simulation inputs."""
import numpy as np
import pandas as pd


def load_prices(path, start=None, end=None):
    """Read a long-format CSV (Date, Ticker, AdjClose) into a Date x Ticker price table."""
    df = pd.read_csv(path, parse_dates=["Date"])
    if start:
        df = df[df["Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Date"] <= pd.Timestamp(end)]
    prices = df.pivot(index="Date", columns="Ticker", values="AdjClose").sort_index()
    return prices.dropna()


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def load_inputs(path, start=None, end=None):
    """Return (tickers, mean daily log returns, daily covariance matrix) as NumPy arrays."""
    rets = log_returns(load_prices(path, start, end))
    if len(rets) < 2:
        raise ValueError("Need at least 3 price rows in the selected date range.")
    return list(rets.columns), rets.mean().to_numpy(), rets.cov().to_numpy()
