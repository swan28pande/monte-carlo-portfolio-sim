"""Monte Carlo portfolio risk engine (NumPy CPU baseline).

Each asset follows Geometric Brownian Motion. Daily log returns are

    r_t = mu + sigma * (L @ z_t),   z_t ~ N(0, I)

where mu is the mean daily log return, sigma the daily volatility and L the
Cholesky factor of the correlation matrix, so the shocks keep the historical
correlation. (mu already equals the GBM drift m - sigma^2 / 2, because it is
estimated from log returns.)

The portfolio starts at value 1, so its simulated loss is 1 - V_T.
"""
import numpy as np


def model_params(mean, cov):
    """Split the covariance matrix into volatilities and the Cholesky factor of the correlation."""
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    return mean, sigma, np.linalg.cholesky(corr)


def simulate_cpu(mean, cov, weights, num_paths=1_000_000, horizon=252, seed=42, chunk=200_000):
    """Simulate terminal portfolio values with NumPy, stepping every path day by day.

    Paths run in chunks so memory stays bounded at 1M+ paths.
    """
    mu, sigma, L = model_params(mean, cov)
    rng = np.random.default_rng(seed)
    n = len(mu)
    out = np.empty(num_paths)
    for start in range(0, num_paths, chunk):
        m = min(chunk, num_paths - start)
        log_s = np.zeros((m, n))
        for _ in range(horizon):
            z = rng.standard_normal((m, n)) @ L.T  # correlated shocks
            log_s += mu + sigma * z
        out[start:start + m] = np.exp(log_s) @ weights
    return out


def risk_metrics(values, confidence=0.95):
    """VaR and CVaR of the loss (1 - V_T), as fractions of the starting value."""
    losses = 1.0 - values
    var = np.quantile(losses, confidence)
    cvar = losses[losses >= var].mean()
    return {"VaR": float(var), "CVaR": float(cvar)}


def report(values, levels=(0.95, 0.99)):
    return {c: risk_metrics(values, c) for c in sorted(set(levels))}
