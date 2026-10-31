"""CUDA GBM path generation with Numba.

One GPU thread simulates one path: it draws its own normals on the device,
correlates them with the Cholesky factor, steps the path through the horizon
and writes the terminal portfolio value. Without a GPU it falls back to the
NumPy simulator.
"""
import math

import numpy as np
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64

from .simulator import model_params, simulate_cpu

MAX_ASSETS = 32  # size of per-thread local arrays (must be a compile-time constant)
THREADS_PER_BLOCK = 256


@cuda.jit
def gbm_kernel(rng_states, mu, sigma, L, weights, horizon, out):
    i = cuda.grid(1)
    if i >= out.shape[0]:
        return
    n = mu.shape[0]
    z = cuda.local.array(MAX_ASSETS, float64)
    log_s = cuda.local.array(MAX_ASSETS, float64)
    for a in range(n):
        log_s[a] = 0.0
    for _ in range(horizon):
        for a in range(n):
            z[a] = xoroshiro128p_normal_float64(rng_states, i)
        # correlated shock for asset a is row a of L times z (L is lower triangular)
        for a in range(n):
            shock = 0.0
            for b in range(a + 1):
                shock += L[a, b] * z[b]
            log_s[a] += mu[a] + sigma[a] * shock
    v = 0.0
    for a in range(n):
        v += weights[a] * math.exp(log_s[a])
    out[i] = v


def gpu_available():
    try:
        return cuda.is_available()
    except Exception:
        return False


def simulate_gpu(mean, cov, weights, num_paths=1_000_000, horizon=252, seed=42):
    """Simulate terminal portfolio values on the GPU (NumPy fallback when no GPU)."""
    if not gpu_available():
        print("No CUDA GPU found, falling back to CPU.")
        return simulate_cpu(mean, cov, weights, num_paths, horizon, seed)
    mu, sigma, L = model_params(mean, cov)
    if len(mu) > MAX_ASSETS:
        raise ValueError(f"GPU kernel supports at most {MAX_ASSETS} assets.")
    blocks = math.ceil(num_paths / THREADS_PER_BLOCK)
    rng_states = create_xoroshiro128p_states(blocks * THREADS_PER_BLOCK, seed=seed)
    out = cuda.device_array(num_paths, dtype=np.float64)
    args = [cuda.to_device(np.ascontiguousarray(x, dtype=np.float64)) for x in (mu, sigma, L, weights)]
    gbm_kernel[blocks, THREADS_PER_BLOCK](rng_states, *args, horizon, out)
    cuda.synchronize()
    return out.copy_to_host()
