# Monte Carlo Portfolio Risk Simulator

GPU-accelerated Monte Carlo engine for portfolio **Value at Risk (VaR)** and **Conditional VaR (CVaR)**. It fits a correlated Geometric Brownian Motion model to historical prices, simulates 1M+ price paths, and reports tail-risk metrics. A NumPy CPU baseline sits next to a Numba CUDA kernel that runs one GPU thread per path.

## Overview

1. **Load** historical adjusted closes (`Date, Ticker, AdjClose`) with Pandas and compute daily log returns.
2. **Estimate** the mean return vector and covariance matrix, then split the covariance into volatilities and a correlation matrix.
3. **Correlate** shocks with the Cholesky factor `L` of the correlation matrix (`corr = L Lᵀ`).
4. **Simulate** each asset day by day under GBM:

   ```
   log S(t+1) = log S(t) + μ + σ ⊙ (L z),   z ~ N(0, I)
   ```

   `μ` is the mean daily log return, which already equals the GBM drift `m − σ²/2`.
5. **Measure risk** on the portfolio loss `1 − V_T` (starting value 1):
   - `VaR_α`: the α-quantile of the loss
   - `CVaR_α`: the average loss beyond `VaR_α` (expected shortfall)

## Architecture

```
monte-carlo-portfolio-sim/
├── cli.py                   # argparse entry point, timing, speedup, optional histogram
├── src/
│   ├── data_loader.py       # CSV -> log returns -> mean vector, covariance (NumPy)
│   ├── simulator.py         # Cholesky + GBM CPU baseline, VaR / CVaR
│   └── cuda_kernels.py      # Numba @cuda.jit kernel, CPU fallback
├── scripts/make_sample_data.py
└── data/sample_portfolio.csv  # 2 years of synthetic daily prices, 5 tickers
```

**CPU baseline** (`simulator.py`): vectorized NumPy. It steps 200k-path chunks through the horizon, so memory stays flat at any path count.

**CUDA kernel** (`cuda_kernels.py`): each thread owns one path. It keeps its state in registers or local memory, and per day it:
- draws normals on the device (`xoroshiro128p`),
- applies the lower-triangular Cholesky product in place,
- accumulates log prices.

At the end it writes a single terminal portfolio value, so the only memory traffic is one float per path. If no CUDA device is found, it falls back to the CPU engine.

## Usage

```bash
pip install -r requirements.txt

# CPU baseline, 1M paths, 1-year horizon
python cli.py --input data/sample_portfolio.csv

# GPU run with speedup vs CPU
python cli.py --input data/sample_portfolio.csv --num-paths 1000000 --use-gpu

# Custom weights, 97.5% VaR, date filter, return histogram
python cli.py --weights 3,2,2,2,1 --confidence 0.975 --start 2024-01-01 --plot returns.png
```

| Flag | Default | Description |
|---|---|---|
| `--input` | `data/sample_portfolio.csv` | Historical prices CSV (`Date, Ticker, AdjClose`) |
| `--num-paths` | `1000000` | Number of Monte Carlo paths |
| `--horizon` | `252` | Horizon in trading days |
| `--confidence` | `0.95` | VaR / CVaR confidence level (95% and 99% are always shown) |
| `--use-gpu` | off | Run the CUDA kernel and report the speedup vs CPU |
| `--weights` | equal | Comma-separated weights, in ticker order |
| `--start`, `--end` | all | Date range for the historical window |
| `--plot` | — | Save a histogram of simulated returns |

To regenerate the sample data: `python scripts/make_sample_data.py` (seeded and reproducible).

## Sample output

```
Assets:  AAPL 20%, AMZN 20%, GOOG 20%, JPM 20%, MSFT 20%
Paths:   1,000,000   Horizon: 252 trading days
CPU (NumPy) time: 30.20 s

Expected 252-day return: +17.09%
Confidence      VaR     CVaR
     95.0%   20.35%   27.00%
     99.0%   31.25%   36.00%
```

At 95% confidence, the equal-weight portfolio should not lose more than 20.4% over one year. In the worst 5% of scenarios, the average loss is 27.0%.

## Performance benchmarks

Workload: 5 assets, 252 daily steps per path (1M paths ≈ 1.26B correlated normal draws).

| Paths | Backend | Hardware | Time |
|---|---|---|---|
| 100,000 | CPU (NumPy) | Apple M1 | 2.75 s |
| 1,000,000 | CPU (NumPy) | Apple M1 | 30.20 s |
| 1,000,000 | GPU (Numba CUDA) | _NVIDIA GPU_ | _run `python cli.py --use-gpu`_ |

The GPU row needs a CUDA machine. A free Colab T4 works:

```python
!git clone https://github.com/swan28pande/monte-carlo-portfolio-sim
%cd monte-carlo-portfolio-sim
!pip install -q -r requirements.txt
!python cli.py --use-gpu
```

The CLI times the kernel after a warm-up launch, so JIT compile time is excluded.

## Notes and limitations

- GBM assumes normal log returns with constant volatility and correlation. Real returns have fat tails and volatility clustering.
- Mean returns estimated from 2 years of data are noisy, so most of the VaR signal comes from the covariance.
- The sample data is synthetic: a seeded correlated random walk, not real market prices.
