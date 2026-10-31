"""Monte Carlo portfolio risk simulator.

    python cli.py --input data/sample_portfolio.csv --num-paths 1000000 --use-gpu
"""
import argparse
import time

import numpy as np

from src.data_loader import load_inputs
from src.simulator import report, simulate_cpu


def main():
    p = argparse.ArgumentParser(description="GPU-accelerated Monte Carlo portfolio VaR / CVaR.")
    p.add_argument("--input", default="data/sample_portfolio.csv", help="historical prices CSV (Date, Ticker, AdjClose)")
    p.add_argument("--num-paths", type=int, default=1_000_000)
    p.add_argument("--horizon", type=int, default=252, help="trading days")
    p.add_argument("--confidence", type=float, default=0.95, help="VaR confidence level")
    p.add_argument("--use-gpu", action="store_true", help="run the CUDA kernel and compare against the CPU baseline")
    p.add_argument("--start", help="first date to use, e.g. 2023-01-01")
    p.add_argument("--end", help="last date to use")
    p.add_argument("--weights", help="comma-separated portfolio weights (default: equal)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", metavar="PNG", help="save a histogram of simulated portfolio returns")
    args = p.parse_args()

    tickers, mean, cov = load_inputs(args.input, args.start, args.end)
    w = np.array([float(x) for x in args.weights.split(",")]) if args.weights else np.ones(len(tickers))
    if len(w) != len(tickers):
        p.error(f"--weights needs {len(tickers)} values ({', '.join(tickers)})")
    w = w / w.sum()

    print(f"Assets:  {', '.join(f'{t} {x:.0%}' for t, x in zip(tickers, w))}")
    print(f"Paths:   {args.num_paths:,}   Horizon: {args.horizon} trading days")

    t0 = time.perf_counter()
    values = simulate_cpu(mean, cov, w, args.num_paths, args.horizon, args.seed)
    cpu_time = time.perf_counter() - t0
    print(f"CPU (NumPy) time: {cpu_time:.2f} s")

    if args.use_gpu:
        from src.cuda_kernels import gpu_available, simulate_gpu

        if gpu_available():
            simulate_gpu(mean, cov, w, 1024, args.horizon, args.seed)  # warm-up: JIT compile
            t0 = time.perf_counter()
            values = simulate_gpu(mean, cov, w, args.num_paths, args.horizon, args.seed)
            gpu_time = time.perf_counter() - t0
            print(f"GPU (CUDA) time:  {gpu_time:.2f} s   speedup: {cpu_time / gpu_time:.1f}x")
        else:
            print("No CUDA GPU found, reporting CPU results.")

    print(f"\nExpected {args.horizon}-day return: {values.mean() - 1:+.2%}")
    print(f"{'Confidence':>10} {'VaR':>8} {'CVaR':>8}")
    for c, m in report(values, (0.95, 0.99, args.confidence)).items():
        print(f"{c:>10.1%} {m['VaR']:>8.2%} {m['CVaR']:>8.2%}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = values - 1
        var = report(values, (args.confidence,))[args.confidence]["VaR"]
        plt.hist(r, bins=200, color="steelblue")
        plt.axvline(-var, color="crimson", label=f"VaR {args.confidence:.0%}: {var:.1%}")
        plt.xlabel(f"{args.horizon}-day portfolio return")
        plt.ylabel("paths")
        plt.legend()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"Saved histogram to {args.plot}")


if __name__ == "__main__":
    main()
