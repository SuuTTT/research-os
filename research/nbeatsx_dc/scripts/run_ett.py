#!/usr/bin/env python3
"""
run_ett.py
Run NBEATSx-DC on ETT datasets and compare against baselines.

Baselines:
  (A) Ridge with ALL exogenous variables
  (B) Ridge with NO exogenous variables (univariate)
  (C) NBEATSx-DC (community-based feature selection)

Usage:
  python scripts/run_ett.py --dataset ETTh1 --horizon 96 --method structural_entropy
  python scripts/run_ett.py --all
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.nbeatsx_dc import NBEATSxDC
from src.backbone import SlidingWindowRidge

DATA_DIR = os.environ.get("ETT_DATA_DIR", "/home/ubuntu/data/ETT")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
HORIZONS = [96, 192, 336, 720]

# ETT columns: HUFL HULL MUFL MULL LUFL LULL OT
# Target = OT (last column, index 6)
TARGET_COL = "OT"


def load_ett(name: str):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    # Drop date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    # Target = OT, exog = rest
    target = df[TARGET_COL].values.astype(np.float32)
    exog_cols = [c for c in df.columns if c != TARGET_COL]
    exog = df[exog_cols].values.astype(np.float32)
    return target, exog, exog_cols


def split_data(target, exog, train_ratio=0.7, val_ratio=0.1):
    T = len(target)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    return (
        target[:t1], exog[:t1],
        target[t1:t2], exog[t1:t2],
        target[t2:], exog[t2:],
    )


def normalize(train_t, train_e, *rest_chunks):
    """Z-score normalize using training statistics only."""
    mu_t, std_t = train_t.mean(), train_t.std() + 1e-8
    mu_e, std_e = train_e.mean(axis=0), train_e.std(axis=0) + 1e-8
    norm_t = lambda x: (x - mu_t) / std_t
    norm_e = lambda x: (x - mu_e) / std_e
    chunks = [norm_t(train_t), norm_e(train_e)]
    for chunk_t, chunk_e in zip(rest_chunks[0::2], rest_chunks[1::2]):
        chunks += [norm_t(chunk_t), norm_e(chunk_e)]
    return chunks


def run_baseline_all(train_t, train_e, test_t, test_e, lookback, horizon):
    """Ridge with all exogenous variables."""
    model = SlidingWindowRidge(lookback=lookback, horizon=horizon)
    train_data = np.column_stack([train_t, train_e])
    test_data = np.column_stack([test_t, test_e])
    model.fit(train_data)
    return model.evaluate(test_data)


def run_baseline_univariate(train_t, test_t, lookback, horizon):
    """Ridge with no exogenous (univariate)."""
    model = SlidingWindowRidge(lookback=lookback, horizon=horizon)
    train_data = train_t.reshape(-1, 1)
    test_data = test_t.reshape(-1, 1)
    model.fit(train_data)
    return model.evaluate(test_data)


def run_nbeatsx_dc(train_t, train_e, test_t, test_e, lookback, horizon,
                   K1, K2, theta, theta_mode, beta, community_method, verbose):
    """Full NBEATSx-DC pipeline."""
    model = NBEATSxDC(
        K1=K1, K2=K2, theta=theta, theta_mode=theta_mode, beta=beta,
        lookback=lookback, horizon=horizon,
        community_method=community_method,
        use_dtw=False,
        verbose=verbose,
    )
    model.fit(train_t, train_e)
    result = model.evaluate(test_t, test_e)
    result["selected_vars"] = model.selected_var_ids_
    result["n_selected"] = len(model.selected_var_ids_)
    result["selection_ratio"] = model.selection_summary_["selection_ratio"]
    return result, model


def run_one(dataset, horizon, K1=20, K2=3, theta=0.50, theta_mode="percentile",
            beta=0.9, community_method="structural_entropy", lookback=96, verbose=True):
    print(f"\n{'='*60}")
    print(f"  Dataset={dataset}  H={horizon}  L={lookback}")
    print(f"  K1={K1}  K2={K2}  theta={theta}({theta_mode})  method={community_method}")
    print(f"{'='*60}")

    target, exog, exog_cols = load_ett(dataset)
    n_exog = exog.shape[1]
    print(f"  Variables: target={TARGET_COL}, exog={exog_cols} (n={n_exog})")

    train_t, train_e, val_t, val_e, test_t, test_e = split_data(target, exog)
    # Normalize using training statistics
    train_t, train_e, val_t, val_e, test_t, test_e = normalize(
        train_t, train_e, val_t, val_e, test_t, test_e
    )
    print(f"  Split: train={len(train_t)}  val={len(val_t)}  test={len(test_t)}")

    results = {"dataset": dataset, "horizon": horizon, "lookback": lookback,
               "K1": K1, "K2": K2, "theta": theta, "theta_mode": theta_mode,
               "community_method": community_method}

    # Baseline A: all exog
    t0 = time.time()
    res_all = run_baseline_all(train_t, train_e, test_t, test_e, lookback, horizon)
    results["baseline_all"] = {**res_all, "time_s": round(time.time() - t0, 2)}
    print(f"  [Baseline ALL]   MSE={res_all['mse']:.4f}  MAE={res_all['mae']:.4f}")

    # Baseline B: univariate
    t0 = time.time()
    res_uni = run_baseline_univariate(train_t, test_t, lookback, horizon)
    results["baseline_univariate"] = {**res_uni, "time_s": round(time.time() - t0, 2)}
    print(f"  [Baseline UNI]   MSE={res_uni['mse']:.4f}  MAE={res_uni['mae']:.4f}")

    # NBEATSx-DC
    t0 = time.time()
    res_dc, model = run_nbeatsx_dc(
        train_t, train_e, test_t, test_e, lookback, horizon,
        K1=K1, K2=K2, theta=theta, theta_mode=theta_mode, beta=beta,
        community_method=community_method, verbose=verbose,
    )
    elapsed = time.time() - t0
    results["nbeatsx_dc"] = {**res_dc, "time_s": round(elapsed, 2)}
    delta_vs_all = res_dc["mse"] - res_all["mse"]
    delta_vs_uni = res_dc["mse"] - res_uni["mse"]
    print(f"  [NBEATSx-DC]     MSE={res_dc['mse']:.4f}  MAE={res_dc['mae']:.4f}  "
          f"selected={res_dc['n_selected']}/{n_exog}  time={elapsed:.1f}s")
    print(f"  Δ vs all-exog:   {delta_vs_all:+.4f} ({100*delta_vs_all/res_all['mse']:+.1f}%)")
    print(f"  Δ vs univariate: {delta_vs_uni:+.4f} ({100*delta_vs_uni/res_uni['mse']:+.1f}%)")
    print(f"  Selected vars:   {[exog_cols[i-1] for i in res_dc['selected_vars']]}")

    results["delta_mse_vs_all"] = round(delta_vs_all, 6)
    results["delta_pct_vs_all"] = round(100 * delta_vs_all / (res_all["mse"] + 1e-10), 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="NBEATSx-DC on ETT")
    parser.add_argument("--dataset", default="ETTh1", choices=DATASETS)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--K1", type=int, default=20)
    parser.add_argument("--K2", type=int, default=3)
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--theta_mode", default="percentile",
                        choices=["max", "percentile", "none"])
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--method", default="structural_entropy",
                        choices=["structural_entropy", "louvain"])
    parser.add_argument("--all", action="store_true",
                        help="Run all datasets × horizons")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.all:
        all_results = []
        for ds in DATASETS:
            for h in HORIZONS:
                r = run_one(ds, h, K1=args.K1, K2=args.K2, theta=args.theta,
                            theta_mode=args.theta_mode,
                            beta=args.beta, community_method=args.method,
                            lookback=args.lookback, verbose=not args.quiet)
                all_results.append(r)
        out_path = os.path.join(RESULTS_DIR, "ett_all.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        # Print summary table
        print(f"\n{'Dataset':<8} {'H':>4}  {'All-MSE':>8}  {'UNI-MSE':>8}  "
              f"{'DC-MSE':>8}  {'Δ%':>7}  {'Sel':>4}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['dataset']:<8} {r['horizon']:>4}  "
                  f"{r['baseline_all']['mse']:>8.4f}  "
                  f"{r['baseline_univariate']['mse']:>8.4f}  "
                  f"{r['nbeatsx_dc']['mse']:>8.4f}  "
                  f"{r['delta_pct_vs_all']:>+7.1f}%  "
                  f"{r['nbeatsx_dc']['n_selected']:>4}")
    else:
        r = run_one(
            args.dataset, args.horizon,
            K1=args.K1, K2=args.K2, theta=args.theta, theta_mode=args.theta_mode,
            beta=args.beta, community_method=args.method, lookback=args.lookback,
            verbose=not args.quiet,
        )
        out_path = os.path.join(RESULTS_DIR,
                                f"{args.dataset}_H{args.horizon}_{args.method}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
