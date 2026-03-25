#!/usr/bin/env python3
"""
scaling.py — Power law curve fitter + H100 extrapolator

Fits:  bpb(C) = a * C^(-β) + floor
Given 3 (compute_budget_seconds, val_bpb) points from run_experiment.sh

Usage:
  python scaling.py --results logs/exp_002_qat_20260324/scaling_results.json
  python scaling.py --budgets 60 180 300 --bpb 1.35 1.25 1.20

Prints:
  - Fitted curve parameters (a, β, floor)
  - Extrapolated bpb at H100 equivalent compute
  - Recommendation: keep or revert vs baseline curve
"""

import argparse
import json
import sys
import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:
    print("[ERROR] scipy not installed. Run: pip install scipy")
    sys.exit(1)


# Rough compute multiplier: 8xH100 for 600s vs A6000 for 300s
# 8 GPUs × ~10x faster per GPU vs A6000 × 2x more time = ~160x
# Conservative estimate: 80x (accounting for memory bandwidth differences)
H100_COMPUTE_MULTIPLIER = 80.0
H100_BUDGET_SECONDS_EQUIV = 300 * H100_COMPUTE_MULTIPLIER  # ~24000s equivalent


def power_law(C, a, beta, floor):
    return a * np.power(C, -beta) + floor


def fit_curve(budgets, bpbs):
    budgets = np.array(budgets, dtype=float)
    bpbs = np.array(bpbs, dtype=float)

    # Initial guesses
    p0 = [1.0, 0.1, 0.9]
    bounds = ([0, 0, 0.5], [100, 2.0, bpbs.min()])

    try:
        popt, pcov = curve_fit(power_law, budgets, bpbs, p0=p0, bounds=bounds, maxfev=10000)
        return popt, np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"[ERROR] Curve fitting failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", help="Path to scaling_results.json from run_experiment.sh")
    parser.add_argument("--budgets", nargs=3, type=float, help="3 budget values in seconds")
    parser.add_argument("--bpb", nargs=3, type=float, help="3 val_bpb values")
    parser.add_argument("--baseline-beta", type=float, default=None,
                        help="β from baseline curve (to compare against)")
    args = parser.parse_args()

    # Load data
    if args.results:
        with open(args.results) as f:
            data = json.load(f)
        runs = data["runs"]
        budgets = [r["budget_seconds"] for r in runs]
        bpbs = [r["val_bpb"] for r in runs]
        exp_name = data.get("experiment", "unknown")
    elif args.budgets and args.bpb:
        budgets = args.budgets
        bpbs = args.bpb
        exp_name = "manual"
    else:
        print("[ERROR] Provide --results or both --budgets and --bpb")
        sys.exit(1)

    print(f"\n{'='*54}")
    print(f"  Scaling Law Analysis — {exp_name}")
    print(f"{'='*54}")
    print(f"  Input data points:")
    for b, bpb in zip(budgets, bpbs):
        print(f"    {int(b):>5}s  →  bpb: {bpb:.4f}")

    # Fit
    (a, beta, floor), (a_err, beta_err, floor_err) = fit_curve(budgets, bpbs)

    print(f"\n  Fitted curve:  bpb(C) = {a:.4f} × C^(-{beta:.4f}) + {floor:.4f}")
    print(f"  Uncertainties: a±{a_err:.4f}  β±{beta_err:.4f}  floor±{floor_err:.4f}")

    # Extrapolate
    predicted_bpb = power_law(H100_BUDGET_SECONDS_EQUIV, a, beta, floor)
    print(f"\n  H100 extrapolation (~{H100_COMPUTE_MULTIPLIER:.0f}x compute):")
    print(f"    Predicted bpb: {predicted_bpb:.4f}")
    print(f"    Irreducible floor estimate: {floor:.4f}")

    # Interpret β
    print(f"\n  Curve steepness (β = {beta:.4f}):")
    if beta > 0.15:
        print(f"    ✅ STEEP — this config benefits strongly from more compute")
    elif beta > 0.05:
        print(f"    ⚠️  MODERATE — some benefit from more compute")
    else:
        print(f"    ❌ FLAT — diminishing returns, technique may be tapped out")

    # Compare to baseline if provided
    if args.baseline_beta is not None:
        delta = beta - args.baseline_beta
        if delta > 0.01:
            print(f"\n  vs baseline β={args.baseline_beta:.4f}: +{delta:.4f} → BETTER scaling ✅")
        elif delta < -0.01:
            print(f"\n  vs baseline β={args.baseline_beta:.4f}: {delta:.4f} → WORSE scaling ❌")
        else:
            print(f"\n  vs baseline β={args.baseline_beta:.4f}: ~equal scaling ≈")

    # Recommendation
    print(f"\n  Recommendation:")
    if predicted_bpb < min(bpbs) - 0.005:
        print(f"    ✅ KEEP — predicted H100 bpb ({predicted_bpb:.4f}) is a meaningful improvement")
    else:
        print(f"    ⚠️  MARGINAL — check if predicted bpb beats your current best")

    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
