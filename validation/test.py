"""
Alpha101 comparison test framework.
Compares user's alpha101.py implementation against self-contained reference.
"""

import importlib.util
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── IndNeutralize-dependent factors (SKIP) ──────────────────────────────────
REF_MISSING = {48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}

# ── Thresholds ──────────────────────────────────────────────────────────────
ATOL = 1e-6
RTOL = 1e-4
MIN_SPEARMAN = 0.98


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic data
# ═════════════════════════════════════════════════════════════════════════════

def make_market_data(n_dates=250, n_stocks=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    stocks = [f"S{i:03d}" for i in range(n_stocks)]

    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_stocks)), axis=0)),
        index=dates, columns=stocks,
    )
    open_ = close * (1 + rng.normal(0, 0.005, (n_dates, n_stocks)))
    high = close * (1 + rng.uniform(0, 0.02, (n_dates, n_stocks)))
    low = close * (1 - rng.uniform(0, 0.02, (n_dates, n_stocks)))
    volume = pd.DataFrame(
        rng.integers(1_000_000, 10_000_000, (n_dates, n_stocks)).astype(float),
        index=dates, columns=stocks,
    )
    amount = close * volume * rng.uniform(0.98, 1.02, (n_dates, n_stocks))
    vwap = amount / (volume + 1)
    returns = close.pct_change().fillna(0)

    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "vwap": vwap, "returns": returns, "amount": amount,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Reference loader
# ═════════════════════════════════════════════════════════════════════════════

def load_reference():
    spec = importlib.util.spec_from_file_location("ref_impl", Path(__file__).parent / "ref_code.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_ref_wide(ref_mod, alpha_id, data):
    alpha_obj = ref_mod.Alpha101Ref(
        open_=data["open"], high=data["high"], low=data["low"],
        close=data["close"], volume=data["volume"],
        returns=data["returns"], vwap=data["vwap"],
    )
    method = getattr(alpha_obj, f"alpha{alpha_id:03d}")
    result = method()
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, np.ndarray):
        return pd.DataFrame(result, index=data["close"].index, columns=data["close"].columns)
    raise TypeError(f"alpha{alpha_id:03d} returned {type(result)}")


# ═════════════════════════════════════════════════════════════════════════════
# User implementation loader
# ═════════════════════════════════════════════════════════════════════════════

def load_your_impl():
    spec = importlib.util.spec_from_file_location("my_alpha101", Path(__file__).parent.parent / "alpha101.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_your_alpha(your_mod, alpha_id, data):
    adv20 = data["volume"].rolling(20).mean()
    alpha_obj = your_mod.Alpha(
        open=data["open"], high=data["high"], low=data["low"],
        close=data["close"], volume=data["volume"],
        returns=data["returns"], vwap=data["vwap"], adv20=adv20,
    )
    method = getattr(alpha_obj, f"alpha{alpha_id:03d}")
    result = method()
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, np.ndarray):
        return pd.DataFrame(result, index=data["close"].index, columns=data["close"].columns)
    raise TypeError(f"alpha{alpha_id:03d} returned {type(result)}")


# ═════════════════════════════════════════════════════════════════════════════
# Comparison
# ═════════════════════════════════════════════════════════════════════════════

def compare(yours, ref, alpha_id, verbose):
    result = {
        "alpha": alpha_id, "status": "PASS", "note": "",
        "max_abs_err": np.nan, "median_spearman": np.nan,
        "nan_mismatch_pct": np.nan, "valid_cells": 0,
    }

    common_idx = yours.index.intersection(ref.index)
    common_col = yours.columns.intersection(ref.columns)
    if len(common_idx) == 0 or len(common_col) == 0:
        result["status"] = "ERROR"
        result["note"] = "Cannot align index/columns"
        return result

    y = yours.loc[common_idx, common_col].astype(float)
    r = ref.loc[common_idx, common_col].astype(float)

    nan_mismatch = (y.isna() != r.isna()).sum().sum()
    result["nan_mismatch_pct"] = round(100 * nan_mismatch / y.size, 2)

    valid_mask = ~y.isna() & ~r.isna()
    n_valid = valid_mask.sum().sum()
    result["valid_cells"] = int(n_valid)

    if n_valid == 0:
        result["status"] = "WARN"
        result["note"] = "No valid cells to compare"
        return result

    y_vals = y.values[valid_mask.values]
    r_vals = r.values[valid_mask.values]

    abs_err = np.abs(y_vals - r_vals)
    result["max_abs_err"] = float(np.max(abs_err))
    num_ok = np.allclose(y_vals, r_vals, atol=ATOL, rtol=RTOL)

    spearman_list = []
    for date in common_idx:
        y_row = y.loc[date].dropna()
        r_row = r.loc[date].dropna()
        shared = y_row.index.intersection(r_row.index)
        if len(shared) < 3:
            continue
        corr, _ = spearmanr(y_row[shared], r_row[shared])
        if not np.isnan(corr):
            spearman_list.append(corr)

    if spearman_list:
        med_spearman = float(np.median(spearman_list))
        result["median_spearman"] = round(med_spearman, 4)
        rank_ok = med_spearman >= MIN_SPEARMAN
    else:
        rank_ok = True

    if not num_ok:
        result["status"] = "FAIL"
        result["note"] = f"Numeric diff, max_abs_err={result['max_abs_err']:.2e}"
    elif not rank_ok:
        result["status"] = "WARN"
        result["note"] = f"Rank diff, median_spearman={med_spearman:.4f} < {MIN_SPEARMAN}"

    if verbose and result["status"] != "PASS":
        diff = (y - r).abs()
        diff_flat = diff.stack().dropna().sort_values(ascending=False)
        print(f"\n  [Alpha{alpha_id:03d}] Top 5 diffs:")
        for (date, col), val in diff_flat.head(5).items():
            print(f"    {date.date()} {col}: yours={y.loc[date, col]:.6f}, ref={r.loc[date, col]:.6f}, diff={val:.2e}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Main test loop
# ═════════════════════════════════════════════════════════════════════════════

def run_tests(alpha_ids, verbose):
    print("=" * 60)
    print("Alpha101 Comparison Test")
    print(f"Numeric tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"Rank tolerance: median Spearman >= {MIN_SPEARMAN}")
    print("=" * 60)

    data = make_market_data()
    print(f"Synthetic data: {len(data['close'])} days x {len(data['close'].columns)} stocks\n")

    ref_mod = load_reference()
    your_mod = load_your_impl()

    rows = []
    pass_count = warn_count = fail_count = skip_count = error_count = 0

    for alpha_id in alpha_ids:
        prefix = f"Alpha{alpha_id:03d}"

        if alpha_id in REF_MISSING:
            print(f"  {prefix}  SKIP  (IndNeutralize-dependent)")
            rows.append({"alpha": alpha_id, "status": "SKIP", "note": "IndNeutralize"})
            skip_count += 1
            continue

        try:
            yours = run_your_alpha(your_mod, alpha_id, data)
        except Exception as e:
            msg = f"Your impl error: {type(e).__name__}: {e}"
            print(f"  {prefix}  ERROR  {msg}")
            if verbose:
                traceback.print_exc()
            rows.append({"alpha": alpha_id, "status": "ERROR", "note": msg})
            error_count += 1
            continue

        try:
            ref_out = run_ref_wide(ref_mod, alpha_id, data)
        except Exception as e:
            msg = f"Ref impl error: {type(e).__name__}: {e}"
            print(f"  {prefix}  ERROR  {msg}")
            if verbose:
                traceback.print_exc()
            rows.append({"alpha": alpha_id, "status": "ERROR", "note": msg})
            error_count += 1
            continue

        res = compare(yours, ref_out, alpha_id, verbose)
        rows.append(res)

        status = res["status"]
        spearman = f"spearman={res['median_spearman']:.4f}" if not np.isnan(res["median_spearman"]) else ""
        max_err = f"max_err={res['max_abs_err']:.2e}" if not np.isnan(res["max_abs_err"]) else ""
        detail = "  ".join(filter(None, [spearman, max_err, res["note"]]))
        print(f"  {prefix}  {status:<5}  {detail}")

        if status == "PASS":
            pass_count += 1
        elif status == "WARN":
            warn_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            error_count += 1

    total = len(alpha_ids)
    print("\n" + "=" * 60)
    print(f"Results ({total} factors)")
    print(f"  PASS   {pass_count}")
    print(f"  WARN   {warn_count}   (rank diff, numeric may be OK)")
    print(f"  FAIL   {fail_count}   (numeric diff)")
    print(f"  ERROR  {error_count}   (runtime error)")
    print(f"  SKIP   {skip_count}   (IndNeutralize)")
    print("=" * 60)

    report_path = Path(__file__).parent / "alpha101_test_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\nReport saved to: {report_path}")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpha101 comparison test")
    parser.add_argument("--alpha", nargs="+", type=int, default=list(range(1, 102)),
                        help="Factor IDs to test (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print diff samples")
    args = parser.parse_args()
    run_tests(sorted(set(args.alpha)), args.verbose)
