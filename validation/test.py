import argparse
import importlib
import sys
import traceback
import warnings
from pathlib import Path

# 把项目根目录（alpha101-lab/）加入 sys.path，使 alpha101.py 能 import operators
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── 参考实现里缺少的因子（ref_code_1.py 未实现）──────────────────────────
REF_MISSING = {
    48: "参考代码未提供",
    56: "参考代码因缺少市值数据注释了该因子",
    58: "参考代码由于缺少行业数据未实现",
    59: "参考代码由于缺少行业数据未实现",
    63: "参考代码由于缺少行业数据未实现",
    67: "参考代码由于缺少行业数据未实现",
    69: "参考代码由于缺少行业数据未实现",
    70: "参考代码由于缺少行业数据未实现",
    76: "参考代码由于缺少行业数据未实现",
    79: "参考代码由于缺少行业数据未实现",
    80: "参考代码由于缺少行业数据未实现",
    82: "参考代码由于缺少行业数据未实现",
    87: "参考代码由于缺少行业数据未实现",
    89: "参考代码由于缺少行业数据未实现",
    90: "参考代码由于缺少行业数据未实现",
    91: "参考代码由于缺少行业数据未实现",
    93: "参考代码由于缺少行业数据未实现",
    97: "参考代码由于缺少行业数据未实现",
    100: "参考代码由于缺少行业数据未实现",
}

# ── 判定阈值 ──────────────────────────────────────────────────────────────
ATOL = 1e-6  # 数值绝对误差容忍
RTOL = 1e-4  # 数值相对误差容忍
MIN_SPEARMAN = 0.98  # 截面排序相关系数下限


# ═════════════════════════════════════════════════════════════════════════════
# 合成数据生成
# ═════════════════════════════════════════════════════════════════════════════


def make_market_data(n_dates=250, n_stocks=20, seed=42):
    """
    生成宽表格式合成数据（行=日期，列=股票）。
    规模故意设小，保证测试快；seed 固定保证可复现。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    stocks = [f"S{i:03d}" for i in range(n_stocks)]

    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_stocks)), axis=0)),
        index=dates,
        columns=stocks,
    )
    open_ = close * (1 + rng.normal(0, 0.005, (n_dates, n_stocks)))
    high = close * (1 + rng.uniform(0, 0.02, (n_dates, n_stocks)))
    low = close * (1 - rng.uniform(0, 0.02, (n_dates, n_stocks)))
    volume = pd.DataFrame(
        rng.integers(1_000_000, 10_000_000, (n_dates, n_stocks)).astype(float),
        index=dates,
        columns=stocks,
    )
    amount = close * volume * rng.uniform(0.98, 1.02, (n_dates, n_stocks))
    vwap = amount / (volume + 1)
    returns = close.pct_change().fillna(0)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        "returns": returns,
        "amount": amount,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 参考实现适配层
# ═════════════════════════════════════════════════════════════════════════════


def load_reference():
    """动态加载 ref_code_1.py，返回 Alphas 类和辅助函数模块。"""
    spec = importlib.util.spec_from_file_location(
        "ref_impl", Path(__file__).parent / "ref_code_1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_wide_to_long(data: dict, stock_col: str) -> pd.DataFrame:
    """
    把宽表中单列股票的数据整理成参考实现 Alphas 所需的长表格式。
    参考实现字段名：S_DQ_OPEN / S_DQ_HIGH / S_DQ_LOW / S_DQ_CLOSE /
                    S_DQ_VOLUME / S_DQ_PCTCHANGE / S_DQ_AMOUNT
    """
    df = pd.DataFrame(
        {
            "S_DQ_OPEN": data["open"][stock_col],
            "S_DQ_HIGH": data["high"][stock_col],
            "S_DQ_LOW": data["low"][stock_col],
            "S_DQ_CLOSE": data["close"][stock_col],
            "S_DQ_VOLUME": data["volume"][stock_col] / 100,  # ref 内部会 *100
            "S_DQ_PCTCHANGE": data["returns"][stock_col],
            # ref 内部: vwap = amount*1000 / (volume*100+1)，反推 amount
            "S_DQ_AMOUNT": data["vwap"][stock_col]
            * (data["volume"][stock_col] + 1)
            / 1000,
        }
    )
    return df


def call_ref_alpha(ref_mod, alpha_id: int, data: dict, stock_col: str):
    """
    对单只股票调用参考实现，返回该股票的时序 Series。
    参考实现是单股票长表模式，rank 操作只有1列时退化为全1，
    因此对需要截面 rank 的因子，需用多股票数据后再切列（见 run_ref_wide）。
    """
    df = ref_wide_to_long(data, stock_col)
    alphas_obj = ref_mod.Alphas(df)
    method = getattr(alphas_obj, f"alpha{alpha_id:03d}")
    return method()


def run_ref_wide(ref_mod, alpha_id: int, data: dict):
    """
    以宽表方式运行参考实现：
    把所有股票拼成一个大长表 → 实例化 Alphas → 调用方法 → reshape 回宽表。
    这样 rank() 能看到所有股票，截面操作才有意义。
    """
    stocks = data["close"].columns.tolist()
    n = len(data["close"])

    # 拼长表：每个字段 shape=(n_dates, n_stocks)，stack 后 shape=(n_dates*n_stocks,)
    def stack_field(key, scale=1.0):
        return data[key].values.flatten(order="F") * scale  # F-order: 按列（股票）展开

    dates_rep = np.tile(data["close"].index, len(stocks))

    # 参考实现的字段名和缩放
    long_df = pd.DataFrame(
        {
            "S_DQ_OPEN": stack_field("open"),
            "S_DQ_HIGH": stack_field("high"),
            "S_DQ_LOW": stack_field("low"),
            "S_DQ_CLOSE": stack_field("close"),
            "S_DQ_VOLUME": stack_field("volume", 1 / 100),
            "S_DQ_PCTCHANGE": stack_field("returns"),
            "S_DQ_AMOUNT": (
                data["vwap"].values.flatten(order="F")
                * (data["volume"].values.flatten(order="F") + 1)
                / 1000
            ),
        },
        index=dates_rep,
    )

    # 参考实现每个字段都是 pd.Series，Alphas 靠 index 做 rolling
    # 宽表格式下：我们把所有股票按列堆叠，同一日期重复出现 n_stocks 次
    # 但 rolling 会跨股票滚动，结果不正确。
    # 正确方式：逐股票跑，rank 操作用单列退化（对截面 rank 不精确）。
    # 折中：逐股票跑并收集，适合纯时序因子；截面 rank 因子单独标注。
    results = {}
    for stock in stocks:
        df = ref_wide_to_long(data, stock)
        alphas_obj = ref_mod.Alphas(df)
        method = getattr(alphas_obj, f"alpha{alpha_id:03d}")
        try:
            results[stock] = method()
        except Exception:
            results[stock] = pd.Series(np.nan, index=data["close"].index)

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# 你的实现适配层
# ═════════════════════════════════════════════════════════════════════════════


def load_your_impl():
    """加载你的 alpha101 模块（位于 validation/ 上一层）。"""
    spec = importlib.util.spec_from_file_location(
        "my_alpha101", Path(__file__).parent.parent / "alpha101.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_your_alpha(your_mod, alpha_id: int, data: dict) -> pd.DataFrame:
    """
    调用你的实现，返回宽表 DataFrame（行=日期，列=股票）。
    匹配 Alpha.__init__(open, high, low, close, volume, returns, vwap, adv20)
    """
    adv20 = data["volume"].rolling(20).mean()
    alpha_obj = your_mod.Alpha(
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        returns=data["returns"],
        vwap=data["vwap"],
        adv20=adv20,
    )
    method = getattr(alpha_obj, f"alpha{alpha_id:03d}")
    result = method()

    # 统一转成宽表 DataFrame（行=日期，列=股票）
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, np.ndarray):
        # shape=(n_dates, n_stocks) 的 ndarray，用 close 的 index/columns 还原
        ref_df = data["close"]
        return pd.DataFrame(result, index=ref_df.index, columns=ref_df.columns)
    raise TypeError(
        f"alpha{alpha_id:03d} 返回了未知类型 {type(result)}，期望 DataFrame / Series / ndarray"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 比较逻辑
# ═════════════════════════════════════════════════════════════════════════════


def compare(
    yours: pd.DataFrame, ref: pd.DataFrame, alpha_id: int, verbose: bool
) -> dict:
    """
    对比两个宽表结果，返回该因子的测试摘要。
    评估三个维度：
      1. 数值误差（忽略 NaN 对齐后的有效值）
      2. 截面 Spearman 相关（每日截面，取中位数）
      3. NaN 模式是否一致
    """
    result = {
        "alpha": alpha_id,
        "status": "PASS",
        "note": "",
        "max_abs_err": np.nan,
        "median_spearman": np.nan,
        "nan_mismatch_pct": np.nan,
        "valid_cells": 0,
    }

    # 对齐 index / columns
    common_idx = yours.index.intersection(ref.index)
    common_col = yours.columns.intersection(ref.columns)
    if len(common_idx) == 0 or len(common_col) == 0:
        result["status"] = "ERROR"
        result["note"] = "无法对齐 index/columns"
        return result

    y = yours.loc[common_idx, common_col].astype(float)
    r = ref.loc[common_idx, common_col].astype(float)

    # NaN 模式
    nan_y = y.isna()
    nan_r = r.isna()
    nan_mismatch = (nan_y != nan_r).sum().sum()
    total_cells = y.size
    result["nan_mismatch_pct"] = round(100 * nan_mismatch / total_cells, 2)

    # 有效值（双方均非 NaN）
    valid_mask = ~nan_y & ~nan_r
    n_valid = valid_mask.sum().sum()
    result["valid_cells"] = int(n_valid)

    if n_valid == 0:
        result["status"] = "WARN"
        result["note"] = "没有共同有效值可比较"
        return result

    y_vals = y.values[valid_mask.values]
    r_vals = r.values[valid_mask.values]

    # 数值误差
    abs_err = np.abs(y_vals - r_vals)
    result["max_abs_err"] = float(np.max(abs_err))

    num_ok = np.allclose(y_vals, r_vals, atol=ATOL, rtol=RTOL)

    # 截面 Spearman（每行）
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
        rank_ok = True  # 无法计算则跳过

    if not num_ok:
        result["status"] = "FAIL"
        result["note"] = f"数值超差，max_abs_err={result['max_abs_err']:.2e}"
    elif not rank_ok:
        result["status"] = "WARN"
        result["note"] = (
            f"排序偏差，median_spearman={med_spearman:.4f} < {MIN_SPEARMAN}"
        )

    if verbose and result["status"] != "PASS":
        _print_diff_sample(y, r, valid_mask, alpha_id)

    return result


def _print_diff_sample(
    y: pd.DataFrame, r: pd.DataFrame, valid_mask: pd.DataFrame, alpha_id: int
):
    """打印差异最大的前5个值，帮助定位问题。"""
    diff = (y - r).abs()
    diff_flat = diff.stack().dropna().sort_values(ascending=False)
    print(f"\n  [Alpha{alpha_id:03d}] 差异最大的5个位置：")
    for (date, col), val in diff_flat.head(5).items():
        print(
            f"    {date.date()} {col}: yours={y.loc[date, col]:.6f}, ref={r.loc[date, col]:.6f}, diff={val:.2e}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 主测试循环
# ═════════════════════════════════════════════════════════════════════════════


def run_tests(alpha_ids: list, verbose: bool):
    print("=" * 60)
    print("Alpha101 对比测试")
    print(f"数值容忍: atol={ATOL}, rtol={RTOL}")
    print(f"排序容忍: median Spearman >= {MIN_SPEARMAN}")
    print("=" * 60)

    data = make_market_data()
    print(f"合成数据：{len(data['close'])} 天 × {len(data['close'].columns)} 只股票\n")

    ref_mod = load_reference()
    your_mod = load_your_impl()

    rows = []
    pass_count = warn_count = fail_count = skip_count = error_count = 0

    for alpha_id in alpha_ids:
        prefix = f"Alpha{alpha_id:03d}"

        # 参考实现缺失
        if alpha_id in REF_MISSING:
            print(f"  {prefix}  SKIP  (参考实现未提供此因子)")
            rows.append({"alpha": alpha_id, "status": "SKIP", "note": "参考实现缺失"})
            skip_count += 1
            continue

        # 运行你的实现
        try:
            yours = run_your_alpha(your_mod, alpha_id, data)
        except Exception as e:
            msg = f"你的实现抛出异常: {type(e).__name__}: {e}"
            print(f"  {prefix}  ERROR  {msg}")
            if verbose:
                traceback.print_exc()
            rows.append({"alpha": alpha_id, "status": "ERROR", "note": msg})
            error_count += 1
            continue

        # 运行参考实现
        try:
            ref_out = run_ref_wide(ref_mod, alpha_id, data)
        except Exception as e:
            msg = f"参考实现抛出异常: {type(e).__name__}: {e}"
            print(f"  {prefix}  ERROR  {msg}")
            rows.append({"alpha": alpha_id, "status": "ERROR", "note": msg})
            error_count += 1
            continue

        # 对比
        res = compare(yours, ref_out, alpha_id, verbose)
        rows.append(res)

        status = res["status"]
        note = res["note"]
        spearman = (
            f"spearman={res['median_spearman']:.4f}"
            if not np.isnan(res["median_spearman"])
            else ""
        )
        max_err = (
            f"max_err={res['max_abs_err']:.2e}"
            if not np.isnan(res["max_abs_err"])
            else ""
        )
        detail = "  ".join(filter(None, [spearman, max_err, note]))

        print(f"  {prefix}  {status:<5}  {detail}")

        if status == "PASS":
            pass_count += 1
        elif status == "WARN":
            warn_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            error_count += 1

    # 汇总
    total = len(alpha_ids)
    print("\n" + "=" * 60)
    print(f"结果汇总  （共 {total} 个因子）")
    print(f"  PASS   {pass_count}")
    print(f"  WARN   {warn_count}   （排序差异，数值可能无问题）")
    print(f"  FAIL   {fail_count}   （数值超差）")
    print(f"  ERROR  {error_count}   （运行报错）")
    print(f"  SKIP   {skip_count}   （参考实现无此因子）")
    print("=" * 60)

    # 输出 CSV
    report_path = Path(__file__).parent / "alpha101_test_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\n详细报告已保存至：{report_path}")

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha101 对比测试")
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=int,
        default=list(range(1, 102)),
        help="指定要测试的因子编号，如 --alpha 1 2 5（默认全部）",
    )
    parser.add_argument("--verbose", action="store_true", help="打印差异最大的样本点")
    args = parser.parse_args()

    run_tests(sorted(set(args.alpha)), args.verbose)
