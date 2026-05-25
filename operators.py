# Created by Github@TrialNeo(shenpanpro@gmail.com)
# Created Time 2026/5/24 22:11.
# 基础算子
import numpy as np
import pandas as pd


def rank(df):
    """截面排名，归一化到 [0, 1]"""
    return df.rank(axis=1, pct=True)


def delta(df, d):
    """d 日差分（非整数 d 取 round）"""
    return df.diff(round(d))


def delay(df, d):
    """延迟 d 天"""
    return df.shift(d)


def ts_rank(df, d):
    """时间序列排名（滚动窗口），返回 0~1 的百分位"""
    d = int(d)
    return df.rolling(d).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)


def ts_stddev(df, d):
    """滚动标准差（ts_stddev 与 stddev 相同，统一命名）"""
    d = int(d)
    return df.rolling(d).std()


def stddev(df, d):
    """滚动标准差"""
    d = int(d)
    return df.rolling(d).std()


def correlation(x, y, d):
    """滚动相关系数"""
    d = int(d)
    return x.rolling(d).corr(y)


def covariance(x, y, d):
    """滚动协方差"""
    d = int(d)
    return x.rolling(d).cov(y)


def ts_max(df, d):
    d = int(d)
    return df.rolling(d).max()


def ts_min(df, d):
    d = int(d)
    return df.rolling(d).min()


def ts_sum(df, d):
    """滚动求和"""
    d = int(d)
    return df.rolling(d).sum()


def ts_mean(df, d):
    """滚动均值"""
    d = int(d)
    return df.rolling(d).mean()


def ts_argmax(df, d):
    """滚动窗口内最大值距今天数（0 = 今天）"""
    d = int(d)
    return df.rolling(d).apply(lambda x: d - 1 - np.argmax(x), raw=True)


def ts_argmin(df, d):
    """滚动窗口内最小值距今天数（0 = 今天）"""
    d = int(d)
    return df.rolling(d).apply(lambda x: d - 1 - np.argmin(x), raw=True)


def decay_linear(df, d):
    """线性衰减加权均值，权重 d, d-1, ..., 1（非整数 d 取 floor）"""
    d = int(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(d).apply(lambda x: np.dot(x, weights), raw=True)


def scale(df, a=1):
    """缩放使得 sum(abs(x)) = a，默认 a = 1"""
    s = df.abs().sum(axis=1)
    s = s.replace(0, np.nan)
    return df.div(s, axis=0) * a


def signedpower(x, a):
    """x^a，保留符号"""
    return np.sign(x) * np.abs(x) ** a


def product(df, d):
    """滚动乘积"""
    d = int(d)
    return df.rolling(d).apply(lambda x: np.prod(x), raw=True)


def indneutralize(x, g):
    """按分组 g 截面中性化（组内去均值）"""
    if isinstance(g, str):
        g = [g] * x.shape[1]
    demeaned = x.sub(x.T.groupby(g).transform('mean').T)
    return demeaned
