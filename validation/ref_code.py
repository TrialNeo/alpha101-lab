"""
Self-contained reference implementation wrapper.
Based on ref_code_1.py (STHSF/alpha101 style) with fixes for:
  - rank: axis=0 (column-wise ranking, matching user's implementation)
  - scale: column-wise normalization
  - decay_linear: modern pandas API
  - pd.Panel removal (use np.maximum/np.minimum)
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ── Auxiliary functions (corrected) ──────────────────────────────────────────

def ts_sum(df, window=10):
    return df.rolling(window).sum()


def sma(df, window=10):
    return df.rolling(window).mean()


def stddev(df, window=10):
    return df.rolling(window).std()


def correlation(x, y, window=10):
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    return x.rolling(window).cov(y)


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    return df.rolling(window).apply(rolling_rank, raw=True)


def rolling_prod(na):
    return np.prod(na)


def product(df, window=10):
    return df.rolling(window).apply(rolling_prod, raw=True)


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def rank(df):
    return df.rank(axis=0, pct=True)


def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum(axis=0) + 1e-20)


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax, raw=True) + 1


def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin, raw=True) + 1


def decay_linear(df, period=10):
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    filled = df.ffill().bfill().fillna(0)
    return filled.rolling(window=period).apply(lambda x: np.dot(x, weights), raw=True)


def product(df, window=10):
    return df.rolling(window).apply(rolling_prod, raw=True)


# ── Reference Alpha class (wide-format) ─────────────────────────────────────

class Alpha101Ref:
    """Reference implementation matching ref_code_1.py formulas exactly."""

    NOT_IMPLEMENTED = {48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100}

    def __init__(self, open_, high, low, close, volume, returns, vwap):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.returns = returns
        self.vwap = vwap

    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5))

    def alpha002(self):
        df = -1 * correlation(rank(delta(np.log(self.volume), 2)), rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        return rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * np.abs(rank((self.close - self.vwap))))

    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * np.sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha

    def alpha008(self):
        return -1 * rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                          delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)))

    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return rank(alpha)

    def alpha011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) *
                rank(delta(self.volume, 3)))

    def alpha012(self):
        return np.sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * rank(delta(self.returns, 3)) * df

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))

    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * rank((stddev(np.abs(self.close - self.open), 5) + (self.close - self.open)) + df)

    def alpha019(self):
        return ((-1 * np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    def alpha021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close.values), index=self.close.index, columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close.values), index=self.close.index, columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2).fillna(0)[cond]
        return alpha

    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))[cond]
        return alpha

    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))

    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * ts_max(df, 3)

    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    def alpha029(self):
        return (ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = np.sign(delta_close) + np.sign(delay(delta_close, 1)) + np.sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / ts_sum(self.volume, 20)

    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = np.sign(scale(df))
        return p1 + p2 + p3

    def alpha032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (
            20 * scale(correlation(self.vwap, delay(self.close, 5), 230)))

    def alpha033(self):
        return rank(-1 + (self.open / self.close))

    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (
            0.7 * rank((self.open - self.close)))) + (
            0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(
            np.abs(correlation(self.vwap, adv20, 6)))) + (
            0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))

    def alpha037(self):
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)

    def alpha039(self):
        adv20 = sma(self.volume, 20)
        return ((-1 * rank(
            delta(self.close, 7) * (1 - rank(decay_linear((self.volume / adv20), 9))))) *
                (1 + rank(sma(self.returns, 250))))

    def alpha040(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self):
        return np.power((self.high * self.low), 0.5) - self.vwap

    def alpha042(self):
        return rank((self.vwap - self.close)) / rank((self.vwap + self.close))

    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / adv20, 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df *
                     rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = -1 * delta(self.close)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / self.close)) * self.volume) / adv20) * (
            (self.high * rank((self.high - self.close))) / (sma(self.high, 5) / 5))) - rank(
            (self.vwap - delay(self.vwap, 5))))

    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = -1 * delta(self.close)
        alpha[inner < -0.1] = 1
        return alpha

    def alpha050(self):
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = -1 * delta(self.close)
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha057(self):
        return (0 - (1 * ((self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2))))

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self):
        adv180 = sma(self.volume, 180)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))).astype(float)

    def alpha062(self):
        adv20 = sma(self.volume, 20)
        return ((rank(correlation(self.vwap, ts_sum(adv20, 22), 10)) < rank(
            ((rank(self.open) + rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))))) * -1).astype(float)

    def alpha064(self):
        adv120 = sma(self.volume, 120)
        return ((rank(
            correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), ts_sum(adv120, 13), 17)) < rank(
            delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 4))) * -1).astype(float)

    def alpha065(self):
        adv60 = sma(self.volume, 60)
        return ((rank(
            correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), ts_sum(adv60, 9), 6)) < rank(
            (self.open - ts_min(self.open, 14)))) * -1).astype(float)

    def alpha066(self):
        return ((rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)

    def alpha068(self):
        adv15 = sma(self.volume, 15)
        return ((ts_rank(correlation(rank(self.high), rank(adv15), 9), 14) < rank(
            delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))) * -1).astype(float)

    def alpha071(self):
        adv180 = sma(self.volume, 180)
        p1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
        p2 = ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))) ** 2), 16), 4)
        return pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha072(self):
        adv40 = sma(self.volume, 40)
        return (rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 9), 10)) / rank(
            decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7), 3)))

    def alpha073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        p2 = ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / (
            (self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1), 3), 17)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha074(self):
        adv30 = sma(self.volume, 30)
        return ((rank(correlation(self.close, ts_sum(adv30, 37), 15)) < rank(
            correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11))) * -1).astype(float)

    def alpha075(self):
        adv50 = sma(self.volume, 50)
        return (rank(correlation(self.vwap, self.volume, 4)) < rank(correlation(rank(self.low), rank(adv50), 12))).astype(float)

    def alpha077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
        p2 = rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3), 6))
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha078(self):
        adv40 = sma(self.volume, 40)
        return (rank(
            correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)).pow(
            rank(correlation(rank(self.vwap), rank(self.volume), 6))))

    def alpha081(self):
        adv10 = sma(self.volume, 10)
        return ((rank(np.log(product(rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)) ** 4)), 15))) < rank(
            correlation(rank(self.vwap), rank(self.volume), 5))) * -1).astype(float)

    def alpha083(self):
        return ((rank(delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)) * rank(rank(self.volume))) / (
            ((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close)))

    def alpha084(self):
        return np.power(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))

    def alpha085(self):
        adv30 = sma(self.volume, 30)
        return (rank(correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10)).pow(
            rank(correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7))))

    def alpha086(self):
        adv20 = sma(self.volume, 20)
        return ((ts_rank(correlation(self.close, ts_sum(adv20, 15), 6), 20) < rank(
            ((self.open + self.close) - (self.vwap + self.open)))) * -1).astype(float)

    def alpha088(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        p2 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8), 7), 3)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha092(self):
        adv30 = sma(self.volume, 30)
        p1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float), 15), 19)
        p2 = ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8), 7), 7)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha094(self):
        adv60 = sma(self.volume, 60)
        return ((rank((self.vwap - ts_min(self.vwap, 12))).pow(
            ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)) * -1))

    def alpha095(self):
        adv40 = sma(self.volume, 40)
        return (rank((self.open - ts_min(self.open, 12))) < ts_rank(
            (rank(correlation(ts_sum(((self.high + self.low) / 2), 19), ts_sum(adv40, 19), 13)) ** 5), 12)).astype(float)

    def alpha096(self):
        adv60 = sma(self.volume, 60)
        p1 = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        p2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        return (rank(decay_linear(correlation(self.vwap, ts_sum(adv5, 26), 5), 7)) - rank(
            decay_linear(ts_rank(ts_argmin(correlation(rank(self.open), rank(adv15), 21), 9), 7), 8)))

    def alpha099(self):
        adv60 = sma(self.volume, 60)
        return ((rank(correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)) < rank(
            correlation(self.low, self.volume, 6))) * -1).astype(float)

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
