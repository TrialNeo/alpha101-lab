"""
Alpha101 - All 101 formulaic alphas (excluding IndNeutralize-dependent factors).
Wide-format implementation: all inputs/outputs are DataFrames (rows=dates, cols=stocks).
"""

import numpy as np
import pandas as pd


class Alpha:
    """
    Alpha101 factor library.
    All inputs are wide-format DataFrames: index=dates, columns=stocks.
    """

    def __init__(self, open, high, low, close, volume, returns, vwap, adv20=None):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.returns = returns
        self.vwap = vwap
        self.adv20 = adv20 if adv20 is not None else volume.rolling(20).mean()

    # ── Operator helpers (matching reference semantics) ──────────────────────

    @staticmethod
    def _rank(df):
        """Column-wise rank, normalized to [0, 1]."""
        return df.rank(axis=0, pct=True)

    @staticmethod
    def _scale(df, k=1):
        """Column-wise scale: sum(abs(x)) = k per column."""
        return df.mul(k).div(np.abs(df).sum(axis=0) + 1e-20)

    @staticmethod
    def _ts_rank(df, window):
        """Time-series rank: raw rank of last value in rolling window."""
        def _rolling_rank(x):
            from scipy.stats import rankdata
            return rankdata(x)[-1]
        return df.rolling(window).apply(_rolling_rank, raw=True)

    @staticmethod
    def _ts_argmax(df, window):
        """1-indexed position of max in rolling window."""
        return df.rolling(window).apply(np.argmax, raw=True) + 1

    @staticmethod
    def _ts_argmin(df, window):
        """1-indexed position of min in rolling window."""
        return df.rolling(window).apply(np.argmin, raw=True) + 1

    @staticmethod
    def _decay_linear(df, period):
        """Linear decay weighted average."""
        weights = np.arange(1, period + 1, dtype=float)
        weights /= weights.sum()
        filled = df.ffill().bfill().fillna(0)
        return filled.rolling(window=period).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def _product(df, window):
        """Rolling product."""
        return df.rolling(window).apply(np.prod, raw=True)

    @staticmethod
    def _safe_corr(x, y, window):
        """Rolling correlation with inf cleanup (no fillna - leave NaN for rank to handle)."""
        df = x.rolling(window).corr(y)
        return df.replace([-np.inf, np.inf], np.nan)

    @staticmethod
    def _corr_filled(x, y, window):
        """Rolling correlation with inf→0 and NaN→0 (for formulas that multiply/compare)."""
        df = x.rolling(window).corr(y)
        return df.replace([-np.inf, np.inf], 0).fillna(0)

    # ── Alpha factors ────────────────────────────────────────────────────────

    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = self.returns.rolling(20).std()[self.returns < 0]
        return self._rank(self._ts_argmax(inner ** 2, 5))

    def alpha002(self):
        return -1 * self._corr_filled(
            self._rank(np.log(self.volume).diff(2)),
            self._rank((self.close - self.open) / self.open), 6)

    def alpha003(self):
        return -1 * self._corr_filled(self._rank(self.open), self._rank(self.volume), 10)

    def alpha004(self):
        return -1 * self._ts_rank(self._rank(self.low), 9)

    def alpha005(self):
        return (self._rank(self.open - self.vwap.rolling(10).mean()) *
                (-1 * np.abs(self._rank(self.close - self.vwap))))

    def alpha006(self):
        return -1 * self._corr_filled(self.open, self.volume, 10)

    def alpha007(self):
        alpha = -1 * self._ts_rank(np.abs(self.close.diff(7)), 60) * np.sign(self.close.diff(7))
        alpha[self.adv20 >= self.volume] = -1
        return alpha

    def alpha008(self):
        inner = self.open.rolling(5).sum() * self.returns.rolling(5).sum()
        return -1 * self._rank(inner - inner.shift(10))

    def alpha009(self):
        d = self.close.diff(1)
        cond1 = d.rolling(5).min() > 0
        cond2 = d.rolling(5).max() < 0
        alpha = -1 * d
        alpha[cond1 | cond2] = d
        return alpha

    def alpha010(self):
        d = self.close.diff(1)
        cond1 = d.rolling(4).min() > 0
        cond2 = d.rolling(4).max() < 0
        alpha = -1 * d
        alpha[cond1 | cond2] = d
        return self._rank(alpha)

    def alpha011(self):
        diff = self.vwap - self.close
        return ((self._rank(diff.rolling(3).max()) + self._rank(diff.rolling(3).min())) *
                self._rank(self.volume.diff(3)))

    def alpha012(self):
        return np.sign(self.volume.diff(1)) * (-1 * self.close.diff(1))

    def alpha013(self):
        return -1 * self._rank(self._rank(self.close).rolling(5).cov(self._rank(self.volume)))

    def alpha014(self):
        return -1 * self._rank(self.returns.diff(3)) * self._corr_filled(self.open, self.volume, 10)

    def alpha015(self):
        corr = self._corr_filled(self._rank(self.high), self._rank(self.volume), 3)
        return -1 * self._rank(corr).rolling(3).sum()

    def alpha016(self):
        return -1 * self._rank(self._rank(self.high).rolling(5).cov(self._rank(self.volume)))

    def alpha017(self):
        return -1 * (self._rank(self._ts_rank(self.close, 10)) *
                     self._rank(self.close.diff(1).diff(1)) *
                     self._rank(self._ts_rank(self.volume / self.adv20, 5)))

    def alpha018(self):
        corr = self._corr_filled(self.close, self.open, 10)
        return -1 * self._rank((self.close - self.open).abs().rolling(5).std() +
                               (self.close - self.open) + corr)

    def alpha019(self):
        return ((-1 * np.sign((self.close - self.close.shift(7)) + self.close.diff(7))) *
                (1 + self._rank(1 + self.returns.rolling(250).sum())))

    def alpha020(self):
        return -1 * (self._rank(self.open - self.high.shift(1)) *
                     self._rank(self.open - self.close.shift(1)) *
                     self._rank(self.open - self.low.shift(1)))

    def alpha021(self):
        cond1 = self.close.rolling(8).mean() + self.close.rolling(8).std() < self.close.rolling(2).mean()
        cond2 = self.adv20 / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close.values), index=self.close.index, columns=self.close.columns)
        alpha[cond1 | cond2] = -1
        return alpha

    def alpha022(self):
        corr = self._corr_filled(self.high, self.volume, 5)
        return -1 * corr.diff(5) * self._rank(self.close.rolling(20).std())

    def alpha023(self):
        cond = self.high.rolling(20).mean() < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close.values), index=self.close.index, columns=self.close.columns)
        alpha[cond] = (-1 * self.high.diff(2).fillna(0))[cond]
        return alpha

    def alpha024(self):
        cond = self.close.rolling(100).mean().diff(100) / self.close.shift(100) <= 0.05
        alpha = -1 * self.close.diff(3)
        alpha[cond] = (-1 * (self.close - self.close.rolling(100).min()))[cond]
        return alpha

    def alpha025(self):
        return self._rank((-1 * self.returns) * self.adv20 * self.vwap * (self.high - self.close))

    def alpha026(self):
        corr = self._corr_filled(self._ts_rank(self.volume, 5), self._ts_rank(self.high, 5), 5)
        return -1 * corr.rolling(3).max()

    def alpha027(self):
        alpha = self._rank(self._corr_filled(self._rank(self.volume), self._rank(self.vwap), 6).rolling(2).mean() / 2.0)
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

    def alpha028(self):
        corr = self._corr_filled(self.adv20, self.low, 5)
        return self._scale(corr + (self.high + self.low) / 2 - self.close)

    def alpha029(self):
        # ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1*rank(delta(close-1,5)))),2))))),5)
        inner = self._rank(-1 * self._rank((self.close - 1).diff(5)))
        inner = inner.rolling(2).sum()
        inner = self._scale(np.log(inner))
        inner = self._rank(inner)
        return inner.rolling(5).min() + self._ts_rank((-1 * self.returns).shift(6), 5)

    def alpha030(self):
        d = self.close.diff(1)
        inner = np.sign(d) + np.sign(d.shift(1)) + np.sign(d.shift(2))
        return ((1.0 - self._rank(inner)) * self.volume.rolling(5).sum()) / self.volume.rolling(20).sum()

    def alpha031(self):
        corr = self._corr_filled(self.adv20, self.low, 12)
        p1 = self._rank(self._rank(self._rank(self._decay_linear(-1 * self._rank(self._rank(self.close.diff(10))), 10))))
        p2 = self._rank(-1 * self.close.diff(3))
        p3 = np.sign(self._scale(corr))
        return p1 + p2 + p3

    def alpha032(self):
        # Reference: scale(sma(close,7)/7 - close) + 20*scale(correlation(vwap,delay(close,5),230))
        return (self._scale(self.close.rolling(7).mean() / 7 - self.close) +
                20 * self._scale(self.vwap.rolling(230).corr(self.close.shift(5))))

    def alpha033(self):
        return self._rank(-1 + self.open / self.close)

    def alpha034(self):
        inner = self.returns.rolling(2).std() / self.returns.rolling(5).std()
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return self._rank(2 - self._rank(inner) - self._rank(self.close.diff(1)))

    def alpha035(self):
        return ((self._ts_rank(self.volume, 32) *
                 (1 - self._ts_rank(self.close + self.high - self.low, 16))) *
                (1 - self._ts_rank(self.returns, 32)))

    def alpha036(self):
        corr1 = self._safe_corr(self.close - self.open, self.volume.shift(1), 15)
        corr2 = self._safe_corr(self.vwap, self.adv20, 6)
        return ((2.21 * self._rank(corr1) +
                 0.7 * self._rank(self.open - self.close) +
                 0.73 * self._rank(self._ts_rank(-1 * self.returns.shift(6), 5)) +
                 self._rank(np.abs(corr2))) +
                0.6 * self._rank((self.close.rolling(200).mean() / 200 - self.open) * (self.close - self.open)))

    def alpha037(self):
        return (self._rank(self._safe_corr((self.open - self.close).shift(1), self.close, 200)) +
                self._rank(self.open - self.close))

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return -1 * self._rank(self._ts_rank(self.open, 10)) * self._rank(inner)

    def alpha039(self):
        return ((-1 * self._rank(self.close.diff(7) * (1 - self._rank(self._decay_linear(self.volume / self.adv20, 9))))) *
                (1 + self._rank(self.returns.rolling(250).mean())))

    def alpha040(self):
        return -1 * self._rank(self.high.rolling(10).std()) * self._corr_filled(self.high, self.volume, 10)

    def alpha041(self):
        return np.power(self.high * self.low, 0.5) - self.vwap

    def alpha042(self):
        return self._rank(self.vwap - self.close) / self._rank(self.vwap + self.close)

    def alpha043(self):
        return (self._ts_rank(self.volume / self.adv20, 20) *
                self._ts_rank(-1 * self.close.diff(7), 8))

    def alpha044(self):
        return -1 * self._corr_filled(self.high, self._rank(self.volume), 5)

    def alpha045(self):
        return -1 * (self._rank(self.close.shift(5).rolling(20).mean()) *
                     self._corr_filled(self.close, self.volume, 2) *
                     self._rank(self.close.rolling(5).sum().rolling(2).corr(self.close.rolling(20).sum())))

    def alpha046(self):
        inner = ((self.close.shift(20) - self.close.shift(10)) / 10 -
                 (self.close.shift(10) - self.close) / 10)
        alpha = -1 * self.close.diff(1)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    def alpha047(self):
        return (((self._rank(1 / self.close) * self.volume / self.adv20) *
                 (self.high * self._rank(self.high - self.close) / (self.high.rolling(5).mean() / 5))) -
                self._rank(self.vwap - self.vwap.shift(5)))

    def alpha049(self):
        inner = ((self.close.shift(20) - self.close.shift(10)) / 10 -
                 (self.close.shift(10) - self.close) / 10)
        alpha = -1 * self.close.diff(1)
        alpha[inner < -0.1] = 1
        return alpha

    def alpha050(self):
        return -1 * self._rank(self._safe_corr(self._rank(self.volume), self._rank(self.vwap), 5)).rolling(5).max()

    def alpha051(self):
        inner = ((self.close.shift(20) - self.close.shift(10)) / 10 -
                 (self.close.shift(10) - self.close) / 10)
        alpha = -1 * self.close.diff(1)
        alpha[inner < -0.05] = 1
        return alpha

    def alpha052(self):
        return ((-1 * self.low.rolling(5).min().diff(5)) *
                self._rank((self.returns.rolling(240).sum() - self.returns.rolling(20).sum()) / 220) *
                self._ts_rank(self.volume, 5))

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * ((((self.close - self.low) - (self.high - self.close)) / inner)).diff(9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    def alpha055(self):
        divisor = (self.high.rolling(12).max() - self.low.rolling(12).min()).replace(0, 0.0001)
        inner = (self.close - self.low.rolling(12).min()) / divisor
        return -1 * self._safe_corr(self._rank(inner), self._rank(self.volume), 6)

    def alpha057(self):
        return (0 - (self.close - self.vwap) / self._decay_linear(self._rank(self._ts_argmax(self.close, 30)), 2))

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return -(2 * self._scale(self._rank(inner)) - self._scale(self._rank(self._ts_argmax(self.close, 10))))

    def alpha061(self):
        adv180 = self.volume.rolling(180).mean()
        return (self._rank(self.vwap - self.vwap.rolling(16).min()) <
                self._rank(self._safe_corr(self.vwap, adv180, 18))).astype(float)

    def alpha062(self):
        return ((self._rank(self._safe_corr(self.vwap, self.adv20.rolling(22).sum(), 10)) <
                 self._rank((self._rank(self.open) + self._rank(self.open)) <
                            (self._rank((self.high + self.low) / 2) + self._rank(self.high)))) * -1).astype(float)

    def alpha064(self):
        adv120 = self.volume.rolling(120).mean()
        inner_open = self.open * 0.178404 + self.low * (1 - 0.178404)
        inner_vwap = (self.high + self.low) / 2 * 0.178404 + self.vwap * (1 - 0.178404)
        return ((self._rank(self._safe_corr(inner_open.rolling(13).mean(), adv120.rolling(13).sum(), 17)) <
                 self._rank(inner_vwap.diff(4))) * -1).astype(float)

    def alpha065(self):
        adv60 = self.volume.rolling(60).mean()
        inner = self.open * 0.00817205 + self.vwap * (1 - 0.00817205)
        return ((self._rank(self._safe_corr(inner, adv60.rolling(9).sum(), 6)) <
                 self._rank(self.open - self.open.rolling(14).min())) * -1).astype(float)

    def alpha066(self):
        denom = self.open - (self.high + self.low) / 2
        denom = denom.replace(0, 0.0001)
        inner = (self.low - self.vwap) / denom
        return ((self._rank(self._decay_linear(self.vwap.diff(4), 7)) +
                 self._ts_rank(self._decay_linear(inner, 11), 7)) * -1)

    def alpha068(self):
        adv15 = self.volume.rolling(15).mean()
        return ((self._ts_rank(self._safe_corr(self._rank(self.high), self._rank(adv15), 9), 14) <
                 self._rank((self.close * 0.518371 + self.low * (1 - 0.518371)).diff(1))) * -1).astype(float)

    def alpha071(self):
        adv180 = self.volume.rolling(180).mean()
        p1 = self._ts_rank(self._decay_linear(self._safe_corr(self._ts_rank(self.close, 3), self._ts_rank(adv180, 12), 18), 4), 16)
        p2 = self._ts_rank(self._decay_linear((self._rank(self.low + self.open - self.vwap - self.vwap) ** 2), 16), 4)
        return pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha072(self):
        adv40 = self.volume.rolling(40).mean()
        return (self._rank(self._decay_linear(self._safe_corr((self.high + self.low) / 2, adv40, 9), 10)) /
                self._rank(self._decay_linear(self._safe_corr(self._ts_rank(self.vwap, 4), self._ts_rank(self.volume, 19), 7), 3)))

    def alpha073(self):
        inner = self.open * 0.147155 + self.low * (1 - 0.147155)
        p1 = self._rank(self._decay_linear(self.vwap.diff(5), 3))
        p2 = self._ts_rank(self._decay_linear(-1 * inner.diff(2) / inner, 3), 17)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha074(self):
        adv30 = self.volume.rolling(30).mean()
        return ((self._rank(self.close.rolling(15).corr(adv30.rolling(37).sum())) <
                 self._rank(self._rank(self.high * 0.0261661 + self.vwap * (1 - 0.0261661)).rolling(11).corr(self._rank(self.volume)))) * -1).astype(float)

    def alpha075(self):
        adv50 = self.volume.rolling(50).mean()
        return (self._rank(self._safe_corr(self.vwap, self.volume, 4)) <
                self._rank(self._safe_corr(self._rank(self.low), self._rank(adv50), 12))).astype(float)

    def alpha077(self):
        adv40 = self.volume.rolling(40).mean()
        p1 = self._rank(self._decay_linear((self.high + self.low) / 2 + self.high - self.vwap - self.high, 20))
        p2 = self._rank(self._decay_linear(self._safe_corr((self.high + self.low) / 2, adv40, 3), 6))
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha078(self):
        adv40 = self.volume.rolling(40).mean()
        inner = self.low * 0.352233 + self.vwap * (1 - 0.352233)
        return (self._rank(inner.rolling(20).sum().rolling(7).corr(adv40.rolling(20).sum())) **
                self._rank(self._rank(self.vwap).rolling(6).corr(self._rank(self.volume))))

    def alpha081(self):
        adv10 = self.volume.rolling(10).mean()
        inner = self._rank(self._safe_corr(self.vwap, adv10.rolling(50).sum(), 8)) ** 4
        return ((self._rank(np.log(self._product(self._rank(inner), 15))) <
                 self._rank(self._safe_corr(self._rank(self.vwap), self._rank(self.volume), 5))) * -1).astype(float)

    def alpha083(self):
        hl_ratio = (self.high - self.low) / (self.close.rolling(5).sum() / 5)
        return ((self._rank(hl_ratio.shift(2)) * self._rank(self._rank(self.volume))) /
                (hl_ratio / (self.vwap - self.close)))

    def alpha084(self):
        return np.power(self._ts_rank(self.vwap - self.vwap.rolling(15).max(), 21), self.close.diff(5))

    def alpha085(self):
        adv30 = self.volume.rolling(30).mean()
        return (self._rank(self._safe_corr(self.high * 0.876703 + self.close * (1 - 0.876703), adv30, 10)) **
                self._rank(self._safe_corr(self._ts_rank((self.high + self.low) / 2, 4), self._ts_rank(self.volume, 10), 7)))

    def alpha086(self):
        return ((self._ts_rank(self._safe_corr(self.close, self.adv20.rolling(15).sum(), 6), 20) <
                 self._rank(self.open + self.close - self.vwap - self.open)) * -1).astype(float)

    def alpha088(self):
        adv60 = self.volume.rolling(60).mean()
        p1 = self._rank(self._decay_linear((self._rank(self.open) + self._rank(self.low)) - (self._rank(self.high) + self._rank(self.close)), 8))
        p2 = self._ts_rank(self._decay_linear(self._ts_rank(self.close, 8).rolling(8).corr(self._ts_rank(adv60, 21)), 7), 3)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha092(self):
        adv30 = self.volume.rolling(30).mean()
        cond = (((self.high + self.low) / 2 + self.close) < (self.low + self.open)).astype(float)
        p1 = self._ts_rank(self._decay_linear(cond, 15), 19)
        p2 = self._ts_rank(self._decay_linear(self._safe_corr(self._rank(self.low), self._rank(adv30), 8), 7), 7)
        return pd.DataFrame(np.minimum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha094(self):
        adv60 = self.volume.rolling(60).mean()
        return (self._rank(self.vwap - self.vwap.rolling(12).min()) **
                self._ts_rank(self._safe_corr(self._ts_rank(self.vwap, 20), self._ts_rank(adv60, 4), 18), 3)) * -1

    def alpha095(self):
        adv40 = self.volume.rolling(40).mean()
        return (self._rank(self.open - self.open.rolling(12).min()) <
                self._ts_rank(self._rank(self._safe_corr((self.high + self.low).rolling(19).mean() / 2,
                                                          adv40.rolling(19).sum(), 13)) ** 5, 12)).astype(float)

    def alpha096(self):
        adv60 = self.volume.rolling(60).mean()
        p1 = self._ts_rank(self._decay_linear(self._safe_corr(self._rank(self.vwap), self._rank(self.volume), 4), 4), 8)
        p2 = self._ts_rank(self._decay_linear(self._ts_argmax(self._safe_corr(self._ts_rank(self.close, 7), self._ts_rank(adv60, 4), 4), 13), 14), 13)
        return -1 * pd.DataFrame(np.maximum(p1.values, p2.values), index=p1.index, columns=p1.columns)

    def alpha098(self):
        adv5 = self.volume.rolling(5).mean()
        adv15 = self.volume.rolling(15).mean()
        return (self._rank(self._decay_linear(self.vwap.rolling(5).corr(adv5.rolling(26).sum()), 7)) -
                self._rank(self._decay_linear(self._ts_rank(self._ts_argmin(self._rank(self.open).rolling(21).corr(self._rank(adv15)), 9), 7), 8)))

    def alpha099(self):
        adv60 = self.volume.rolling(60).mean()
        return ((self._rank(self._safe_corr((self.high + self.low).rolling(20).mean() / 2,
                                            adv60.rolling(20).sum(), 9)) <
                 self._rank(self._safe_corr(self.low, self.volume, 6))) * -1).astype(float)

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)
