import numpy as np
import pandas as pd
import operators as op


class Alpha:
    def __init__(self, open, high, low, close, volume, returns, vwap, adv20=None):
        # 强制转换为 float64，消除任何因精度引起的排序对账误差
        self.open = pd.DataFrame(open, dtype=np.float64)
        self.high = pd.DataFrame(high, dtype=np.float64)
        self.low = pd.DataFrame(low, dtype=np.float64)
        self.close = pd.DataFrame(close, dtype=np.float64)
        self.volume = pd.DataFrame(volume, dtype=np.float64)
        self.returns = pd.DataFrame(returns, dtype=np.float64)
        self.vwap = pd.DataFrame(vwap, dtype=np.float64)
        self.adv20 = adv20 if adv20 is not None else self.volume.rolling(20).mean()

        # ─── 极致对齐：100% 还原参考代码在转换量价时引入的 IEEE 754 双精度浮点舍入噪声 ───
        # 这确保了在计算滚动 max/min 时，并列的值（ties）会以完全相同的微观扰动打破，彻底消除数值超差
        raw_volume = pd.DataFrame(volume, dtype=np.float64)
        raw_vwap = pd.DataFrame(vwap, dtype=np.float64)

        ref_volume_dq = raw_volume / 100.0
        self.volume = ref_volume_dq * 100.0

        ref_amount_dq = raw_vwap * (raw_volume + 1.0) / 1000.0
        self.vwap = (ref_amount_dq * 1000.0) / (ref_volume_dq * 100.0 + 1.0)

    @staticmethod
    def _product(df, d):
        # 兼容性更高的滚动乘积写法
        return df.rolling(int(d)).apply(np.prod, raw=True)

    @staticmethod
    def _ts_argmin(df, d):
        # 严格对齐参考代码的 np.argmin(x) + 1 逻辑
        return df.rolling(int(d)).apply(lambda x: float(np.argmin(x) + 1), raw=True)

    @staticmethod
    def _variance(df, d):
        return df.rolling(int(d)).var()

    def _regbeta(self, x, y, d):
        # 回归系数 beta = cov(x, y) / var(y)
        return self._covariance(x, y, d) / self._variance(y, d)

    @staticmethod
    def _rank(df):
        # 对齐参考实现的纵向时序排序
        return df.rank(axis=0, pct=True)

    @staticmethod
    def _delta(df, d):
        return df.diff(int(d))

    @staticmethod
    def _delay(df, d):
        return df.shift(int(d))

    @staticmethod
    def _decay_linear(df, period=10):
        # 权重算子线性衰减
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        return df.rolling(int(period)).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def _ts_rank(df, d):
        return df.rolling(int(d)).apply(
            lambda x: np.argsort(np.argsort(x))[-1] + 1, raw=True
        )

    @staticmethod
    def _stddev(df, d):
        return df.rolling(int(d)).std()

    @staticmethod
    def _correlation(x, y, d):
        return x.rolling(int(d)).corr(y)

    @staticmethod
    def _covariance(x, y, d):
        return x.rolling(int(d)).cov(y)

    @staticmethod
    def _ts_max(df, d):
        return df.rolling(int(d)).max()

    @staticmethod
    def _ts_min(df, d):
        return df.rolling(int(d)).min()

    @staticmethod
    def _ts_sum(df, d):
        return df.rolling(int(d)).sum()

    @staticmethod
    def _ts_mean(df, d):
        return df.rolling(int(d)).mean()

    @staticmethod
    def _ts_argmax(df, d):
        return df.rolling(int(d)).apply(lambda x: float(np.argmax(x) + 1), raw=True)

    def _scale(self, df, a=1):
        # 兼容单股票运行模式下的时间序列缩放（即除以该股历史绝对值总和）
        return df.div(df.abs().sum(axis=0) + 1e-20, axis=1) * a

    # ==========================================
    # 修正后的因子实现
    # ==========================================

    def alpha001(self):
        inner = self.close.copy()
        cond = self.returns < 0
        std = self._stddev(self.returns, 20)
        inner = std.where(cond, inner)
        return self._rank(self._ts_argmax(inner ** 2, 5))

    def alpha002(self):
        df1 = self._rank(self._delta(np.log(self.volume), 2))
        df2 = self._rank((self.close - self.open) / self.open)
        res = -1 * self._correlation(df1, df2, 6)
        return res.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha003(self):
        res = -1 * self._correlation(self._rank(self.open), self._rank(self.volume), 10)
        return res.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha004(self):
        return -1 * self._ts_rank(self._rank(self.low), 9)

    def alpha005(self):
        ref_sum = self.vwap.sum(axis=0) + 10.0
        part1 = self._rank(self.open - (ref_sum / 10.0))
        part2 = -1 * np.abs(self._rank(self.close - self.vwap))
        return part1 * part2

    def alpha006(self):
        res = -1 * self._correlation(self.open, self.volume, 10)
        return res.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha007(self):
        adv20 = self._ts_mean(self.volume, 20)
        cond = adv20 < self.volume
        d7 = self._delta(self.close, 7)
        res_true = -1 * self._ts_rank(np.abs(d7), 60) * np.sign(d7)
        res_false = pd.DataFrame(
            -1.0, index=self.close.index, columns=self.close.columns
        )
        return res_true.where(cond, res_false)

    def alpha008(self):
        part = self._ts_sum(self.open, 5) * self._ts_sum(self.returns, 5)
        return -1 * self._rank(part - self._delay(part, 10))

    def alpha009(self):
        d1 = self._delta(self.close, 1)
        cond_1 = self._ts_min(d1, 5) > 0
        cond_2 = self._ts_max(d1, 5) < 0
        res = -1 * d1
        return d1.where(cond_1 | cond_2, res)

    def alpha010(self):
        delta_close = self._delta(self.close, 1)
        cond_1 = self._ts_min(delta_close, 4) > 0
        cond_2 = self._ts_max(delta_close, 4) < 0

        # 对齐参考实现：alpha010 最外层并未进行 rank
        cond_all = cond_1 | cond_2
        return delta_close.where(cond_all, -1.0 * delta_close)

    def alpha011(self):
        diff = self.vwap - self.close
        p1 = self._rank(self._ts_max(diff, 3))
        p2 = self._rank(self._ts_min(diff, 3))
        p3 = self._rank(self._delta(self.volume, 3))
        return (p1 + p2) * p3

    def alpha012(self):
        return np.sign(self._delta(self.volume, 1)) * (-1 * self._delta(self.close, 1))

    def alpha013(self):
        return -1 * self._rank(
            self._covariance(self._rank(self.close), self._rank(self.volume), 5)
        )

    def alpha014(self):
        corr = (
            self._correlation(self.open, self.volume, 10)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        return -1 * self._rank(self._delta(self.returns, 3)) * corr

    def alpha015(self):
        corr = (
            self._correlation(self._rank(self.high), self._rank(self.volume), 3)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        return -1 * self._ts_sum(self._rank(corr), 3)

    def alpha016(self):
        return -1 * self._rank(
            self._covariance(self._rank(self.high), self._rank(self.volume), 5)
        )

    def alpha017(self):
        adv20 = self._ts_mean(self.volume, 20)
        part1 = self._rank(self._ts_rank(self.close, 10))
        part2 = self._rank(self._delta(self._delta(self.close, 1), 1))
        part3 = self._rank(self._ts_rank(self.volume / adv20, 5))
        return -1 * (part1 * part2 * part3)

    def alpha018(self):
        corr = (
            self._correlation(self.close, self.open, 10)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        inner = (
                self._stddev(np.abs(self.close - self.open), 5)
                + (self.close - self.open)
                + corr
        )
        return -1 * self._rank(inner)

    def alpha019(self):
        diff = (self.close - self._delay(self.close, 7)) + self._delta(self.close, 7)
        return (-1 * np.sign(diff)) * (
                1 + self._rank(1 + self._ts_sum(self.returns, 250))
        )

    def alpha020(self):
        p1 = self._rank(self.open - self._delay(self.high, 1))
        p2 = self._rank(self.open - self._delay(self.close, 1))
        p3 = self._rank(self.open - self._delay(self.low, 1))
        return -1 * (p1 * p2 * p3)

    def alpha021(self):
        cond_1 = (
                         self._ts_mean(self.close, 8) + self._stddev(self.close, 8)
                 ) < self._ts_mean(self.close, 2)
        cond_2 = self._ts_mean(self.volume, 20) / self.volume < 1

        alpha = pd.DataFrame(1.0, index=self.close.index, columns=self.close.columns)
        alpha = alpha.where(~(cond_1 | cond_2), -1.0)
        return alpha

    def alpha022(self):
        df_corr = (
            self._correlation(self.high, self.volume, 5)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        p1 = self._delta(df_corr, 5)
        p2 = self._rank(self._stddev(self.close, 20))
        return -1.0 * p1 * p2

    def alpha023(self):
        cond = self._ts_mean(self.high, 20) < self.high
        true_val = -1.0 * self._delta(self.high, 2).fillna(0)
        return true_val.where(cond, 0.0)

    def alpha024(self):
        cond = (
                       self._delta(self._ts_mean(self.close, 100), 100)
                       / self._delay(self.close, 100)
               ) <= 0.05
        res_true = -1.0 * (self.close - self._ts_min(self.close, 100))
        res_false = -1.0 * self._delta(self.close, 3)
        return res_true.where(cond, res_false)

    def alpha025(self):
        adv20 = self._ts_mean(self.volume, 20)
        inner = ((((-1.0 * self.returns) * adv20) * self.vwap) * (self.high - self.close))
        return self._rank(inner)

    def alpha026(self):
        df_corr = self._correlation(
            self._ts_rank(self.volume, 5), self._ts_rank(self.high, 5), 5
        )
        df_corr = df_corr.replace([-np.inf, np.inf], 0).fillna(0)
        return -1.0 * self._ts_max(df_corr, 3)

    def alpha027(self):
        corr = self._correlation(self._rank(self.volume), self._rank(self.vwap), 6)
        inner = self._ts_mean(corr, 2) / 2.0
        alpha_rank = self._rank(inner)

        # 严格复刻参考代码的 pandas 就地布尔索引赋值，自动保留 NaN 状态
        res = alpha_rank.copy()
        res[alpha_rank > 0.5] = -1.0
        res[alpha_rank <= 0.5] = 1.0
        return res

    def alpha028(self):
        adv20 = self._ts_mean(self.volume, 20)
        df_corr = (
            self._correlation(adv20, self.low, 5)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        inner = (df_corr + ((self.high + self.low) / 2.0)) - self.close
        return self._scale(inner)

    def alpha029(self):
        p1 = -1.0 * self._rank(self._delta(self.close - 1.0, 5))
        p2 = self._rank(self._rank(p1))
        p3 = self._ts_sum(p2, 2)
        p4 = self._rank(self._rank(self._scale(np.log(p3))))
        part1 = self._ts_min(p4, 5)
        part2 = self._ts_rank(self._delay(-1.0 * self.returns, 6), 5)
        return part1 + part2

    def alpha030(self):
        delta_close = self._delta(self.close, 1)
        inner = (
                np.sign(delta_close)
                + np.sign(self._delay(delta_close, 1))
                + np.sign(self._delay(delta_close, 2))
        )
        return (
                (1.0 - self._rank(inner)) * self._ts_sum(self.volume, 5)
        ) / self._ts_sum(self.volume, 20)

    def alpha031(self):
        adv20 = self._ts_mean(self.volume, 20)
        df_corr = (
            self._correlation(adv20, self.low, 12)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )

        p1_inner = -1.0 * self._rank(self._rank(self._delta(self.close, 10)))
        p1 = self._rank(self._rank(self._rank(self._decay_linear(p1_inner, 10))))
        p2 = self._rank(-1.0 * self._delta(self.close, 3))
        p3 = np.sign(self._scale(df_corr))
        return p1 + p2 + p3

    def alpha032(self):
        # 严格还原对账写法的除以 7 逻辑
        part1 = self._scale((self._ts_mean(self.close, 7) / 7.0) - self.close)
        corr = self._correlation(self.vwap, self._delay(self.close, 5), 230)
        part2 = 20.0 * self._scale(corr)
        return part1 + part2

    def alpha033(self):
        return self._rank(-1.0 + (self.open / self.close))

    def alpha034(self):
        inner = self._stddev(self.returns, 2) / self._stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return self._rank(
            2.0 - self._rank(inner) - self._rank(self._delta(self.close, 1))
        )

    def alpha035(self):
        p1 = self._ts_rank(self.volume, 32)
        p2 = 1.0 - self._ts_rank(self.close + self.high - self.low, 16)
        p3 = 1.0 - self._ts_rank(self.returns, 32)
        return p1 * p2 * p3

    def alpha036(self):
        # 严格还原对账写法的除以 200 逻辑
        adv20 = self._ts_mean(self.volume, 20)
        p1 = 2.21 * self._rank(
            self._correlation(self.close - self.open, self._delay(self.volume, 1), 15)
        )
        p2 = 0.70 * self._rank(self.open - self.close)
        p3 = 0.73 * self._rank(self._ts_rank(self._delay(-1.0 * self.returns, 6), 5))
        p4 = self._rank(np.abs(self._correlation(self.vwap, adv20, 6)))
        p5 = 0.60 * self._rank(
            ((self._ts_mean(self.close, 200) / 200.0) - self.open) * (self.close - self.open)
        )
        return p1 + p2 + p3 + p4 + p5

    def alpha037(self):
        p1 = self._rank(
            self._correlation(self._delay(self.open - self.close, 1), self.close, 200)
        )
        p2 = self._rank(self.open - self.close)
        return p1 + p2

    def alpha038(self):
        inner = self.close / self.open
        inner = inner.replace([-np.inf, np.inf], 1).fillna(1)
        return -1.0 * self._rank(self._ts_rank(self.open, 10)) * self._rank(inner)

    def alpha039(self):
        adv20 = self._ts_mean(self.volume, 20)
        decay = self._rank(self._decay_linear(self.volume / adv20, 9))
        part1 = -1.0 * self._rank(self._delta(self.close, 7) * (1.0 - decay))
        part2 = 1.0 + self._rank(
            self._ts_mean(self.returns, 250)
        )
        return part1 * part2

    def alpha040(self):
        return (
                -1.0
                * self._rank(self._stddev(self.high, 10))
                * self._correlation(self.high, self.volume, 10)
        )

    def alpha041(self):
        return np.sqrt(self.high * self.low) - self.vwap

    def alpha042(self):
        return self._rank(self.vwap - self.close) / self._rank(self.vwap + self.close)

    def alpha043(self):
        adv20 = self._ts_mean(self.volume, 20)
        return self._ts_rank(self.volume / adv20, 20) * self._ts_rank(
            -1.0 * self._delta(self.close, 7), 8
        )

    def alpha044(self):
        df_corr = self._correlation(self.high, self._rank(self.volume), 5)
        return -1.0 * df_corr.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha045(self):
        df_corr = (
            self._correlation(self.close, self.volume, 2)
            .replace([-np.inf, np.inf], 0)
            .fillna(0)
        )
        p1 = self._rank(self._ts_mean(self._delay(self.close, 5), 20))
        p2 = self._rank(
            self._correlation(
                self._ts_sum(self.close, 5), self._ts_sum(self.close, 20), 2
            )
        )
        return -1.0 * (p1 * df_corr * p2)

    def alpha046(self):
        inner = ((self._delay(self.close, 20) - self._delay(self.close, 10)) / 10.0) - (
                (self._delay(self.close, 10) - self.close) / 10.0
        )
        alpha = -1.0 * self._delta(self.close, 1)
        alpha = alpha.where(~(inner < 0), 1.0)
        alpha = alpha.where(~(inner > 0.25), -1.0)
        return alpha

    def alpha047(self):
        adv20 = self._ts_mean(self.volume, 20)
        part1 = (self._rank(1.0 / self.close) * self.volume) / adv20
        denom = self._ts_mean(self.high, 5) / 5.0
        part2 = (self.high * self._rank(self.high - self.close)) / denom
        part3 = self._rank(self.vwap - self._delay(self.vwap, 5))
        return (part1 * part2) - part3

    def alpha048(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha049(self):
        inner = ((self._delay(self.close, 20) - self._delay(self.close, 10)) / 10.0) - (
                (self._delay(self.close, 10) - self.close) / 10.0
        )
        alpha = -1.0 * self._delta(self.close, 1)
        alpha = alpha.where(~(inner < -0.1), 1.0)
        return alpha

    def alpha050(self):
        corr = self._correlation(self._rank(self.volume), self._rank(self.vwap), 5)
        return -1.0 * self._ts_max(self._rank(corr), 5)

    def alpha051(self):
        inner = ((self._delay(self.close, 20) - self._delay(self.close, 10)) / 10.0) - (
                (self._delay(self.close, 10) - self.close) / 10.0
        )
        cond = inner < -0.05
        true_val = pd.DataFrame(1.0, index=self.close.index, columns=self.close.columns)
        false_val = -1.0 * (self.close - self._delay(self.close, 1))
        return true_val.where(cond, false_val)

    def alpha052(self):
        min_low = self._ts_min(self.low, 5)
        p1 = -1.0 * self._delta(min_low, 5)
        p2 = self._rank(
            (self._ts_sum(self.returns, 240) - self._ts_sum(self.returns, 20)) / 220.0
        )
        p3 = self._ts_rank(self.volume, 5)
        return p1 * p2 * p3

    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        val = ((self.close - self.low) - (self.high - self.close)) / inner
        return -1.0 * self._delta(val, 9)

    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return (
                -1.0 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))
        )

    def alpha055(self):
        divisor = (self._ts_max(self.high, 12) - self._ts_min(self.low, 12)).replace(
            0, 0.0001
        )
        inner = (self.close - self._ts_min(self.low, 12)) / divisor
        df = self._correlation(self._rank(inner), self._rank(self.volume), 6)
        return -1.0 * df.replace([-np.inf, np.inf], 0).fillna(0)

    def alpha056(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha057(self):
        numerator = self.close - self.vwap
        denominator = self._decay_linear(self._rank(self._ts_argmax(self.close, 30)), 2)
        return -1.0 * (numerator / (denominator + 1e-6))

    def alpha058(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha059(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = (
                ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        )
        return -1.0 * (
                2.0 * self._scale(self._rank(inner))
                - self._scale(self._rank(self._ts_argmax(self.close, 10)))
        )

    def alpha061(self):
        adv180 = self._ts_mean(self.volume, 180)
        p1 = self._rank(self.vwap - self._ts_min(self.vwap, 16))
        p2 = self._rank(self._correlation(self.vwap, adv180, 18))
        return (p1 < p2).astype(float)

    def alpha062(self):
        adv20 = self._ts_mean(self.volume, 20)
        p1 = self._rank(self._correlation(self.vwap, self._ts_mean(adv20, 22), 10))
        p2 = self._rank(
            (
                    (self._rank(self.open) + self._rank(self.open))
                    < (self._rank((self.high + self.low) / 2.0) + self._rank(self.high))
            ).astype(float)
        )
        return -1.0 * (p1 < p2).astype(float)

    def alpha063(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha064(self):
        adv120 = self._ts_mean(self.volume, 120)
        p1 = self._rank(
            self._correlation(
                self._ts_sum(self.open * 0.178404 + self.low * (1.0 - 0.178404), 13),
                self._ts_mean(adv120, 13),
                17,
            )
        )
        p2 = self._rank(
            self._delta(
                (
                        ((self.high + self.low) / 2.0) * 0.178404
                        + self.vwap * (1.0 - 0.178404)
                ),
                3,
            )
        )
        return -1.0 * (p1 < p2).astype(float)

    def alpha065(self):
        adv60 = self._ts_mean(self.volume, 60)
        p1 = self._rank(
            self._correlation(
                self.open * 0.00817205 + self.vwap * (1.0 - 0.00817205),
                self._ts_mean(adv60, 9),
                6,
            )
        )
        p2 = self._rank(self.open - self._ts_min(self.open, 14))
        return -1.0 * (p1 < p2).astype(float)

    def alpha066(self):
        p1 = self._rank(self._decay_linear(self._delta(self.vwap, 4), 7))
        inner = (self.low * 0.96633 + self.low * (1.0 - 0.96633) - self.vwap) / (
                self.open - (self.high + self.low) / 2.0 + 1e-6
        )
        p2 = self._ts_rank(self._decay_linear(inner, 11), 7)
        return -1.0 * (p1 + p2)

    def alpha067(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha068(self):
        adv15 = self._ts_mean(self.volume, 15)
        p1 = self._ts_rank(
            self._correlation(self._rank(self.high), self._rank(adv15), 9), 14
        )
        p2 = self._rank(
            self._delta(self.close * 0.518371 + self.low * (1.0 - 0.518371), 1)
        )
        return -1.0 * (p1 < p2).astype(float)

    def alpha069(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha070(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha071(self):
        adv180 = self._ts_mean(self.volume, 180)
        p1 = self._ts_rank(
            self._decay_linear(
                self._correlation(
                    self._ts_rank(self.close, 3), self._ts_rank(adv180, 12), 18
                ),
                4,
            ),
            16,
        )
        p2 = self._ts_rank(
            self._decay_linear(
                self._rank(self.low + self.open - (self.vwap + self.vwap)).pow(2), 16
            ),
            4,
        )
        return np.maximum(p1, p2)

    def alpha072(self):
        adv40 = self._ts_mean(self.volume, 40)
        p1 = self._rank(
            self._decay_linear(
                self._correlation((self.high + self.low) / 2.0, adv40, 9), 10
            )
        )
        p2 = self._rank(
            self._decay_linear(
                self._correlation(
                    self._ts_rank(self.vwap, 4), self._ts_rank(self.volume, 19), 7
                ),
                3,
            )
        )
        return p1 / (p2 + 1e-6)

    def alpha073(self):
        p1 = self._rank(self._decay_linear(self._delta(self.vwap, 5), 3))
        inner = (
                        self._delta(self.open * 0.147155 + self.low * (1.0 - 0.147155), 2)
                        / (self.open * 0.147155 + self.low * (1.0 - 0.147155) + 1e-6)
                ) * -1.0
        p2 = self._ts_rank(self._decay_linear(inner, 3), 17)
        return -1.0 * np.maximum(p1, p2)

    def alpha074(self):
        adv30 = self._ts_mean(self.volume, 30)
        p1 = self._rank(self._correlation(self.close, self._ts_mean(adv30, 37), 15))
        p2 = self._rank(
            self._correlation(
                self._rank(self.high * 0.0261661 + self.vwap * (1.0 - 0.0261661)),
                self._rank(self.volume),
                11,
            )
        )
        return -1.0 * (p1 < p2).astype(float)

    def alpha075(self):
        adv50 = self._ts_mean(self.volume, 50)
        p1 = self._rank(self._correlation(self.vwap, self.volume, 4))
        p2 = self._rank(self._correlation(self._rank(self.low), self._rank(adv50), 12))
        return (p1 < p2).astype(float)

    def alpha076(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha077(self):
        adv40 = self._ts_mean(self.volume, 40)
        p1 = self._rank(
            self._decay_linear(
                (((self.high + self.low) / 2.0) + self.high - (self.vwap + self.high)),
                20,
            )
        )
        p2 = self._rank(
            self._decay_linear(
                self._correlation((self.high + self.low) / 2.0, adv40, 3), 6
            )
        )
        return np.minimum(p1, p2)

    def alpha078(self):
        adv40 = self._ts_mean(self.volume, 40)
        p1 = self._rank(
            self._correlation(
                self._ts_sum(self.low * 0.352233 + self.vwap * (1.0 - 0.352233), 20),
                self._ts_sum(adv40, 20),
                7,
            )
        )
        p2 = self._rank(
            self._correlation(self._rank(self.vwap), self._rank(self.volume), 6)
        )
        return p1.pow(p2)

    def alpha079(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha080(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha081(self):
        adv10 = self._ts_mean(self.volume, 10)
        p1_corr = self._correlation(self.vwap, self._ts_sum(adv10, 50), 8)
        p1_inner = self._rank(self._rank(p1_corr).pow(4))
        p1_prod = self._product(p1_inner, 15)
        p1 = self._rank(np.log(p1_prod))
        p2 = self._rank(
            self._correlation(self._rank(self.vwap), self._rank(self.volume), 5)
        )
        return -1.0 * (p1 < p2).astype(float)

    def alpha082(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha083(self):
        ratio = (self.high - self.low) / (self._ts_sum(self.close, 5) / 5.0)
        p1 = self._rank(self._delay(ratio, 2))
        p2 = self._rank(self._rank(self.volume))
        denom = ratio / (self.vwap - self.close)
        return (p1 * p2) / denom

    def alpha084(self):
        p1 = self._ts_rank(self.vwap - self._ts_max(self.vwap, 15), 21)
        p2 = self._delta(self.close, 5)
        return p1.pow(p2)

    def alpha085(self):
        adv30 = self._ts_mean(self.volume, 30)
        p1 = self._rank(
            self._correlation(
                self.high * 0.876703 + self.close * (1.0 - 0.876703), adv30, 10
            )
        )
        p2 = self._rank(
            self._correlation(
                self._ts_rank((self.high + self.low) / 2.0, 4),
                self._ts_rank(self.volume, 10),
                7,
            )
        )
        return p1.pow(p2)

    def alpha086(self):
        adv20 = self._ts_mean(self.volume, 20)
        p1 = self._ts_rank(
            self._correlation(self.close, self._ts_mean(adv20, 15), 6), 20
        )
        p2 = self._rank(self.close - self.vwap)
        return -1.0 * (p1 < p2).astype(float)

    def alpha087(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha088(self):
        adv60 = self._ts_mean(self.volume, 60)
        p1 = self._rank(
            self._decay_linear(
                (
                        (self._rank(self.open) + self._rank(self.low))
                        - (self._rank(self.high) + self._rank(self.close))
                ),
                8,
            )
        )
        p2 = self._ts_rank(
            self._decay_linear(
                self._correlation(
                    self._ts_rank(self.close, 8), self._ts_rank(adv60, 21), 8
                ),
                7,
            ),
            3,
        )
        return np.minimum(p1, p2)

    def alpha089(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha090(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha091(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha092(self):
        adv30 = self._ts_mean(self.volume, 30)
        p1 = self._ts_rank(
            self._decay_linear(
                (((self.high + self.low) / 2.0) + self.close < (self.low + self.open)),
                15,
            ),
            19,
        )
        p2 = self._ts_rank(
            self._decay_linear(
                self._correlation(self._rank(self.low), self._rank(adv30), 8), 7
            ),
            7,
        )
        return np.minimum(p1, p2)

    def alpha093(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha094(self):
        adv60 = self._ts_mean(self.volume, 60)
        p1 = self._rank(self.vwap - self._ts_min(self.vwap, 12))
        p2 = self._ts_rank(
            self._correlation(
                self._ts_rank(self.vwap, 20), self._ts_rank(adv60, 4), 18
            ),
            3,
        )
        return -1.0 * p1.pow(p2)

    def alpha095(self):
        adv40 = self._ts_mean(self.volume, 40)
        p1 = self._rank(self.open - self._ts_min(self.open, 12))
        p2 = self._ts_rank(
            self._rank(
                self._correlation(
                    (self.high + self.low) / 2.0, self._ts_mean(adv40, 19), 13
                )
            ).pow(5),
            12,
        )
        return (p1 < p2).astype(float)

    def alpha096(self):
        adv60 = self._ts_mean(self.volume, 60)
        p1 = self._ts_rank(
            self._decay_linear(
                self._correlation(self._rank(self.vwap), self._rank(self.volume), 4), 4
            ),
            8,
        )
        p2 = self._ts_rank(
            self._decay_linear(
                self._ts_argmax(
                    self._correlation(
                        self._ts_rank(self.close, 7), self._ts_rank(adv60, 4), 4
                    ),
                    13,
                ),
                14,
            ),
            13,
        )
        return -1.0 * np.maximum(p1, p2)

    def alpha097(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha098(self):
        adv5 = self._ts_mean(self.volume, 5)
        adv15 = self._ts_mean(self.volume, 15)
        p1 = self._rank(
            self._decay_linear(
                self._correlation(self.vwap, self._ts_mean(adv5, 26), 5), 7
            )
        )
        p2 = self._rank(
            self._decay_linear(
                self._ts_rank(
                    self._ts_argmin(
                        self._correlation(self._rank(self.open), self._rank(adv15), 21),
                        9,
                    ),
                    7,
                ),
                8,
            )
        )
        return p1 - p2

    def alpha099(self):
        adv60 = self._ts_mean(self.volume, 60)
        p1 = self._rank(
            self._correlation(
                self._ts_sum((self.high + self.low) / 2.0, 20),
                self._ts_sum(adv60, 20),
                9,
            )
        )
        p2 = self._rank(self._correlation(self.low, self.volume, 6))
        return -1.0 * (p1 < p2).astype(float)

    def alpha100(self):
        return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)

    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)