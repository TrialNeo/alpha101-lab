import numpy as np
import pandas as pd
import operators as op


class Alpha:
    def __init__(self, open, high, low, close, volume, returns, vwap, adv20=None):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.returns = returns
        self.vwap = vwap
        self.adv20 = adv20

    def alpha001(self):
        # rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
        inner = self.close.where(self.returns >= 0, op.stddev(self.returns, 20))
        return op.rank(op.ts_argmax(op.signedpower(inner, 2), 5)) - 0.5

    def alpha002(self):
        # -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
        return -1 * op.correlation(
            op.rank(op.delta(np.log(self.volume), 2)),
            op.rank((self.close - self.open) / self.open),
            6
        )

    def alpha003(self):
        # -1 * correlation(rank(open), rank(volume), 10)
        return -1 * op.correlation(op.rank(self.open), op.rank(self.volume), 10)

    def alpha004(self):
        # -1 * Ts_Rank(rank(low), 9)
        return -1 * op.ts_rank(op.rank(self.low), 9)

    def alpha005(self):
        # rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
        return (
            op.rank(self.open - (op.ts_sum(self.vwap, 10) / 10))
            * (-1 * abs(op.rank(self.close - self.vwap)))
        )

    def alpha006(self):
        # -1 * correlation(open, volume, 10)
        return -1 * op.correlation(self.open, self.volume, 10)

    def alpha007(self):
        # ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        condition = adv20 < self.volume
        true_val = (-1 * op.ts_rank(abs(op.delta(self.close, 7)), 60)) * np.sign(op.delta(self.close, 7))
        false_val = -1
        return np.where(condition, true_val, false_val)

    def alpha008(self):
        # (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        return -1 * op.rank(
            (op.ts_sum(self.open, 5) * op.ts_sum(self.returns, 5))
            - op.delay((op.ts_sum(self.open, 5) * op.ts_sum(self.returns, 5)), 10)
        )

    def alpha009(self):
        # ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
        delta_close = op.delta(self.close, 1)
        condition1 = 0 < op.ts_min(delta_close, 5)
        condition2 = op.ts_max(delta_close, 5) < 0
        return np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close))

    def alpha010(self):
        # rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
        delta_close = op.delta(self.close, 1)
        condition1 = 0 < op.ts_min(delta_close, 4)
        condition2 = op.ts_max(delta_close, 4) < 0
        inner = pd.DataFrame(
            np.where(condition1, delta_close, np.where(condition2, delta_close, -1 * delta_close)),
            index=self.close.index, columns=self.close.columns
        )
        return op.rank(inner)

    def alpha011(self):
        # ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
        vwap_close = self.vwap - self.close
        return (
            op.rank(op.ts_max(vwap_close, 3)) + op.rank(op.ts_min(vwap_close, 3))
        ) * op.rank(op.delta(self.volume, 3))

    def alpha012(self):
        # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        return np.sign(op.delta(self.volume, 1)) * (-1 * op.delta(self.close, 1))

    def alpha013(self):
        # (-1 * rank(covariance(rank(close), rank(volume), 5)))
        return -1 * op.rank(op.covariance(op.rank(self.close), op.rank(self.volume), 5))

    def alpha014(self):
        # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        return (-1 * op.rank(op.delta(self.returns, 3))) * op.correlation(self.open, self.volume, 10)

    def alpha015(self):
        # (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        return -1 * op.ts_sum(op.rank(op.correlation(op.rank(self.high), op.rank(self.volume), 3)), 3)

    def alpha016(self):
        # (-1 * rank(covariance(rank(high), rank(volume), 5)))
        return -1 * op.rank(op.covariance(op.rank(self.high), op.rank(self.volume), 5))

    def alpha017(self):
        # (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        return (
            (-1 * op.rank(op.ts_rank(self.close, 10)))
            * op.rank(op.delta(op.delta(self.close, 1), 1))
            * op.rank(op.ts_rank(self.volume / adv20, 5))
        )

    def alpha018(self):
        # (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        return -1 * op.rank(
            op.stddev(np.abs(self.close - self.open), 5)
            + (self.close - self.open)
            + op.correlation(self.close, self.open, 10)
        )

    def alpha019(self):
        # ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        return (
            -1 * np.sign((self.close - op.delay(self.close, 7)) + op.delta(self.close, 7))
        ) * (1 + op.rank(1 + op.ts_sum(self.returns, 250)))

    def alpha020(self):
        # (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        return (
            (-1 * op.rank(self.open - op.delay(self.high, 1)))
            * op.rank(self.open - op.delay(self.close, 1))
            * op.rank(self.open - op.delay(self.low, 1))
        )

    def alpha021(self):
        # ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        ma8 = op.ts_sum(self.close, 8) / 8
        ma2 = op.ts_sum(self.close, 2) / 2
        std8 = op.stddev(self.close, 8)
        vol_ratio = self.volume / adv20
        condition1 = (ma8 + std8) < ma2
        condition2 = ma2 < (ma8 - std8)
        condition3 = (1 < vol_ratio) | (vol_ratio == 1)
        return np.where(condition1, -1, np.where(condition2, 1, np.where(condition3, 1, -1)))

    def alpha022(self):
        # (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        return -1 * (op.delta(op.correlation(self.high, self.volume, 5), 5) * op.rank(op.stddev(self.close, 20)))

    def alpha023(self):
        # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        condition = (op.ts_sum(self.high, 20) / 20) < self.high
        return np.where(condition, -1 * op.delta(self.high, 2), 0)

    def alpha024(self):
        # ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
        ma100 = op.ts_sum(self.close, 100) / 100
        delta_ma100 = op.delta(ma100, 100)
        ratio = delta_ma100 / op.delay(self.close, 100)
        condition = (ratio < 0.05) | (ratio == 0.05)
        return np.where(condition, -1 * (self.close - op.ts_min(self.close, 100)), -1 * op.delta(self.close, 3))

    def alpha025(self):
        # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        return op.rank((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close))

    def alpha026(self):
        # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
        return -1 * op.ts_max(
            op.correlation(op.ts_rank(self.volume, 5), op.ts_rank(self.high, 5), 5),
            3
        )

    def alpha027(self):
        # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
        inner = op.rank(op.ts_sum(op.correlation(op.rank(self.volume), op.rank(self.vwap), 6), 2) / 2.0)
        condition = 0.5 < inner
        return np.where(condition, -1, 1)

    def alpha028(self):
        # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        return op.scale((op.correlation(adv20, self.low, 5) + ((self.high + self.low) / 2)) - self.close)

    def alpha029(self):
        # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
        inner = -1 * op.rank(op.delta(self.close - 1, 5))
        inner = op.rank(op.rank(inner))
        inner = op.ts_min(inner, 2)
        inner = op.ts_sum(inner, 1)
        inner = np.log(inner)
        inner = op.scale(inner)
        inner = op.rank(op.rank(inner))
        inner = op.product(inner, 1)
        inner = np.minimum(inner, 5)
        return inner + op.ts_rank(op.delay(-1 * self.returns, 6), 5)

    def alpha030(self):
        # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        sign_sum = (
            np.sign(self.close - op.delay(self.close, 1))
            + np.sign(op.delay(self.close, 1) - op.delay(self.close, 2))
            + np.sign(op.delay(self.close, 2) - op.delay(self.close, 3))
        )
        return ((1.0 - op.rank(sign_sum)) * op.ts_sum(self.volume, 5)) / op.ts_sum(self.volume, 20)

    def alpha031(self):
        # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        term1 = op.rank(op.rank(op.rank(op.decay_linear(-1 * op.rank(op.rank(op.delta(self.close, 10))), 10))))
        term2 = op.rank(-1 * op.delta(self.close, 3))
        term3 = np.sign(op.scale(op.correlation(adv20, self.low, 12)))
        return term1 + term2 + term3

    def alpha032(self):
        # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
        return op.scale((op.ts_sum(self.close, 7) / 7) - self.close) + (
            20 * op.scale(op.correlation(self.vwap, op.delay(self.close, 5), 230))
        )

    def alpha033(self):
        # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        # ^1 is power of 1, which is a no-op
        return op.rank(-1 * (1 - (self.open / self.close)))

    def alpha034(self):
        # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        return op.rank(
            (1 - op.rank(op.stddev(self.returns, 2) / op.stddev(self.returns, 5)))
            + (1 - op.rank(op.delta(self.close, 1)))
        )

    def alpha035(self):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        return (
            op.ts_rank(self.volume, 32)
            * (1 - op.ts_rank((self.close + self.high) - self.low, 16))
            * (1 - op.ts_rank(self.returns, 32))
        )

    def alpha036(self):
        # Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        t1 = 2.21 * op.rank(op.correlation(self.close - self.open, op.delay(self.volume, 1), 15))
        t2 = 0.7 * op.rank(self.open - self.close)
        t3 = 0.73 * op.rank(op.ts_rank(op.delay(-1 * self.returns, 6), 5))
        t4 = op.rank(np.abs(op.correlation(self.vwap, adv20, 6)))
        t5 = 0.6 * op.rank(((op.ts_sum(self.close, 200) / 200) - self.open) * (self.close - self.open))
        return t1 + t2 + t3 + t4 + t5

    def alpha037(self):
        # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        return op.rank(op.correlation(op.delay(self.open - self.close, 1), self.close, 200)) + op.rank(
            self.open - self.close
        )

    def alpha038(self):
        # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        return (-1 * op.rank(op.ts_rank(self.close, 10))) * op.rank(self.close / self.open)

    def alpha039(self):
        # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        inner = op.delta(self.close, 7) * (1 - op.rank(op.decay_linear(self.volume / adv20, 9)))
        return (-1 * op.rank(inner)) * (1 + op.rank(op.ts_sum(self.returns, 250)))

    def alpha040(self):
        # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
        return (-1 * op.rank(op.stddev(self.high, 10))) * op.correlation(self.high, self.volume, 10)

    def alpha041(self):
        # Alpha#41: (((high * low)^0.5) - vwap)
        return np.sqrt(self.high * self.low) - self.vwap

    def alpha042(self):
        # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
        return op.rank(self.vwap - self.close) / op.rank(self.vwap + self.close)

    def alpha043(self):
        # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        return op.ts_rank(self.volume / adv20, 20) * op.ts_rank(-1 * op.delta(self.close, 7), 8)

    def alpha044(self):
        # Alpha#44: (-1 * correlation(high, rank(volume), 5))
        return -1 * op.correlation(self.high, op.rank(self.volume), 5)

    def alpha045(self):
        # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
        term1 = op.rank(op.ts_sum(op.delay(self.close, 5), 20) / 20)
        term2 = op.correlation(self.close, self.volume, 2)
        term3 = op.rank(op.correlation(op.ts_sum(self.close, 5), op.ts_sum(self.close, 20), 2))
        return -1 * ((term1 * term2) * term3)

    def alpha046(self):
        # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
        slope = ((op.delay(self.close, 20) - op.delay(self.close, 10)) / 10) - ((op.delay(self.close, 10) - self.close) / 10)
        condition1 = 0.25 < slope
        condition2 = slope < 0
        return np.where(condition1, -1, np.where(condition2, 1, (-1 * 1) * (self.close - op.delay(self.close, 1))))

    def alpha047(self):
        # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        term1 = ((op.rank(1 / self.close) * self.volume) / adv20) * ((self.high * op.rank(self.high - self.close)) / (op.ts_sum(self.high, 5) / 5))
        term2 = op.rank(self.vwap - op.delay(self.vwap, 5))
        return term1 - term2

    def alpha048(self):
        # Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
        numerator = op.indneutralize(
            (op.correlation(op.delta(self.close, 1), op.delta(op.delay(self.close, 1), 1), 250) * op.delta(self.close, 1)) / self.close,
            'subindustry'
        )
        denominator = op.ts_sum((op.delta(self.close, 1) / op.delay(self.close, 1)) ** 2, 250)
        return numerator / denominator

    def alpha049(self):
        # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        slope = ((op.delay(self.close, 20) - op.delay(self.close, 10)) / 10) - ((op.delay(self.close, 10) - self.close) / 10)
        condition = slope < (-1 * 0.1)
        return np.where(condition, 1, (-1 * 1) * (self.close - op.delay(self.close, 1)))

    def alpha050(self):
        # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
        return -1 * op.ts_max(op.rank(op.correlation(op.rank(self.volume), op.rank(self.vwap), 5)), 5)

    def alpha051(self):
        # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        slope = ((op.delay(self.close, 20) - op.delay(self.close, 10)) / 10) - ((op.delay(self.close, 10) - self.close) / 10)
        condition = slope < (-1 * 0.05)
        return np.where(condition, 1, (-1 * 1) * (self.close - op.delay(self.close, 1)))

    def alpha052(self):
        # Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
        term1 = (-1 * op.ts_min(self.low, 5)) + op.delay(op.ts_min(self.low, 5), 5)
        term2 = op.rank((op.ts_sum(self.returns, 240) - op.ts_sum(self.returns, 20)) / 220)
        term3 = op.ts_rank(self.volume, 5)
        return term1 * term2 * term3

    def alpha053(self):
        # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low)
        return -1 * op.delta(inner, 9)

    def alpha054(self):
        # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        return (-1 * ((self.low - self.close) * (self.open ** 5))) / ((self.low - self.high) * (self.close ** 5))

    def alpha055(self):
        # Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
        inner = (self.close - op.ts_min(self.low, 12)) / (op.ts_max(self.high, 12) - op.ts_min(self.low, 12))
        return -1 * op.correlation(op.rank(inner), op.rank(self.volume), 6)

    def alpha056(self):
        # Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
        # cap is proxied by close
        cap = self.close
        return 0 - (1 * (op.rank(op.ts_sum(self.returns, 10) / op.ts_sum(op.ts_sum(self.returns, 2), 3)) * op.rank(self.returns * cap)))

    def alpha057(self):
        # Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
        return 0 - (1 * ((self.close - self.vwap) / op.decay_linear(op.rank(op.ts_argmax(self.close, 30)), 2)))

    def alpha058(self):
        # Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
        return -1 * op.ts_rank(
            op.decay_linear(
                op.correlation(op.indneutralize(self.vwap, 'sector'), self.volume, 3.92795),
                7.89291
            ),
            5.50322
        )

    def alpha059(self):
        # Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
        inner = (self.vwap * 0.728317) + (self.vwap * (1 - 0.728317))
        return -1 * op.ts_rank(
            op.decay_linear(
                op.correlation(op.indneutralize(inner, 'industry'), self.volume, 4.25197),
                16.2289
            ),
            8.19648
        )

    def alpha060(self):
        # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
        inner = (((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)) * self.volume
        return 0 - (1 * ((2 * op.scale(op.rank(inner))) - op.scale(op.rank(op.ts_argmax(self.close, 10)))))

    def alpha061(self):
        # Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
        adv180 = op.ts_mean(self.volume, 180)
        return op.rank(self.vwap - op.ts_min(self.vwap, 16.1219)) < op.rank(op.correlation(self.vwap, adv180, 17.9282))

    def alpha062(self):
        # Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        left = op.rank(op.correlation(self.vwap, op.ts_sum(adv20, 22.4101), 9.91009))
        right = op.rank(
            (op.rank(self.open) + op.rank(self.open)) < (op.rank((self.high + self.low) / 2) + op.rank(self.high))
        )
        return (left < right) * -1

    def alpha063(self):
        # Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
        adv180 = op.ts_mean(self.volume, 180)
        term1 = op.rank(op.decay_linear(op.delta(op.indneutralize(self.close, 'industry'), 2.25164), 8.22237))
        weighted = (self.vwap * 0.318108) + (self.open * (1 - 0.318108))
        term2 = op.rank(op.decay_linear(op.correlation(weighted, op.ts_sum(adv180, 37.2467), 13.557), 12.2883))
        return (term1 - term2) * -1

    def alpha064(self):
        # Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
        adv120 = op.ts_mean(self.volume, 120)
        weighted_sum = op.ts_sum((self.open * 0.178404) + (self.low * (1 - 0.178404)), 12.7054)
        left = op.rank(op.correlation(weighted_sum, op.ts_sum(adv120, 12.7054), 16.6208))
        weighted_delta = op.delta((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404)), 3.69741)
        right = op.rank(weighted_delta)
        return (left < right) * -1

    def alpha065(self):
        # Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
        adv60 = op.ts_mean(self.volume, 60)
        weighted = (self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))
        left = op.rank(op.correlation(weighted, op.ts_sum(adv60, 8.6911), 6.40374))
        right = op.rank(self.open - op.ts_min(self.open, 13.635))
        return (left < right) * -1

    def alpha066(self):
        # Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
        term1 = op.rank(op.decay_linear(op.delta(self.vwap, 3.51013), 7.23052))
        inner = (((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))
        term2 = op.ts_rank(op.decay_linear(inner, 11.4157), 6.72611)
        return (term1 + term2) * -1

    def alpha067(self):
        # Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        base = op.rank(self.high - op.ts_min(self.high, 2.14593))
        exp = op.rank(op.correlation(op.indneutralize(self.vwap, 'sector'), op.indneutralize(adv20, 'subindustry'), 6.02936))
        return (base ** exp) * -1

    def alpha068(self):
        # Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
        adv15 = op.ts_mean(self.volume, 15)
        left = op.ts_rank(op.correlation(op.rank(self.high), op.rank(adv15), 8.91644), 13.9333)
        right = op.rank(op.delta((self.close * 0.518371) + (self.low * (1 - 0.518371)), 1.06157))
        return (left < right) * -1

    def alpha069(self):
        # Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        base = op.rank(op.ts_max(op.delta(op.indneutralize(self.vwap, 'industry'), 2.72412), 4.79344))
        weighted = (self.close * 0.490655) + (self.vwap * (1 - 0.490655))
        exp = op.ts_rank(op.correlation(weighted, adv20, 4.92416), 9.0615)
        return (base ** exp) * -1

    def alpha070(self):
        # Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
        adv50 = op.ts_mean(self.volume, 50)
        base = op.rank(op.delta(self.vwap, 1.29456))
        exp = op.ts_rank(op.correlation(op.indneutralize(self.close, 'industry'), adv50, 17.8256), 17.9171)
        return (base ** exp) * -1

    def alpha071(self):
        # Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
        adv180 = op.ts_mean(self.volume, 180)
        term1 = op.ts_rank(
            op.decay_linear(
                op.correlation(
                    op.ts_rank(self.close, 3.43976),
                    op.ts_rank(adv180, 12.0647),
                    18.0175
                ),
                4.20501
            ),
            15.6948
        )
        term2 = op.ts_rank(
            op.decay_linear(
                (op.rank(((self.low + self.open) - (self.vwap + self.vwap))) ** 2),
                16.4662
            ),
            4.4388
        )
        return np.maximum(term1, term2)

    def alpha072(self):
        # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
        adv40 = op.ts_mean(self.volume, 40)
        numerator = op.rank(
            op.decay_linear(
                op.correlation(((self.high + self.low) / 2), adv40, 8.93345),
                10.1519
            )
        )
        denominator = op.rank(
            op.decay_linear(
                op.correlation(
                    op.ts_rank(self.vwap, 3.72469),
                    op.ts_rank(self.volume, 18.5188),
                    6.86671
                ),
                2.95011
            )
        )
        return numerator / denominator

    def alpha073(self):
        # Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        weighted = (self.open * 0.147155) + (self.low * (1 - 0.147155))
        term1 = op.rank(op.decay_linear(op.delta(self.vwap, 4.72775), 2.91864))
        term2 = op.ts_rank(
            op.decay_linear(
                ((op.delta(weighted, 2.03608) / weighted) * -1),
                3.33829
            ),
            16.7411
        )
        return np.maximum(term1, term2) * -1

    def alpha074(self):
        # Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
        adv30 = op.ts_mean(self.volume, 30)
        left = op.rank(op.correlation(self.close, op.ts_sum(adv30, 37.4843), 15.1365))
        weighted = (self.high * 0.0261661) + (self.vwap * (1 - 0.0261661))
        right = op.rank(op.correlation(op.rank(weighted), op.rank(self.volume), 11.4791))
        return (left < right) * -1

    def alpha075(self):
        # Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
        adv50 = op.ts_mean(self.volume, 50)
        left = op.rank(op.correlation(self.vwap, self.volume, 4.24304))
        right = op.rank(op.correlation(op.rank(self.low), op.rank(adv50), 12.4413))
        return left < right

    def alpha076(self):
        # Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
        adv81 = op.ts_mean(self.volume, 81)
        term1 = op.rank(op.decay_linear(op.delta(self.vwap, 1.24383), 11.8259))
        term2 = op.ts_rank(
            op.decay_linear(
                op.ts_rank(
                    op.correlation(op.indneutralize(self.low, 'sector'), adv81, 8.14941),
                    19.569
                ),
                17.1543
            ),
            19.383
        )
        return np.maximum(term1, term2) * -1

    def alpha077(self):
        # Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
        adv40 = op.ts_mean(self.volume, 40)
        term1 = op.rank(
            op.decay_linear(
                ((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)),
                20.0451
            )
        )
        term2 = op.rank(
            op.decay_linear(
                op.correlation(((self.high + self.low) / 2), adv40, 3.1614),
                5.64125
            )
        )
        return np.minimum(term1, term2)

    def alpha078(self):
        # Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
        adv40 = op.ts_mean(self.volume, 40)
        weighted = (self.low * 0.352233) + (self.vwap * (1 - 0.352233))
        base = op.rank(
            op.correlation(
                op.ts_sum(weighted, 19.7428),
                op.ts_sum(adv40, 19.7428),
                6.83313
            )
        )
        exp = op.rank(op.correlation(op.rank(self.vwap), op.rank(self.volume), 5.77492))
        return base ** exp

    def alpha079(self):
        # Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
        adv150 = op.ts_mean(self.volume, 150)
        weighted = (self.close * 0.60733) + (self.open * (1 - 0.60733))
        left = op.rank(op.delta(op.indneutralize(weighted, 'sector'), 1.23438))
        right = op.rank(
            op.correlation(
                op.ts_rank(self.vwap, 3.60973),
                op.ts_rank(adv150, 9.18637),
                14.6644
            )
        )
        return left < right

    def alpha080(self):
        # Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
        adv10 = op.ts_mean(self.volume, 10)
        weighted = (self.open * 0.868128) + (self.high * (1 - 0.868128))
        base = op.rank(np.sign(op.delta(op.indneutralize(weighted, 'industry'), 4.04545)))
        exp = op.ts_rank(op.correlation(self.high, adv10, 5.11456), 5.53756)
        return (base ** exp) * -1

    def alpha081(self):
        # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
        adv10 = op.ts_mean(self.volume, 10)
        inner = op.rank(
            (op.rank(op.correlation(self.vwap, op.ts_sum(adv10, 49.6054), 8.47743)) ** 4)
        )
        left = op.rank(np.log(op.product(inner, 14.9655)))
        right = op.rank(op.correlation(op.rank(self.vwap), op.rank(self.volume), 5.07914))
        return (left < right) * -1

    def alpha082(self):
        # Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
        weighted = (self.open * 0.634196) + (self.open * (1 - 0.634196))
        term1 = op.rank(op.decay_linear(op.delta(self.open, 1.46063), 14.8717))
        term2 = op.ts_rank(
            op.decay_linear(
                op.correlation(op.indneutralize(self.volume, 'sector'), weighted, 17.4842),
                6.92131
            ),
            13.4283
        )
        return np.minimum(term1, term2) * -1

    def alpha083(self):
        # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
        hl_range = (self.high - self.low) / (op.ts_sum(self.close, 5) / 5)
        numerator = op.rank(op.delay(hl_range, 2)) * op.rank(op.rank(self.volume))
        denominator = hl_range / (self.vwap - self.close)
        return numerator / denominator

    def alpha084(self):
        # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
        return op.signedpower(
            op.ts_rank(self.vwap - op.ts_max(self.vwap, 15.3217), 20.7127),
            op.delta(self.close, 4.96796)
        )

    def alpha085(self):
        # Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
        adv30 = op.ts_mean(self.volume, 30)
        weighted = (self.high * 0.876703) + (self.close * (1 - 0.876703))
        base = op.rank(op.correlation(weighted, adv30, 9.61331))
        exp = op.rank(
            op.correlation(
                op.ts_rank(((self.high + self.low) / 2), 3.70596),
                op.ts_rank(self.volume, 10.1595),
                7.11408
            )
        )
        return base ** exp

    def alpha086(self):
        # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        left = op.ts_rank(
            op.correlation(self.close, op.ts_sum(adv20, 14.7444), 6.00049),
            20.4195
        )
        right = op.rank((self.open + self.close) - (self.vwap + self.open))
        return (left < right) * -1

    def alpha087(self):
        # Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
        adv81 = op.ts_mean(self.volume, 81)
        weighted = (self.close * 0.369701) + (self.vwap * (1 - 0.369701))
        term1 = op.rank(op.decay_linear(op.delta(weighted, 1.91233), 2.65461))
        term2 = op.ts_rank(
            op.decay_linear(
                np.abs(op.correlation(op.indneutralize(adv81, 'industry'), self.close, 13.4132)),
                4.89768
            ),
            14.4535
        )
        return np.maximum(term1, term2) * -1

    def alpha088(self):
        # Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
        adv60 = op.ts_mean(self.volume, 60)
        term1 = op.rank(
            op.decay_linear(
                (op.rank(self.open) + op.rank(self.low)) - (op.rank(self.high) + op.rank(self.close)),
                8.06882
            )
        )
        term2 = op.ts_rank(
            op.decay_linear(
                op.correlation(
                    op.ts_rank(self.close, 8.44728),
                    op.ts_rank(adv60, 20.6966),
                    8.01266
                ),
                6.65053
            ),
            2.61957
        )
        return np.minimum(term1, term2)

    def alpha089(self):
        # Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
        adv10 = op.ts_mean(self.volume, 10)
        weighted = (self.low * 0.967285) + (self.low * (1 - 0.967285))
        term1 = op.ts_rank(
            op.decay_linear(
                op.correlation(weighted, adv10, 6.94279),
                5.51607
            ),
            3.79744
        )
        term2 = op.ts_rank(
            op.decay_linear(
                op.delta(op.indneutralize(self.vwap, 'industry'), 3.48158),
                10.1466
            ),
            15.3012
        )
        return term1 - term2

    def alpha090(self):
        # Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
        adv40 = op.ts_mean(self.volume, 40)
        base = op.rank(self.close - op.ts_max(self.close, 4.66719))
        exp = op.ts_rank(
            op.correlation(op.indneutralize(adv40, 'subindustry'), self.low, 5.38375),
            3.21856
        )
        return (base ** exp) * -1

    def alpha091(self):
        # Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
        adv30 = op.ts_mean(self.volume, 30)
        term1 = op.ts_rank(
            op.decay_linear(
                op.decay_linear(
                    op.correlation(op.indneutralize(self.close, 'industry'), self.volume, 9.74928),
                    16.398
                ),
                3.83219
            ),
            4.8667
        )
        term2 = op.rank(
            op.decay_linear(
                op.correlation(self.vwap, adv30, 4.01303),
                2.6809
            )
        )
        return (term1 - term2) * -1

    def alpha092(self):
        # Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
        adv30 = op.ts_mean(self.volume, 30)
        term1 = op.ts_rank(
            op.decay_linear(
                ((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float),
                14.7221
            ),
            18.8683
        )
        term2 = op.ts_rank(
            op.decay_linear(
                op.correlation(op.rank(self.low), op.rank(adv30), 7.58555),
                6.94024
            ),
            6.80584
        )
        return np.minimum(term1, term2)

    def alpha093(self):
        # Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
        adv81 = op.ts_mean(self.volume, 81)
        weighted = (self.close * 0.524434) + (self.vwap * (1 - 0.524434))
        numerator = op.ts_rank(
            op.decay_linear(
                op.correlation(op.indneutralize(self.vwap, 'industry'), adv81, 17.4193),
                19.848
            ),
            7.54455
        )
        denominator = op.rank(
            op.decay_linear(
                op.delta(weighted, 2.77377),
                16.2664
            )
        )
        return numerator / denominator

    def alpha094(self):
        # Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
        adv60 = op.ts_mean(self.volume, 60)
        base = op.rank(self.vwap - op.ts_min(self.vwap, 11.5783))
        exp = op.ts_rank(
            op.correlation(
                op.ts_rank(self.vwap, 19.6462),
                op.ts_rank(adv60, 4.02992),
                18.0926
            ),
            2.70756
        )
        return (base ** exp) * -1

    def alpha095(self):
        # Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
        adv40 = op.ts_mean(self.volume, 40)
        left = op.rank(self.open - op.ts_min(self.open, 12.4105))
        right = op.ts_rank(
            op.rank(
                op.correlation(
                    op.ts_sum(((self.high + self.low) / 2), 19.1351),
                    op.ts_sum(adv40, 19.1351),
                    12.8742
                )
            ) ** 5,
            11.7584
        )
        return left < right

    def alpha096(self):
        # Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
        adv60 = op.ts_mean(self.volume, 60)
        term1 = op.ts_rank(
            op.decay_linear(
                op.correlation(op.rank(self.vwap), op.rank(self.volume), 3.83878),
                4.16783
            ),
            8.38151
        )
        term2 = op.ts_rank(
            op.decay_linear(
                op.ts_argmax(
                    op.correlation(
                        op.ts_rank(self.close, 7.45404),
                        op.ts_rank(adv60, 4.13242),
                        3.65459
                    ),
                    12.6556
                ),
                14.0365
            ),
            13.4143
        )
        return np.maximum(term1, term2) * -1

    def alpha097(self):
        # Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
        adv60 = op.ts_mean(self.volume, 60)
        weighted = (self.low * 0.721001) + (self.vwap * (1 - 0.721001))
        term1 = op.rank(
            op.decay_linear(
                op.delta(op.indneutralize(weighted, 'industry'), 3.3705),
                20.4523
            )
        )
        term2 = op.ts_rank(
            op.decay_linear(
                op.ts_rank(
                    op.correlation(
                        op.ts_rank(self.low, 7.87871),
                        op.ts_rank(adv60, 17.255),
                        4.97547
                    ),
                    18.5925
                ),
                15.7152
            ),
            6.71659
        )
        return (term1 - term2) * -1

    def alpha098(self):
        # Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
        adv5 = op.ts_mean(self.volume, 5)
        adv15 = op.ts_mean(self.volume, 15)
        term1 = op.rank(
            op.decay_linear(
                op.correlation(self.vwap, op.ts_sum(adv5, 26.4719), 4.58418),
                7.18088
            )
        )
        term2 = op.rank(
            op.decay_linear(
                op.ts_rank(
                    op.ts_argmin(
                        op.correlation(op.rank(self.open), op.rank(adv15), 20.8187),
                        8.62571
                    ),
                    6.95668
                ),
                8.07206
            )
        )
        return term1 - term2

    def alpha099(self):
        # Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
        adv60 = op.ts_mean(self.volume, 60)
        left = op.rank(
            op.correlation(
                op.ts_sum(((self.high + self.low) / 2), 19.8975),
                op.ts_sum(adv60, 19.8975),
                8.8136
            )
        )
        right = op.rank(op.correlation(self.low, self.volume, 6.28259))
        return (left < right) * -1

    def alpha100(self):
        # Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
        adv20 = self.adv20 if self.adv20 is not None else op.ts_mean(self.volume, 20)
        inner_rank = op.rank((((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)) * self.volume)
        term_a = 1.5 * op.scale(
            op.indneutralize(
                op.indneutralize(inner_rank, 'subindustry'),
                'subindustry'
            )
        )
        term_b = op.scale(
            op.indneutralize(
                op.correlation(self.close, op.rank(adv20), 5) - op.rank(op.ts_argmin(self.close, 30)),
                'subindustry'
            )
        )
        return 0 - (1 * ((term_a - term_b) * (self.volume / adv20)))

    def alpha101(self):
        # Alpha#101: ((close - open) / ((high - low) + .001))
        return (self.close - self.open) / ((self.high - self.low) + 0.001)