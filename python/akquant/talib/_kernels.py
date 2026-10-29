"""Python 后端的 TA-Lib 口径内核(numpy/pandas 实现).

每个函数只吃**不含非有限值**的 float64 一维数组(非有限值行由
:func:`akquant.talib.core.skip_non_finite_rows` 在外层剔除后再散回), 返回与输入
等长的数组, 预热期为 NaN。口径逐个对照 TA-Lib 0.8.2(C 库 0.8.1)验证过:
种子取法、预热长度、除零时的取值都与之一致。
"""

import math
from typing import cast

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

#: TA-Lib ``TA_IS_ZERO`` 的阈值
TA_EPSILON = 1e-14


def _nan(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=float)


def _is_zero(values: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, np.abs(values) < TA_EPSILON)


def _is_zero_scaled(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """TA-Lib ``TA_IS_ZERO_SCALED``: 差值相对两个操作数的量级可忽略."""
    return cast(np.ndarray, np.abs(values) <= TA_EPSILON * scale)


def _rolling(values: np.ndarray, period: int, how: str) -> np.ndarray:
    """逐窗口精确求 sum/mean/max/min(不用滑动累加, 避免浮点漂移), 可写."""
    out = _nan(values.shape[0])
    if values.shape[0] >= period:
        windows = sliding_window_view(values, period)
        out[period - 1 :] = getattr(windows, how)(axis=1)
    return out


def _first_valid(values: np.ndarray) -> int:
    valid = ~np.isnan(values)
    return int(np.argmax(valid)) if bool(valid.any()) else int(values.shape[0])


def _seeded_ewm(seed: float, tail: np.ndarray, alpha: float) -> np.ndarray:
    """``y[0]=seed``, ``y[i]=(1-alpha)*y[i-1]+alpha*tail[i-1]``."""
    series = pd.Series(np.concatenate(([seed], tail)), dtype=float)
    return np.array(series.ewm(alpha=alpha, adjust=False).mean(), dtype=float)


# --------------------------------------------------------------------- EMA 家族
def ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA: 前 period 个有效值的简单平均作种子, 之后 ``k=2/(period+1)`` 递推.

    允许开头有 NaN(复合指标把上一层的预热期 NaN 传进来), 从首个有效值起算。
    """
    out = _nan(values.shape[0])
    start = _first_valid(values)
    segment = values[start:]
    if segment.shape[0] < period:
        return out
    out[start + period - 1 :] = _seeded_ewm(
        float(segment[:period].mean()), segment[period:], 2.0 / (period + 1.0)
    )
    return out


def dema(values: np.ndarray, period: int) -> np.ndarray:
    """DEMA = 2*EMA - EMA(EMA), 第二层在第一层的首个有效值处重新起种子."""
    first = ema(values, period)
    return cast(np.ndarray, 2.0 * first - ema(first, period))


def tema(values: np.ndarray, period: int) -> np.ndarray:
    """TEMA = 3*EMA1 - 3*EMA2 + EMA3."""
    first = ema(values, period)
    second = ema(first, period)
    return cast(np.ndarray, 3.0 * first - 3.0 * second + ema(second, period))


def trix(values: np.ndarray, period: int) -> np.ndarray:
    """TRIX: 三重 EMA 的单期变化率(%); 前值为 0 时记 0."""
    triple = ema(ema(ema(values, period), period), period)
    out = _nan(values.shape[0])
    prev = triple[:-1]
    cur = triple[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(prev == 0.0, 0.0, (cur - prev) / prev * 100.0)
    out[1:][np.isnan(prev) | np.isnan(cur)] = np.nan
    return out


def t3(values: np.ndarray, period: int, vfactor: float) -> np.ndarray:
    """T3: 六层 EMA 的加权组合."""
    e1 = ema(values, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)
    a = vfactor
    c1 = -(a**3)
    c2 = 3.0 * a * a + 3.0 * a**3
    c3 = -6.0 * a * a - 3.0 * a - 3.0 * a**3
    c4 = 1.0 + 3.0 * a + a**3 + 3.0 * a * a
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def macd(
    values: np.ndarray, fast: int, slow: int, signal: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD: 快慢 EMA **同在第 slow 根**起种子(快线种子取最后 fast 根均值).

    这是 TA-Lib 的实际行为, 与"两条 EMA 各自起算"不同; 三个输出都要等信号线
    就绪才给值。``slow < fast`` 时与 TA-Lib 一样互换。
    """
    if slow < fast:
        fast, slow = slow, fast
    length = values.shape[0]
    line = _nan(length)
    start = slow - 1
    if length > start:
        fast_line = _seeded_ewm(
            float(values[start - fast + 1 : start + 1].mean()),
            values[start + 1 :],
            2.0 / (fast + 1.0),
        )
        slow_line = _seeded_ewm(
            float(values[: start + 1].mean()), values[start + 1 :], 2.0 / (slow + 1.0)
        )
        line[start:] = fast_line - slow_line
    signal_line = ema(line, signal)
    line[: start + signal - 1] = np.nan
    return line, signal_line, line - signal_line


def apo(values: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """APO: 两条 EMA **各自**起种子后相减(与 MACD 不同)."""
    return cast(np.ndarray, ema(values, fast) - ema(values, slow))


def ppo(values: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """PPO: ``(快 - 慢) / 慢 * 100``, 慢线为 0 时记 0."""
    fast_line = ema(values, fast)
    slow_line = ema(values, slow)
    out = _nan(values.shape[0])
    valid = ~np.isnan(slow_line)
    nonzero = valid & ~_is_zero(slow_line)
    out[valid] = 0.0
    out[nonzero] = (
        (fast_line[nonzero] - slow_line[nonzero]) / slow_line[nonzero] * 100.0
    )
    return out


def kama(values: np.ndarray, period: int) -> np.ndarray:
    """KAMA: 以第 period 根价格为递推起点, 首个输出在下标 period."""
    length = values.shape[0]
    out = _nan(length)
    if length <= period:
        return out
    fast_sc, slow_sc = 2.0 / 3.0, 2.0 / 31.0
    change = values[period:] - values[:-period]
    volatility = _rolling(np.abs(np.diff(values)), period, "sum")[period - 1 :]
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(
            (volatility <= 0.0) | (volatility <= change),
            1.0,
            np.abs(change / volatility),
        )
    smoothing = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    prev = float(values[period - 1])
    for i in range(period, length):
        prev = (float(values[i]) - prev) * float(smoothing[i - period]) + prev
        out[i] = prev
    return out


# --------------------------------------------------------------------- Wilder 家族
def trange(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """真实波幅; 首根没有前收盘, 记 NaN(TA-Lib lookback=1)."""
    out = _nan(close.shape[0])
    prev = close[:-1]
    out[1:] = np.maximum.reduce(
        [high[1:] - low[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)]
    )
    return out


def _wilder_average(values: np.ndarray, period: int, first: int) -> np.ndarray:
    """Wilder 平均: 在下标 first 处起种子, 之后按 Wilder 公式递推.

    种子是 ``mean(values[first-period+1..first])``, 递推式
    ``(prev*(period-1)+x)/period``。
    """
    out = _nan(values.shape[0])
    if values.shape[0] <= first:
        return out
    out[first:] = _seeded_ewm(
        float(values[first - period + 1 : first + 1].mean()),
        values[first + 1 :],
        1.0 / period,
    )
    return out


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ATR: 前 period 个真实波幅(第 1..period 根)的均值作种子, Wilder 平滑."""
    true_range = trange(high, low, close)
    if period == 1:
        return true_range
    return _wilder_average(true_range, period, period)


def natr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """NATR = ATR / close * 100, close 为 0 时记 0."""
    average = atr(high, low, close, period)
    out = _nan(close.shape[0])
    valid = ~np.isnan(average)
    nonzero = valid & (close != 0.0)
    out[valid] = 0.0
    out[nonzero] = average[nonzero] / close[nonzero] * 100.0
    return out


def _directional_movement(
    high: np.ndarray, low: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    up = np.r_[0.0, high[1:] - high[:-1]]
    down = np.r_[0.0, low[:-1] - low[1:]]
    plus = np.where((up > 0.0) & (up > down), up, 0.0)
    minus = np.where((down > 0.0) & (down > up), down, 0.0)
    return plus, minus


def _wilder_sum(values: np.ndarray, period: int) -> np.ndarray:
    """+DM/-DM/TR 的 Wilder 平滑和.

    第 1..period-1 根只累加, 第 period 根起 ``s = s - s/period + x``,
    等价于对 ``period*x`` 做 alpha=1/period 的 EWM。
    """
    out = _nan(values.shape[0])
    if values.shape[0] <= period:
        return out
    out[period - 1 :] = _seeded_ewm(
        float(values[1:period].sum()), values[period:] * period, 1.0 / period
    )
    out[period - 1] = np.nan
    return out


def plus_minus_di(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """(+DI, -DI); 平滑 TR 不为正时两者都记 0."""
    plus, minus = _directional_movement(high, low)
    true_range = np.nan_to_num(trange(high, low, close))
    smooth_plus = _wilder_sum(plus, period)
    smooth_minus = _wilder_sum(minus, period)
    smooth_tr = _wilder_sum(true_range, period)
    length = close.shape[0]
    plus_di = _nan(length)
    minus_di = _nan(length)
    valid = ~np.isnan(smooth_tr)
    zero = valid & (smooth_tr <= 0.0)
    nonzero = valid & ~zero
    plus_di[zero] = 0.0
    minus_di[zero] = 0.0
    plus_di[nonzero] = 100.0 * smooth_plus[nonzero] / smooth_tr[nonzero]
    minus_di[nonzero] = 100.0 * smooth_minus[nonzero] / smooth_tr[nonzero]
    return plus_di, minus_di


def dx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """DX; +DI 与 -DI 之和为 0 时沿用上一个 DX(首个之前记 0)."""
    plus_di, minus_di = plus_minus_di(high, low, close, period)
    total = plus_di + minus_di
    started = ~np.isnan(total)
    raw = _nan(close.shape[0])
    nonzero = started & ~_is_zero(total)
    raw[nonzero] = 100.0 * np.abs(plus_di[nonzero] - minus_di[nonzero]) / total[nonzero]
    out = np.array(pd.Series(raw).ffill(), dtype=float)
    out[started & np.isnan(out)] = 0.0
    out[~started] = np.nan
    return out


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ADX: 前 period 个 DX 的均值作种子(首个输出在下标 2*period-1)."""
    return _wilder_average(dx(high, low, close, period), period, 2 * period - 1)


def adxr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """ADXR = (ADX[i] + ADX[i-(period-1)]) / 2, 首个输出在下标 3*period-2."""
    average = adx(high, low, close, period)
    out = _nan(close.shape[0])
    first = 3 * period - 2
    if close.shape[0] > first:
        lagged = average[first - (period - 1) : average.shape[0] - (period - 1)]
        out[first:] = (average[first:] + lagged) / 2.0
    return out


def _gain_loss_averages(
    close: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    if close.shape[0] == 0:
        return _nan(0), _nan(0)
    diff = np.r_[np.nan, np.diff(close)]
    gain = np.where(diff > 0.0, diff, 0.0)
    loss = np.where(diff < 0.0, -diff, 0.0)
    return _wilder_average(gain, period, period), _wilder_average(loss, period, period)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI = 100 * 平均涨幅 / (平均涨幅 + 平均跌幅), 分母为 0 时记 0."""
    gain, loss = _gain_loss_averages(close, period)
    total = gain + loss
    out = _nan(close.shape[0])
    valid = ~np.isnan(total)
    nonzero = valid & (total > 0.0)
    out[valid] = 0.0
    out[nonzero] = 100.0 * gain[nonzero] / total[nonzero]
    return out


def cmo(close: np.ndarray, period: int) -> np.ndarray:
    """CMO: 与 RSI 同一套 Wilder 平均, ``100*(涨-跌)/(涨+跌)``."""
    gain, loss = _gain_loss_averages(close, period)
    total = gain + loss
    out = _nan(close.shape[0])
    valid = ~np.isnan(total)
    nonzero = valid & (total > 0.0)
    out[valid] = 0.0
    out[nonzero] = 100.0 * (gain[nonzero] - loss[nonzero]) / total[nonzero]
    return out


# --------------------------------------------------------------------- 窗口/口径类
def cci(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, c: float
) -> np.ndarray:
    """CCI; 分子或平均偏差相对均值可忽略(停牌、一字板)时记 0."""
    typical = (high + low + close) / 3.0
    out = _nan(close.shape[0])
    if close.shape[0] < period:
        return out
    windows = sliding_window_view(typical, period)
    mean = windows.sum(axis=1) / period
    deviation = np.abs(windows - mean[:, None]).sum(axis=1) / period
    numerator = typical[period - 1 :] - mean
    divisor = c * deviation
    scale = np.abs(mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = numerator / divisor
    zero = (
        _is_zero_scaled(numerator, scale)
        | _is_zero_scaled(deviation, scale)
        | (divisor == 0.0)
    )
    out[period - 1 :] = np.where(zero, 0.0, value)
    return out


def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
) -> np.ndarray:
    """MFI; 典型价变动相对其量级可忽略时不计资金流, 窗口无资金流时记 0."""
    if close.shape[0] == 0:
        return _nan(0)
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    change = np.r_[0.0, typical[1:] - typical[:-1]]
    scale = np.r_[0.0, np.abs(typical[1:]) + np.abs(typical[:-1])]
    flow = np.where(_is_zero_scaled(change, scale), 0.0, money_flow)
    flow[0] = 0.0
    positive = _rolling(np.where(change < 0.0, 0.0, flow), period, "sum")
    negative = _rolling(np.where(change < 0.0, flow, 0.0), period, "sum")
    total = positive + negative
    out = _nan(close.shape[0])
    tail = slice(period, None)
    clamped = np.clip(positive, 0.0, np.maximum(total, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out[tail] = np.where(
            total[tail] <= 0.0, 0.0, 100.0 * clamped[tail] / total[tail]
        )
    return out


def aroon(
    high: np.ndarray, low: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """(aroondown, aroonup); 窗口是 period+1 根, 并列极值取最近的一根."""
    length = high.shape[0]
    down = _nan(length)
    up = _nan(length)
    if length <= period:
        return down, up
    # 反转窗口后 argmax/argmin 直接给出"距今几根", 并列时取到最近的那根
    high_windows = sliding_window_view(high, period + 1)[:, ::-1]
    low_windows = sliding_window_view(low, period + 1)[:, ::-1]
    factor = 100.0 / period
    up[period:] = factor * (period - np.argmax(high_windows, axis=1))
    down[period:] = factor * (period - np.argmin(low_windows, axis=1))
    return down, up


def ultosc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period1: int,
    period2: int,
    period3: int,
) -> np.ndarray:
    """ULTOSC; 三个周期先排序(与 TA-Lib 一致), 首个输出在下标 max(period)."""
    short, medium, long = sorted((period1, period2, period3))
    length = close.shape[0]
    prev_close = np.r_[np.nan, close[:-1]]
    true_low = np.minimum(low, prev_close)
    buying_pressure = close - true_low
    true_range = np.maximum(high, prev_close) - true_low
    total = np.zeros(length)
    for period, weight in ((short, 4.0), (medium, 2.0), (long, 1.0)):
        bp_sum = _rolling(buying_pressure, period, "sum")
        tr_sum = _rolling(true_range, period, "sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            total = total + np.where(tr_sum > 0.0, weight * bp_sum / tr_sum, 0.0)
    out = _nan(length)
    out[long:] = 100.0 * total[long:] / 7.0
    return out


def _simple_returns(values: np.ndarray) -> np.ndarray:
    prev = values[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(_is_zero(prev), 0.0, (values[1:] - prev) / prev)
    return cast(np.ndarray, np.r_[np.nan, returns])


def beta(real0: np.ndarray, real1: np.ndarray, period: int) -> np.ndarray:
    """BETA: 两条序列先换成单期收益率, 取 real1 收益对 real0 收益的回归斜率."""
    x = _simple_returns(real0)
    y = _simple_returns(real1)
    sxx, sx, sxy, sy = (_rolling(v, period, "sum") for v in (x * x, x, x * y, y))
    denominator = period * sxx - sx * sx
    out = _nan(real0.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(
            denominator > TA_EPSILON * period * sxx,
            (period * sxy - sx * sy) / denominator,
            0.0,
        )
    out[period:] = value[period:]
    return out


def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV: 首根取该根成交量, 之后按涨跌加减成交量."""
    steps = np.sign(np.r_[0.0, np.diff(close)]) * volume
    if steps.shape[0]:
        steps[0] = volume[0]
    return np.cumsum(steps)


def stoch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int,
    slowk_period: int,
    slowd_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """STOCH(均线类型 SMA); 两个输出都从下标 fastk+slowk+slowd-3 起给值."""
    highest = _rolling(high, fastk_period, "max")
    lowest = _rolling(low, fastk_period, "min")
    price_range = highest - lowest
    flat = _is_zero_scaled(price_range, np.abs(highest) + np.abs(lowest))
    with np.errstate(divide="ignore", invalid="ignore"):
        fast_k = np.where(flat, 0.0, (close - lowest) / price_range * 100.0)
    fast_k[: fastk_period - 1] = np.nan
    slow_k = _rolling(fast_k, slowk_period, "mean")
    slow_d = _rolling(slow_k, slowd_period, "mean")
    slow_k[: fastk_period + slowk_period + slowd_period - 3] = np.nan
    return slow_k, slow_d


def sar(
    high: np.ndarray, low: np.ndarray, acceleration: float, maximum: float
) -> np.ndarray:
    """Parabolic SAR; 方向由前两根的 -DM 决定, 首个输出在下标 1."""
    length = high.shape[0]
    out = _nan(length)
    if length < 2:
        return out
    acceleration = min(acceleration, maximum)
    af = acceleration
    up = float(high[1] - high[0])
    down = float(low[0] - low[1])
    is_long = not (down > 0.0 and up < down)
    if is_long:
        ep, value = float(high[1]), float(low[0])
    else:
        ep, value = float(low[1]), float(high[0])
    new_low, new_high = float(low[1]), float(high[1])
    for i in range(1, length):
        prev_low, prev_high = new_low, new_high
        new_low, new_high = float(low[i]), float(high[i])
        if is_long:
            if new_low <= value:
                is_long = False
                value = max(ep, prev_high, new_high)
                out[i] = value
                af = acceleration
                ep = new_low
                value = max(value + af * (ep - value), prev_high, new_high)
            else:
                out[i] = value
                if new_high > ep:
                    ep = new_high
                    af = min(af + acceleration, maximum)
                value = min(value + af * (ep - value), prev_low, new_low)
        elif new_high >= value:
            is_long = True
            value = min(ep, prev_low, new_low)
            out[i] = value
            af = acceleration
            ep = new_high
            value = min(value + af * (ep - value), prev_low, new_low)
        else:
            out[i] = value
            if new_low < ep:
                ep = new_low
                af = min(af + acceleration, maximum)
            value = max(value + af * (ep - value), prev_high, new_high)
    return out


# --------------------------------------------------------------------- 退化窗口
def variance(values: np.ndarray, period: int) -> np.ndarray:
    """总体方差; 每个窗口减去窗口首值再求和, 平盘窗口恰为 0, 负的舍入残差记 0."""
    out = _nan(values.shape[0])
    if values.shape[0] < period:
        return out
    windows = sliding_window_view(values, period)
    shifted = windows - windows[:, :1]
    mean = shifted.sum(axis=1) / period
    out[period - 1 :] = np.maximum(
        (shifted * shifted).sum(axis=1) / period - mean * mean, 0.0
    )
    return out


def stddev(values: np.ndarray, period: int, nbdev: float) -> np.ndarray:
    """标准差 * nbdev."""
    return cast(np.ndarray, np.sqrt(variance(values, period)) * nbdev)


def bbands(
    values: np.ndarray, period: int, nbdevup: float, nbdevdn: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """布林带 (upper, middle, lower), 中轨为 SMA."""
    middle = _rolling(values, period, "mean")
    deviation = np.sqrt(variance(values, period))
    return middle + nbdevup * deviation, middle, middle - nbdevdn * deviation


def correl(real0: np.ndarray, real1: np.ndarray, period: int) -> np.ndarray:
    """皮尔逊相关系数; 任一序列在窗口内没有离散度时记 0, 结果夹到 [-1, 1]."""
    out = _nan(real0.shape[0])
    if real0.shape[0] < period:
        return out
    wx = sliding_window_view(real0, period)
    wy = sliding_window_view(real1, period)
    dx = wx - wx[:, :1]
    dy = wy - wy[:, :1]
    sx, sy = dx.sum(axis=1), dy.sum(axis=1)
    sxx, syy, sxy = (dx * dx).sum(axis=1), (dy * dy).sum(axis=1), (dx * dy).sum(axis=1)
    inv = 1.0 / period
    ssx = sxx - sx * sx * inv
    ssy = syy - sy * sy * inv
    spxy = sxy - sx * sy * inv
    ok = (ssx > TA_EPSILON * sxx) & (ssy > TA_EPSILON * syy) & (ssx * ssy > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.clip(spxy / np.sqrt(ssx * ssy), -1.0, 1.0)
    out[period - 1 :] = np.where(ok, value, 0.0)
    return out


def willr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Williams %R; 窗口高低价相同时记 0, 结果夹到 [-100, 0]."""
    highest = _rolling(high, period, "max")
    lowest = _rolling(low, period, "min")
    price_range = highest - lowest
    flat = _is_zero_scaled(price_range, np.abs(highest) + np.abs(lowest))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.clip((highest - close) / price_range * -100.0, -100.0, 0.0)
    out = np.where(flat, 0.0, value)
    out[: period - 1] = np.nan
    return out


def extreme_index(values: np.ndarray, period: int, find_max: bool) -> np.ndarray:
    """MAXINDEX/MININDEX: 复刻 TA-Lib 的并列规则.

    极值滑出窗口后重扫时取最早的一根(严格比较), 新来的值与当前极值相等时取
    新的一根(非严格比较)。所以并列时取哪一根取决于历史路径, 与 TA-Lib 一致。
    """
    length = values.shape[0]
    out = _nan(length)
    best_idx = -1
    best = 0.0
    for today in range(period - 1, length):
        trailing = today - period + 1
        value = float(values[today])
        if best_idx < trailing:
            best_idx = trailing
            best = float(values[trailing])
            for i in range(trailing + 1, today + 1):
                candidate = float(values[i])
                if (candidate > best) if find_max else (candidate < best):
                    best_idx = i
                    best = candidate
        elif (value >= best) if find_max else (value <= best):
            best_idx = today
            best = value
        out[today] = best_idx
    return out


# --------------------------------------------------------------------- Hilbert 变换
_HILBERT_A = 0.0962
_HILBERT_B = 0.5769
_RAD2DEG = 45.0 / math.atan(1.0)


class _HilbertLeg:
    """TA-Lib ``DO_HILBERT_ODD/EVEN`` 的一路状态."""

    def __init__(self) -> None:
        self.buffer = [0.0, 0.0, 0.0]
        self.prev = 0.0
        self.prev_input = 0.0

    def step(self, value: float, idx: int, adjusted_period: float) -> float:
        hilbert_temp = _HILBERT_A * value
        out = -self.buffer[idx]
        self.buffer[idx] = hilbert_temp
        out += hilbert_temp
        out -= self.prev
        self.prev = _HILBERT_B * self.prev_input
        out += self.prev
        self.prev_input = value
        return out * adjusted_period


class _HilbertCore:
    """四路 Hilbert(detrender/Q1/jI/jQ)各分奇偶两套, 加周期自适应."""

    def __init__(self) -> None:
        self.odd = [_HilbertLeg() for _ in range(4)]
        self.even = [_HilbertLeg() for _ in range(4)]
        self.idx = 0
        self.period = 0.0
        self.prev_i2 = 0.0
        self.prev_q2 = 0.0
        self.re = 0.0
        self.im = 0.0
        self.i1_odd_prev3 = 0.0
        self.i1_odd_prev2 = 0.0
        self.i1_even_prev3 = 0.0
        self.i1_even_prev2 = 0.0

    def step(self, smoothed: float, even: bool) -> tuple[float, float, float, float]:
        """返回 (Q1, I1, Q2, I2)."""
        adjusted = 0.075 * self.period + 0.54
        idx = self.idx
        legs = self.even if even else self.odd
        i1 = self.i1_even_prev3 if even else self.i1_odd_prev3
        detrender = legs[0].step(smoothed, idx, adjusted)
        q1 = legs[1].step(detrender, idx, adjusted)
        ji = legs[2].step(i1, idx, adjusted)
        jq = legs[3].step(q1, idx, adjusted)
        if even:
            self.idx = (self.idx + 1) % 3
            self.i1_odd_prev3 = self.i1_odd_prev2
            self.i1_odd_prev2 = detrender
        else:
            self.i1_even_prev3 = self.i1_even_prev2
            self.i1_even_prev2 = detrender
        q2 = 0.2 * (q1 + ji) + 0.8 * self.prev_q2
        i2 = 0.2 * (i1 - jq) + 0.8 * self.prev_i2
        return q1, i1, q2, i2

    def adapt_period(self, q2: float, i2: float) -> None:
        self.re = 0.2 * (i2 * self.prev_i2 + q2 * self.prev_q2) + 0.8 * self.re
        self.im = 0.2 * (i2 * self.prev_q2 - q2 * self.prev_i2) + 0.8 * self.im
        self.prev_q2 = q2
        self.prev_i2 = i2
        old = self.period
        if self.im != 0.0 and self.re != 0.0:
            self.period = 360.0 / (math.atan(self.im / self.re) * _RAD2DEG)
        self.period = min(self.period, 1.5 * old)
        self.period = max(self.period, 0.67 * old)
        self.period = min(max(self.period, 6.0), 50.0)
        self.period = 0.2 * self.period + 0.8 * old


class _PriceWma:
    """TA-Lib ``DO_PRICE_WMA``: (4,3,2,1)/10 加权的增量形式, 前 3 根只初始化."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.sub = float(values[0] + values[1] + values[2])
        self.sum = float(values[0] + values[1] * 2.0 + values[2] * 3.0)
        self.trailing_idx = 0
        self.trailing_value = 0.0

    def step(self, value: float) -> float:
        self.sub += value
        self.sub -= self.trailing_value
        self.sum += value * 4.0
        self.trailing_value = float(self.values[self.trailing_idx])
        self.trailing_idx += 1
        out = self.sum * 0.1
        self.sum -= self.sub
        return out


def mama(
    values: np.ndarray, fast_limit: float, slow_limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """MESA 自适应均线(MAMA, FAMA), 首个输出在下标 32."""
    length = values.shape[0]
    out_mama = _nan(length)
    out_fama = _nan(length)
    lookback = 32
    if length <= lookback:
        return out_mama, out_fama
    wma = _PriceWma(values)
    today = 3
    for _ in range(9):
        wma.step(float(values[today]))
        today += 1
    core = _HilbertCore()
    mama_value = 0.0
    fama_value = 0.0
    prev_phase = 0.0
    while today < length:
        value = float(values[today])
        q1, i1, q2, i2 = core.step(wma.step(value), today % 2 == 0)
        phase = math.atan(q1 / i1) * _RAD2DEG if i1 != 0.0 else 0.0
        delta = max(prev_phase - phase, 1.0)
        prev_phase = phase
        alpha = max(fast_limit / delta, slow_limit) if delta > 1.0 else fast_limit
        mama_value = alpha * value + (1.0 - alpha) * mama_value
        alpha *= 0.5
        fama_value = alpha * mama_value + (1.0 - alpha) * fama_value
        if today >= lookback:
            out_mama[today] = mama_value
            out_fama[today] = fama_value
        core.adapt_period(q2, i2)
        today += 1
    return out_mama, out_fama


def ht_trendline(values: np.ndarray) -> np.ndarray:
    """Hilbert 瞬时趋势线, 首个输出在下标 63."""
    length = values.shape[0]
    out = _nan(length)
    lookback = 63
    if length <= lookback:
        return out
    wma = _PriceWma(values)
    today = 3
    for _ in range(34):
        wma.step(float(values[today]))
        today += 1
    core = _HilbertCore()
    smooth_period = 0.0
    trend1 = trend2 = trend3 = 0.0
    while today < length:
        _, _, q2, i2 = core.step(wma.step(float(values[today])), today % 2 == 0)
        core.adapt_period(q2, i2)
        smooth_period = 0.33 * core.period + 0.67 * smooth_period
        cycle = int(smooth_period + 0.5)
        total = 0.0
        for k in range(cycle):
            total += float(values[today - k])
        if cycle > 0:
            total /= cycle
        trendline = (4.0 * total + 3.0 * trend1 + 2.0 * trend2 + trend3) / 10.0
        trend3, trend2, trend1 = trend2, trend1, total
        if today >= lookback:
            out[today] = trendline
        today += 1
    return out
