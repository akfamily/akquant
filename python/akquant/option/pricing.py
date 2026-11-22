"""期权定价: Black-76 欧式定价与 Greeks、BAW 美式近似、隐含波动率、隐含远期.

所有定价都以远期价格 ``forward`` 为输入, 商品期权、股指期权、ETF 期权共用一套公式,
区别只在远期从哪里来:

- 商品期权: 标的期货价格;
- 股指 / ETF 期权: 用同一到期的认购、认沽价格按平价关系反推(:func:`implied_forward`)。

参数均支持标量或可广播的数组; 全部输入为标量时返回 ``float``, 否则返回 ``ndarray``。
``option_type`` 接受 ``"CALL"`` / ``"PUT"`` / ``"C"`` / ``"P"`` (不区分大小写)。
Black-76 的定价、Greeks 与隐含波动率在 Rust 中批量计算, 数组较长时多线程。
"""

import datetime as dt
import math
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np

from ..akquant import (
    _option_black76_greeks,
    _option_black76_implied_vol,
    _option_black76_price,
)

ArrayLike = Union[float, Any]
FloatOrArray = Union[float, np.ndarray]

_SECONDS_PER_YEAR = 365.0 * 86400.0
#: 隐含波动率求解区间
_SIGMA_LOW = 1e-6
_SIGMA_HIGH = 5.0
#: 价格容差: 时间价值不超过它时, 价格中没有可分辨的波动率信息, 视为无解
_IV_TOL = 1e-8
#: 收敛判据: 波动率步长或区间宽度小于它
_SIGMA_TOL = 1e-10
_IV_MAX_ITER = 100


def _is_call(option_type: Any) -> np.ndarray:
    values = np.asarray(option_type, dtype=str)
    # 取值只有寥寥几种: 先去重再逐个判定, 避免对每个元素做字符串处理
    unique, inverse = np.unique(values, return_inverse=True)
    normalized = [str(v).strip().upper() for v in unique]
    unknown = sorted({v for v in normalized if v not in ("C", "CALL", "P", "PUT")})
    if unknown:
        raise ValueError(f"option_type 只接受 CALL/PUT/C/P, 实际: {unknown}")
    flags = np.array([v in ("C", "CALL") for v in normalized], dtype=bool)
    return flags[inverse].reshape(values.shape)


def _flatten(
    values: Tuple[Any, ...], option_type: Any
) -> Tuple[Tuple[int, ...], List[np.ndarray], np.ndarray]:
    """广播后展平为连续一维数组, 供 Rust 批量计算; 返回 (原形状, 浮点数组, is_call)."""
    arrays = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in values), _is_call(option_type)
    )
    floats = [np.ascontiguousarray(a, dtype=float).ravel() for a in arrays[:-1]]
    calls = np.ascontiguousarray(arrays[-1], dtype=bool).ravel()
    return arrays[0].shape, floats, calls


def _all_scalar(*args: Any) -> bool:
    return all(np.ndim(a) == 0 for a in args)


def _out(flat: np.ndarray, shape: Tuple[int, ...], scalar: bool) -> FloatOrArray:
    value = np.asarray(flat, dtype=float).reshape(shape)
    return float(value) if scalar else value


def black76_price(
    forward: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
    option_type: Any,
) -> FloatOrArray:
    """
    Black-76 欧式期权价格.

    :param forward: 远期价格(商品期权取标的期货价格)
    :param strike: 行权价
    :param t: 剩余期限(年), 可用 :func:`year_fraction` 计算
    :param rate: 无风险利率(连续复利, 年化)
    :param sigma: 波动率(年化, 0.2 表示 20%)
    :param option_type: CALL / PUT
    :return: 期权价格; ``t<=0`` 或 ``sigma<=0`` 时为贴现内在价值
    """
    scalar = _all_scalar(forward, strike, t, rate, sigma, option_type)
    shape, (f, k, tt, r, s), c = _flatten(
        (forward, strike, t, rate, sigma), option_type
    )
    return _out(_option_black76_price(f, k, tt, r, s, c), shape, scalar)


def black76_greeks(
    forward: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
    option_type: Any,
) -> Dict[str, FloatOrArray]:
    """
    Black-76 Greeks(对远期价格求导).

    单位均为原始偏导数, 不做按 1% 或按天的缩放:

    - ``delta`` = dV/dF, ``gamma`` = d²V/dF²
    - ``vega`` = dV/dσ (σ 变化 1.0 即 100 个波动率点; 按 1% 计请除以 100)
    - ``theta`` = dV/dt, 按自然时间流逝、单位为每年(按天计请除以 365)
    - ``rho`` = dV/dr

    ``t<=0`` 或 ``sigma<=0`` 时只给出到期时的 delta(实值 ±贴现因子, 否则 0),
    其余为 0。

    :return: ``{"delta", "gamma", "vega", "theta", "rho"}``
    """
    scalar = _all_scalar(forward, strike, t, rate, sigma, option_type)
    shape, (f, k, tt, r, s), c = _flatten(
        (forward, strike, t, rate, sigma), option_type
    )
    columns = _option_black76_greeks(f, k, tt, r, s, c)
    names = ("delta", "gamma", "vega", "theta", "rho")
    return {name: _out(col, shape, scalar) for name, col in zip(names, columns)}


def _black76_scalar(
    f: float, k: float, t: float, r: float, s: float, call: bool
) -> float:
    def one(x: float) -> np.ndarray:
        return np.array([x], dtype=float)

    out = _option_black76_price(
        one(f), one(k), one(t), one(r), one(s), np.array([call])
    )
    return float(out[0])


def _baw_scalar(f: float, k: float, t: float, r: float, s: float, call: bool) -> float:
    def b76(x: float) -> float:
        return _black76_scalar(x, k, t, r, s, call)

    euro = b76(f)
    intrinsic = max(f - k, 0.0) if call else max(k - f, 0.0)
    # r<=0 时期货期权提前行权没有价值; 到期或零波动时价值即内在价值
    if t <= 0 or s <= 0 or r <= 0:
        return max(euro, intrinsic) if t > 0 else intrinsic

    sd = s * math.sqrt(t)
    disc = math.exp(-r * t)
    m = 2.0 * r / (s * s)
    kq = 1.0 - disc

    def n_cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    def n_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def d1_of(x: float) -> float:
        return (math.log(x / k) + 0.5 * sd * sd) / sd

    # 临界价格用 Barone-Adesi & Whaley (1987) 的牛顿迭代(b=0 的期货期权情形)
    if call:
        q = (1.0 + math.sqrt(1.0 + 4.0 * m / kq)) / 2.0
        q_inf = (1.0 + math.sqrt(1.0 + 4.0 * m)) / 2.0
        s_inf = k / (1.0 - 1.0 / q_inf)
        si = k + (s_inf - k) * (1.0 - math.exp(-2.0 * sd * k / (s_inf - k)))
        for _ in range(100):
            d1 = d1_of(si)
            rhs = b76(si) + (1.0 - disc * n_cdf(d1)) * si / q
            if abs(si - k - rhs) / k < 1e-8:
                break
            bi = disc * n_cdf(d1) * (1.0 - 1.0 / q) + (1.0 - disc * n_pdf(d1) / sd) / q
            si = (k + rhs - bi * si) / (1.0 - bi)
        if f >= si:
            return f - k
        a = si / q * (1.0 - disc * n_cdf(d1_of(si)))
        return euro + a * float((f / si) ** q)

    q = (1.0 - math.sqrt(1.0 + 4.0 * m / kq)) / 2.0
    q_inf = (1.0 - math.sqrt(1.0 + 4.0 * m)) / 2.0
    s_inf = k / (1.0 - 1.0 / q_inf)
    si = s_inf + (k - s_inf) * math.exp(-2.0 * sd * k / (k - s_inf))
    for _ in range(100):
        d1 = d1_of(si)
        rhs = b76(si) - (1.0 - disc * n_cdf(-d1)) * si / q
        if abs(k - si - rhs) / k < 1e-8:
            break
        bi = -disc * n_cdf(-d1) * (1.0 - 1.0 / q) - (1.0 + disc * n_pdf(-d1) / sd) / q
        si = (k - rhs + bi * si) / (1.0 + bi)
    if f <= si:
        return k - f
    a = -si / q * (1.0 - disc * n_cdf(-d1_of(si)))
    return euro + a * float((f / si) ** q)


_baw_vec = np.vectorize(_baw_scalar, otypes=[float])


def baw_price(
    forward: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    rate: ArrayLike,
    sigma: ArrayLike,
    option_type: Any,
) -> FloatOrArray:
    """
    美式期货期权价格(Barone-Adesi-Whaley 近似).

    适用于美式商品期权。参数含义同 :func:`black76_price`; 结果不低于同参数的
    Black-76 价格与内在价值。逐个合约在 Python 中计算, 远慢于 Black-76。
    """
    scalar = _all_scalar(forward, strike, t, rate, sigma, option_type)
    shape, (f, k, tt, r, s), c = _flatten(
        (forward, strike, t, rate, sigma), option_type
    )
    return _out(_baw_vec(f, k, tt, r, s, c), shape, scalar)


def _baw_implied_vol(
    p: np.ndarray,
    f: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """BAW 隐含波动率: 二分法, 无解判定与 Black-76 的 Rust 求解器一致."""
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero((t > 0) & (f > 0) & (k > 0) & np.isfinite(p))
    p, f, k, t, r, c = (a[idx] for a in (p, f, k, t, r, c))
    lo = np.full(p.shape, _SIGMA_LOW)
    hi = np.full(p.shape, _SIGMA_HIGH)
    solvable = (_baw_vec(f, k, t, r, lo, c) < p - tol) & (
        _baw_vec(f, k, t, r, hi, c) > p
    )
    for _ in range(max_iter):
        if (hi - lo < _SIGMA_TOL).all():
            break
        mid = 0.5 * (lo + hi)
        above = _baw_vec(f, k, t, r, mid, c) > p
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    converged = solvable & (hi - lo < _SIGMA_TOL)
    out[idx] = np.where(converged, 0.5 * (lo + hi), np.nan)
    return out


def implied_vol(
    price: ArrayLike,
    forward: ArrayLike,
    strike: ArrayLike,
    t: ArrayLike,
    rate: ArrayLike,
    option_type: Any,
    model: Literal["black76", "baw"] = "black76",
    tol: float = _IV_TOL,
    max_iter: int = _IV_MAX_ITER,
) -> FloatOrArray:
    """
    由期权价格反推隐含波动率.

    在 ``[1e-6, 5.0]`` 内求解, 波动率步长小于 1e-10 即收敛: Black-76 用带区间保护的
    牛顿法(Rust 批量计算, 数组较长时多线程), BAW 用二分法(Python, 较慢)。以下情况
    返回 ``nan``: ``t<=0``、远期或行权价不为正、时间价值(价格减去波动率 1e-6 时的
    价格)不超过 ``tol``、价格不低于波动率 500% 时的价格、``max_iter`` 次内不收敛。

    :param price: 期权价格
    :param model: ``"black76"`` (欧式, 默认) 或 ``"baw"`` (美式商品期权)
    :param tol: 价格容差, 时间价值不超过它视为无解
    :param max_iter: 最大迭代次数
    :return: 隐含波动率(年化)
    """
    if model not in ("black76", "baw"):
        raise ValueError(f"model 只接受 black76/baw, 实际: {model!r}")
    scalar = _all_scalar(price, forward, strike, t, rate, option_type)
    shape, (p, f, k, tt, r), c = _flatten(
        (price, forward, strike, t, rate), option_type
    )
    if model == "black76":
        flat = _option_black76_implied_vol(
            p, f, k, tt, r, c, tol, _SIGMA_TOL, _SIGMA_LOW, _SIGMA_HIGH, max_iter
        )
    else:
        flat = _baw_implied_vol(p, f, k, tt, r, c, tol, max_iter)
    return _out(flat, shape, scalar)


def implied_forward(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    put_prices: ArrayLike,
    t: float,
    rate: float,
) -> float:
    """
    按认购-认沽平价从同一到期的期权报价反推远期价格.

    取认购、认沽价差绝对值最小(最接近平值)的行权价 K, 远期
    ``F = K + exp(r*t) * (C - P)``。用于股指 / ETF 期权, 远期中已含基差与分红。

    :param strikes: 行权价
    :param call_prices: 各行权价的认购价格
    :param put_prices: 各行权价的认沽价格
    :param t: 剩余期限(年)
    :param rate: 无风险利率
    :return: 远期价格
    :raises ValueError: 没有任何一对有效的认购、认沽价格
    """
    k, c, p = np.broadcast_arrays(
        np.asarray(strikes, dtype=float),
        np.asarray(call_prices, dtype=float),
        np.asarray(put_prices, dtype=float),
    )
    mask = np.isfinite(k) & np.isfinite(c) & np.isfinite(p)
    if not mask.any():
        raise ValueError("implied_forward 需要至少一对有效的认购、认沽价格")
    k, c, p = k[mask], c[mask], p[mask]
    idx = int(np.argmin(np.abs(c - p)))
    return float(k[idx] + math.exp(rate * t) * (c[idx] - p[idx]))


def year_fraction(start: dt.datetime, end: dt.datetime) -> float:
    """
    两个时刻之间的年化期限, 按自然日 365 天计, 不足 0 时取 0.

    :param start: 起始时刻(通常为当前时间)
    :param end: 结束时刻(通常为到期时刻)
    :return: 年化期限
    """
    return max((end - start).total_seconds(), 0.0) / _SECONDS_PER_YEAR
