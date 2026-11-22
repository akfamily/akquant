"""akquant.option: Black-76 / BAW 定价、Greeks、隐含波动率、隐含远期、代码解析."""

import datetime as dt
import math

import numpy as np
import pytest
from akquant.option import (
    baw_price,
    black76_greeks,
    black76_price,
    implied_forward,
    implied_vol,
    parse_option_symbol,
    year_fraction,
)

F, K, T, R, SIGMA = 3000.0, 3100.0, 0.25, 0.02, 0.22


def _crr_american(
    f: float, k: float, t: float, r: float, s: float, call: bool, n: int = 2000
) -> float:
    """美式期货期权二叉树(CRR), 作为 BAW 近似的对照."""
    dt_ = t / n
    u = math.exp(s * math.sqrt(dt_))
    d = 1.0 / u
    p = (1.0 - d) / (u - d)
    disc = math.exp(-r * dt_)
    j = np.arange(n + 1)
    fut = f * u ** (n - j) * d**j
    value = np.maximum(fut - k, 0.0) if call else np.maximum(k - fut, 0.0)
    for _ in range(n):
        fut = fut[:-1] / u
        value = disc * (p * value[:-1] + (1.0 - p) * value[1:])
        value = np.maximum(value, fut - k if call else k - fut)
    return float(value[0])


def test_black76_matches_hull_example() -> None:
    """Hull 期货期权例题: F=K=20, r=9%, T=4 个月, σ=25% 的欧式认沽约 1.12."""
    assert black76_price(20, 20, 4 / 12, 0.09, 0.25, "PUT") == pytest.approx(
        1.12, abs=0.005
    )


def test_black76_put_call_parity() -> None:
    """C - P = exp(-rT) (F - K)."""
    call = black76_price(F, K, T, R, SIGMA, "C")
    put = black76_price(F, K, T, R, SIGMA, "P")
    assert call - put == pytest.approx(math.exp(-R * T) * (F - K), abs=1e-9)


@pytest.mark.parametrize("kind", ["CALL", "PUT"])
def test_black76_greeks_match_finite_differences(kind: str) -> None:
    """Greeks 与有限差分一致(theta 为自然时间流逝方向)."""
    g = black76_greeks(F, K, T, R, SIGMA, kind)
    h = 1e-4

    def px(**kw: float) -> float:
        args = dict(forward=F, strike=K, t=T, rate=R, sigma=SIGMA)
        args.update(kw)
        return float(black76_price(option_type=kind, **args))

    assert g["delta"] == pytest.approx(
        (px(forward=F + h) - px(forward=F - h)) / (2 * h), rel=1e-6
    )
    assert g["gamma"] == pytest.approx(
        (px(forward=F + 1) - 2 * px() + px(forward=F - 1)), rel=1e-4
    )
    assert g["vega"] == pytest.approx(
        (px(sigma=SIGMA + h) - px(sigma=SIGMA - h)) / (2 * h), rel=1e-6
    )
    assert g["theta"] == pytest.approx(-(px(t=T + h) - px(t=T - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx(
        (px(rate=R + h) - px(rate=R - h)) / (2 * h), rel=1e-6
    )


def test_scalar_inputs_return_float_and_arrays_broadcast() -> None:
    """全标量输入返回 float; 数组输入按广播返回 ndarray."""
    assert isinstance(black76_price(F, K, T, R, SIGMA, "C"), float)
    out = black76_price(F, [2900.0, 3100.0], T, R, SIGMA, ["P", "C"])
    assert isinstance(out, np.ndarray) and out.shape == (2,)


def test_expired_or_zero_vol_prices_at_intrinsic() -> None:
    """t<=0 时为内在价值, delta 实值为 ±1、虚值为 0."""
    assert black76_price(3200.0, K, 0.0, R, SIGMA, "C") == pytest.approx(100.0)
    assert black76_greeks(3200.0, K, 0.0, R, SIGMA, "C")["delta"] == 1.0
    assert black76_greeks(3200.0, K, 0.0, R, SIGMA, "P")["delta"] == 0.0


def test_invalid_option_type_raises() -> None:
    """option_type 只接受 CALL/PUT/C/P."""
    with pytest.raises(ValueError, match="option_type"):
        black76_price(F, K, T, R, SIGMA, "X")


def test_implied_vol_round_trip_over_chain() -> None:
    """虚值一侧报价反推的隐含波动率还原输入波动率."""
    strikes = np.linspace(2400.0, 3600.0, 25)
    kinds = np.where(strikes < F, "PUT", "CALL")
    prices = black76_price(F, strikes, T, R, 0.25, kinds)
    iv = implied_vol(prices, F, strikes, T, R, kinds)
    assert np.allclose(iv, 0.25, atol=1e-8)


def test_implied_vol_nan_outside_arbitrage_bounds() -> None:
    """价格不高于内在价值、已到期时返回 nan."""
    intrinsic = math.exp(-R * T) * (3200.0 - K)
    assert math.isnan(implied_vol(intrinsic - 1.0, 3200.0, K, T, R, "C"))
    assert math.isnan(implied_vol(50.0, F, K, 0.0, R, "C"))


@pytest.mark.parametrize(
    "f, k, t, r, s, call",
    [
        (100.0, 100.0, 0.5, 0.08, 0.3, True),
        (100.0, 120.0, 0.5, 0.08, 0.3, False),
        (100.0, 90.0, 1.0, 0.10, 0.25, True),
        (3000.0, 3100.0, 0.25, 0.02, 0.22, False),
    ],
)
def test_baw_close_to_binomial_tree(
    f: float, k: float, t: float, r: float, s: float, call: bool
) -> None:
    """BAW 与美式二叉树相差 3% 以内, 且不低于欧式价格与内在价值."""
    kind = "C" if call else "P"
    price = baw_price(f, k, t, r, s, kind)
    assert price == pytest.approx(_crr_american(f, k, t, r, s, call), rel=0.03)
    assert price >= black76_price(f, k, t, r, s, kind) - 1e-12
    assert price >= max(f - k if call else k - f, 0.0)


def test_baw_equals_black76_without_rate() -> None:
    """r=0 时期货期权提前行权无价值, BAW 等于 Black-76."""
    assert baw_price(F, K, T, 0.0, SIGMA, "P") == pytest.approx(
        black76_price(F, K, T, 0.0, SIGMA, "P")
    )


def test_implied_vol_baw_round_trip() -> None:
    """model='baw' 时由美式价格还原波动率."""
    strikes = np.array([2800.0, 3000.0, 3200.0])
    prices = baw_price(F, strikes, T, 0.05, 0.25, "P")
    iv = implied_vol(prices, F, strikes, T, 0.05, "P", model="baw")
    assert np.allclose(iv, 0.25, atol=1e-6)


def test_implied_forward_recovers_forward() -> None:
    """由同到期认购、认沽报价按平价还原远期."""
    strikes = np.arange(3800.0, 4201.0, 50.0)
    forward = 3987.5
    calls = black76_price(forward, strikes, T, R, 0.2, "C")
    puts = black76_price(forward, strikes, T, R, 0.2, "P")
    assert implied_forward(strikes, calls, puts, T, R) == pytest.approx(forward)


def test_implied_forward_requires_a_pair() -> None:
    """没有有效报价对时报错."""
    with pytest.raises(ValueError):
        implied_forward([3000.0], [math.nan], [10.0], T, R)


def test_year_fraction_calendar_days() -> None:
    """按自然日 365 天计, 已过期取 0."""
    start = dt.datetime(2026, 1, 1, 15)
    assert year_fraction(start, dt.datetime(2027, 1, 1, 15)) == pytest.approx(1.0)
    assert year_fraction(start, dt.datetime(2025, 12, 1)) == 0.0


@pytest.mark.parametrize(
    "symbol, underlying, kind, strike",
    [
        ("m2701-P-3100", "m2701", "PUT", 3100.0),  # 大商所
        ("m2703-MS-C-3300", "m2703", "CALL", 3300.0),  # 大商所月度期权
        ("SR701C5700", "SR701", "CALL", 5700.0),  # 郑商所
        ("RM703MSP2250", "RM703", "PUT", 2250.0),  # 郑商所月度期权
        ("cu2612P94000", "cu2612", "PUT", 94000.0),  # 上期所
        ("sc2701C660", "sc2701", "CALL", 660.0),  # 能源中心
        ("lc2701-C-154000", "lc2701", "CALL", 154000.0),  # 广期所
        ("IO2612-C-4000", "IO2612", "CALL", 4000.0),  # 中金所
    ],
)
def test_parse_option_symbol(
    symbol: str, underlying: str, kind: str, strike: float
) -> None:
    """各交易所期权代码格式(样例取自 CTP 合约查询)."""
    parsed = parse_option_symbol(symbol)
    assert parsed is not None
    assert (parsed.underlying, parsed.option_type, parsed.strike) == (
        underlying,
        kind,
        strike,
    )


@pytest.mark.parametrize("symbol", ["10007000", "rb2701", "IF2612", ""])
def test_parse_option_symbol_rejects_non_options(symbol: str) -> None:
    """ETF 期权数字代码与期货代码返回 None."""
    assert parse_option_symbol(symbol) is None
