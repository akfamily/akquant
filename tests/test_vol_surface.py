"""波动率微笑与曲面拟合: SVI / SABR 参数还原、套利检查、曲面插值、策略接口."""

import math
from typing import Any, Dict, List, Literal, Tuple

import akquant
import numpy as np
import pandas as pd
import pytest
from akquant import BarBatch, Strategy
from akquant.akquant import _sabr_vol
from akquant.option import VolSmile, black76_price, fit_smile, fit_surface
from akquant.option.surface import _butterfly_min_g

SVI = {"a": 0.004, "b": 0.05, "rho": -0.35, "m": 0.03, "sigma": 0.12}


def _svi_vol(strikes: np.ndarray, forward: float, t: float, p: Dict[str, float]) -> Any:
    k = np.log(strikes / forward)
    w = p["a"] + p["b"] * (
        p["rho"] * (k - p["m"]) + np.sqrt((k - p["m"]) ** 2 + p["sigma"] ** 2)
    )
    return np.sqrt(w / t)


def test_fit_smile_svi_recovers_parameters() -> None:
    """无噪声数据: SVI 参数还原, 误差趋于 0, 没有蝶式套利."""
    forward, t = 3000.0, 0.25
    strikes = np.arange(2400.0, 3601.0, 50.0)
    smile = fit_smile(strikes, _svi_vol(strikes, forward, t, SVI), forward, t)
    assert smile.model == "svi" and smile.converged
    for name, value in SVI.items():
        assert smile.params[name] == pytest.approx(value, abs=1e-6)
    assert smile.rmse < 1e-8
    assert not smile.butterfly_arbitrage
    assert smile.vol(3000.0) == pytest.approx(
        _svi_vol(np.array([3000.0]), forward, t, SVI)[0]
    )
    assert np.asarray(smile.vol(strikes)).shape == strikes.shape


def test_fit_smile_sabr_recovers_parameters() -> None:
    """无噪声数据: SABR(β=0.5) 参数还原."""
    forward, t, beta = 500.0, 0.4, 0.5
    alpha, rho, nu = 5.0, -0.3, 0.8
    strikes = np.arange(380.0, 641.0, 20.0)
    n = len(strikes)
    vols = _sabr_vol(
        np.full(n, forward),
        strikes,
        np.full(n, t),
        np.full(n, alpha),
        beta,
        np.full(n, rho),
        np.full(n, nu),
    )
    smile = fit_smile(strikes, vols, forward, t, model="sabr", beta=beta)
    assert smile.params["alpha"] == pytest.approx(alpha, rel=1e-5)
    assert smile.params["rho"] == pytest.approx(rho, abs=1e-5)
    assert smile.params["nu"] == pytest.approx(nu, rel=1e-5)
    assert smile.params["beta"] == beta
    assert smile.rmse < 1e-8


def test_butterfly_check_flags_vogt_example() -> None:
    """Gatheral-Jacquier(2014) 引用的 Vogt 参数存在蝶式套利, g(k) 最小值为负."""
    vogt = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}
    smile = VolSmile(
        model="svi",
        forward=1.0,
        t=1.0,
        params=vogt,
        rmse=0.0,
        n_points=0,
        converged=True,
        k_range=(-1.5, 1.5),
        butterfly_min_g=math.nan,
    )
    assert _butterfly_min_g(smile) < 0


def _chain(expiries: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """按 SVI 参数生成期权链: 虚值一侧为真值, 实值一侧填错误的 iv(应被忽略)."""
    rows: List[Dict[str, Any]] = []
    for expiry, spec in expiries.items():
        forward, t = spec["forward"], spec["t"]
        for strike in spec["strikes"]:
            vol = float(_svi_vol(np.array([strike]), forward, t, spec["svi"])[0])
            for kind in ("CALL", "PUT"):
                otm = (kind == "PUT") == (strike < forward)
                rows.append(
                    {
                        "symbol": f"x{expiry}{kind[0]}{strike:g}",
                        "option_type": kind,
                        "strike": strike,
                        "expiry_date": expiry,
                        "t": t,
                        "forward": forward,
                        "iv": vol if otm else 2.0,
                    }
                )
    return pd.DataFrame(rows)


NEAR = {"a": 0.002, "b": 0.04, "rho": -0.3, "m": 0.0, "sigma": 0.1}
FAR = {"a": 0.006, "b": 0.06, "rho": -0.3, "m": 0.0, "sigma": 0.15}


def test_fit_surface_selects_otm_and_interpolates_total_variance() -> None:
    """只用虚值一侧; 期限间按总方差线性插值; 报价不足的到期日记入 failures."""
    strikes = np.arange(2600.0, 3401.0, 50.0)
    chain = _chain(
        {
            20260417: {"forward": 3000.0, "t": 0.1, "strikes": strikes, "svi": NEAR},
            20260619: {"forward": 3050.0, "t": 0.3, "strikes": strikes, "svi": FAR},
            20260918: {
                "forward": 3100.0,
                "t": 0.55,
                "strikes": [2900.0, 3000.0, 3200.0],
                "svi": FAR,
            },
        }
    )
    chain.loc[chain["symbol"] == "x20260417P2600", "iv"] = np.nan
    surface = fit_surface(chain)
    assert surface.expiries == [20260417, 20260619]
    assert surface.failures == {
        20260918: "有隐含波动率的虚值合约 3 个, 少于 svi 拟合所需的 5 个"
    }
    near = surface.smile(20260417)
    assert near.n_points == len(strikes) - 1
    assert near.rmse < 1e-8
    assert surface.calendar_arbitrage == []
    # t 在两个到期日之间: k = ln(K/F(t)), w 按期限线性插值
    t, strike = 0.2, 3100.0
    forward = 3000.0 + 0.5 * 50.0
    k = math.log(strike / forward)
    w_near = (
        float(_svi_vol(np.array([3000.0 * math.exp(k)]), 3000.0, 0.1, NEAR)[0]) ** 2
        * 0.1
    )
    w_far = (
        float(_svi_vol(np.array([3050.0 * math.exp(k)]), 3050.0, 0.3, FAR)[0]) ** 2
        * 0.3
    )
    assert surface.vol(strike, t) == pytest.approx(
        math.sqrt(0.5 * (w_near + w_far) / t)
    )
    assert surface.vol(strike, 0.1) == pytest.approx(near.vol(strike))
    with pytest.raises(
        ValueError, match=r"^t=0.6 超出已拟合到期日的期限范围 \[0.1, 0.3\], 不外推$"
    ):
        surface.vol(strike, 0.6)
    with pytest.raises(
        ValueError, match="^到期日 20260918 未能拟合: 有隐含波动率的虚值合约 3 个"
    ):
        surface.smile(20260918)


def test_fit_surface_flags_calendar_arbitrage() -> None:
    """远月总方差低于近月时记为日历套利."""
    strikes = np.arange(2600.0, 3401.0, 50.0)
    chain = _chain(
        {
            20260417: {"forward": 3000.0, "t": 0.1, "strikes": strikes, "svi": FAR},
            20260619: {"forward": 3000.0, "t": 0.3, "strikes": strikes, "svi": NEAR},
        }
    )
    assert fit_surface(chain).calendar_arbitrage == [(20260417, 20260619)]


def test_fit_smile_rejects_invalid_input() -> None:
    """参数错误报出确切原因."""
    strikes, ivs = [90.0, 95.0, 100.0, 105.0, 110.0], [0.3, 0.28, 0.27, 0.28, 0.3]
    cases: List[Tuple[Dict[str, Any], str]] = [
        (dict(model="heston"), "model 只接受 svi/sabr, 实际: 'heston'"),
        (dict(model="sabr", beta=1.5), "SABR 的 beta 须在 [0, 1] 内, 实际: 1.5"),
        (dict(strikes=strikes[:4]), "strikes、ivs、weights 长度须相同, 实际 4、5、4"),
        (dict(forward=0.0), "forward 须为正数, 实际: 0.0"),
        (
            dict(ivs=[0.3, 0.28, math.nan, 0.28, 0.3]),
            "ivs 须全部为正数(先剔除算不出隐含波动率的点)",
        ),
        (dict(strikes=[90.0, 90.0, 100.0, 105.0, 110.0]), "strikes 不能重复"),
        (
            dict(strikes=strikes[:4], ivs=ivs[:4]),
            "svi 拟合至少需要 5 个报价, 实际 4 个",
        ),
    ]
    for override, message in cases:
        kwargs: Dict[str, Any] = dict(strikes=strikes, ivs=ivs, forward=100.0, t=0.5)
        kwargs.update(override)
        with pytest.raises(ValueError) as exc:
            fit_smile(**kwargs)
        assert str(exc.value) == message


UNDERLYING, EXPIRY = "m2605", 20260512
STRIKES = [2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0]
DAYS = pd.date_range("2026-03-02 15:00", periods=3, freq="B", tz="Asia/Shanghai")


class _Surface(Strategy):
    """收盘时拟合曲面, 记录参数与同一批期权链是否复用拟合结果."""

    option_refresh_interval = 3600.0

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Any] = []

    def on_bars(self, batch: BarBatch) -> None:
        first = self.vol_surface(UNDERLYING)
        self.records.append((first, first is self.vol_surface(UNDERLYING)))


def test_strategy_vol_surface_from_option_chain() -> None:
    """期权链按 SVI 定价: 策略拟合出原参数, 同一批期权链复用同一个曲面."""
    forward = 3000.0
    t = akquant.option.year_fraction(
        DAYS[0].to_pydatetime(),
        pd.Timestamp("2026-05-12 15:00", tz="Asia/Shanghai").to_pydatetime(),
    )
    data = {UNDERLYING: forward}
    configs = [
        akquant.InstrumentConfig(
            symbol=UNDERLYING, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
        )
    ]
    for strike in STRIKES:
        kind: Literal["CALL", "PUT"] = "PUT" if strike < forward else "CALL"
        symbol = f"m2605-{kind[0]}-{strike:g}"
        vol = float(_svi_vol(np.array([strike]), forward, t, SVI)[0])
        data[symbol] = float(black76_price(forward, strike, t, 0.0, vol, kind))
        configs.append(
            akquant.InstrumentConfig(
                symbol=symbol,
                asset_type="OPTION",
                multiplier=10.0,
                option_type=kind,
                strike_price=strike,
                expiry_date=EXPIRY,
                underlying_symbol=UNDERLYING,
            )
        )
    frames = {
        s: pd.DataFrame(
            {
                "date": DAYS[:1],
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": 100,
                "symbol": s,
            }
        )
        for s, p in data.items()
    }
    strategy = _Surface()
    akquant.run_backtest(
        data=frames,
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=configs,
        ),
        freq="1d",
        show_progress=False,
    )
    surface, reused = strategy.records[0]
    assert reused
    smile = surface.smile(EXPIRY)
    assert smile.n_points == len(STRIKES)
    assert smile.rmse < 1e-6
    assert smile.vol(2950.0) == pytest.approx(
        float(_svi_vol(np.array([2950.0]), forward, t, SVI)[0]), abs=1e-6
    )


def test_fit_surface_rejects_mixed_forwards_in_one_expiry() -> None:
    """同一到期日出现多个远期价格(多个标的同一到期日)时报错."""
    strikes = np.arange(2600.0, 3401.0, 50.0)
    chain = _chain(
        {20260417: {"forward": 3000.0, "t": 0.1, "strikes": strikes, "svi": NEAR}}
    )
    chain.loc[chain.index[:4], "forward"] = 3100.0
    with pytest.raises(ValueError) as exc:
        fit_surface(chain)
    assert str(exc.value) == (
        "到期日 20260417 的合约远期价格不一致(2 个), 同一到期日的多个标的须分开拟合"
    )
