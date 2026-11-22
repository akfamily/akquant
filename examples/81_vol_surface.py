#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
波动率曲面拟合：SVI / SABR 微笑、期限插值与相对价值.

演示策略内的 ``vol_surface()``：由期权链按到期日拟合波动率微笑，组成曲面。

- 商品期权各月份标的不同（豆粕 1 月期权的标的是 ``m2401``、5 月期权是
  ``m2405``），把多个标的传给 ``vol_surface([...])`` 合成一个曲面；
- ``surface.smile(到期日)``：单个到期日的微笑，含参数、拟合误差、蝶式套利检查；
- ``surface.vol(行权价, 期限)``：两个到期日之间按总方差插值，得到固定期限的
  波动率（如 60 天平值波动率）；
- 市场隐含波动率减去拟合值，找出相对微笑偏贵、偏便宜的合约。

数据是本地合成的，不需要联网：两个月份的期货随机游走，各行权价的隐含波动率按
SVI 微笑生成后叠加 ±1 个波动率点的随机噪声（模拟成交价噪声），再按 Black-76 定价。
日线打 15:00 戳，与期权链“到期日 15:00 到期”的期限口径一致。

本例不下单。实盘中成交稀少的期权最新价可能过时，拟合宜把类属性
``option_price_source`` 设为 ``"mid"``（按买卖中间价求隐含波动率）。

运行方式::

    python examples/81_vol_surface.py

输出：每周五两个到期日 90%/平值/110% 行权价处的波动率与拟合误差、60 天平值波动率；
最后一天 SVI 与 SABR 的对比，以及相对微笑最贵、最便宜的合约。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from akquant import (
    BacktestConfig,
    BarBatch,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.option import black76_price

#: 标的 -> (期权到期日, 期货起始价, SVI 参数(总方差口径, 以期限 1 年计))
SERIES: Dict[str, Any] = {
    "m2401": (20231207, 3000.0, dict(a=0.02, b=0.10, rho=-0.3, m=0.02, sigma=0.15)),
    "m2405": (20240408, 2950.0, dict(a=0.03, b=0.09, rho=-0.2, m=0.03, sigma=0.20)),
}
STRIKES = list(range(2200, 3801, 50))
MULTIPLIER = 10.0
DATES = pd.date_range(
    "2023-10-09 15:00", "2023-11-24 15:00", freq="B", tz="Asia/Shanghai"
)


def svi_vol(strike: float, forward: float, t: float, p: Dict[str, float]) -> float:
    """SVI 隐含波动率; 参数按 1 年期给出, 总方差随期限等比缩放."""
    k = np.log(strike / forward)
    w = p["a"] + p["b"] * (
        p["rho"] * (k - p["m"]) + np.sqrt((k - p["m"]) ** 2 + p["sigma"] ** 2)
    )
    return float(np.sqrt(w))


def option_symbol(underlying: str, kind: str, strike: int) -> str:
    """大商所期权代码格式, 如 m2401-C-3000."""
    return f"{underlying}-{kind[0]}-{strike}"


def frame(symbol: str, close: np.ndarray) -> pd.DataFrame:
    """日线(收盘 15:00, 上海时区)."""
    return pd.DataFrame(
        {
            "date": DATES,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 10_000,
            "symbol": symbol,
        }
    )


def make_data(seed: int = 7) -> Dict[str, pd.DataFrame]:
    """合成两个月份的期货与期权日线."""
    rng = np.random.default_rng(seed)
    data: Dict[str, pd.DataFrame] = {}
    for underlying, (expiry, start, svi) in SERIES.items():
        steps = rng.normal(0.0, 0.25 / np.sqrt(252), len(DATES))
        futures = start * np.exp(np.cumsum(steps) - steps[0])
        data[underlying] = frame(underlying, futures)
        end = pd.Timestamp(f"{expiry} 15:00", tz="Asia/Shanghai")
        t = np.array([(end - d).total_seconds() / (365 * 86400) for d in DATES])
        for strike in STRIKES:
            noise = rng.normal(0.0, 0.01, len(DATES))
            sigma = np.array([svi_vol(strike, f, 1.0, svi) for f in futures]) + noise
            for kind in ("CALL", "PUT"):
                symbol = option_symbol(underlying, kind, strike)
                price = np.asarray(black76_price(futures, strike, t, 0.0, sigma, kind))
                data[symbol] = frame(symbol, price)
    return data


def make_config() -> BacktestConfig:
    """两个期货与各自期权的合约配置."""
    instruments: List[InstrumentConfig] = []
    for underlying, (expiry, _, _) in SERIES.items():
        instruments.append(
            InstrumentConfig(
                symbol=underlying,
                asset_type="FUTURES",
                multiplier=MULTIPLIER,
                margin_ratio=0.1,
            )
        )
        for strike in STRIKES:
            for kind in ("CALL", "PUT"):
                instruments.append(
                    InstrumentConfig(
                        symbol=option_symbol(underlying, kind, strike),
                        asset_type="OPTION",
                        multiplier=MULTIPLIER,
                        option_type="CALL" if kind == "CALL" else "PUT",
                        strike_price=float(strike),
                        expiry_date=expiry,
                        underlying_symbol=underlying,
                    )
                )
    return BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=instruments,
    )


class SurfaceMonitor(Strategy):
    """每周五打印曲面摘要, 最后一天比较 SVI/SABR 并找出相对微笑偏离最大的合约."""

    def on_bars(self, batch: BarBatch) -> None:
        """每个交易日所有合约的日线到齐后执行."""
        now = self.now
        if now is None:
            return
        if now == DATES[-1]:
            self.final_report()
        elif now.day_of_week == 4:
            self.weekly_report(now.strftime("%Y-%m-%d"))

    def weekly_report(self, day: str) -> None:
        """两个到期日在 90%/100%/110% 行权价处的波动率, 以及 60 天平值波动率.

        行权价范围窄、报价有噪声时 SVI 的参数(尤其 rho、m、sigma)在相邻时刻之间
        可能跳动, 但拟合出的曲线稳定, 所以这里看曲线上的点而不是参数。
        """
        surface = self.vol_surface(list(SERIES))
        parts = []
        for expiry in surface.expiries:
            smile = surface.smile(expiry)
            f = smile.forward
            parts.append(
                f"{expiry} 90%/平值/110% {smile.vol(0.9 * f):.2%}/"
                f"{smile.vol(f):.2%}/{smile.vol(1.1 * f):.2%} 误差 {smile.rmse:.2%}"
            )
        near = surface.smile(surface.expiries[0])
        atm_60d = surface.vol(near.forward, 60 / 365)
        print(f"{day} | " + " | ".join(parts) + f" | 60 天平值 {atm_60d:.2%}")

    def final_report(self) -> None:
        """SVI 与 SABR 的拟合对比, 以及近月相对 SVI 微笑最贵、最便宜的合约."""
        svi = self.vol_surface(list(SERIES))
        sabr = self.vol_surface(list(SERIES), model="sabr")
        for expiry in svi.expiries:
            a, b = svi.smile(expiry), sabr.smile(expiry)
            print(
                f"{expiry}: SVI 误差 {a.rmse:.2%}(蝶式套利 {a.butterfly_arbitrage}) "
                f"/ SABR 误差 {b.rmse:.2%}(nu {b.params['nu']:.2f})"
            )
        print(f"日历套利: {svi.calendar_arbitrage or '无'}")
        underlying = next(iter(SERIES))
        chain = self.option_chain(underlying)
        smile = svi.smile(int(chain["expiry_date"].iloc[0]))
        forward = smile.forward
        otm = chain[(chain["strike"] < forward) == (chain["option_type"] == "PUT")]
        otm = otm.dropna(subset=["iv"])
        rich = (otm["iv"] - smile.vol(otm["strike"].to_numpy())).to_numpy()
        ranked = otm.assign(rich=rich).sort_values("rich")
        for label, rows in (("最便宜", ranked.head(2)), ("最贵", ranked.tail(2))):
            text = ", ".join(f"{r.symbol} {r.rich:+.2%}" for r in rows.itertuples())
            print(f"相对 SVI 微笑{label}: {text}")


def main() -> Any:
    """运行回测(不下单)并打印曲面."""
    return run_backtest(
        data=make_data(),
        strategy=SurfaceMonitor(),
        config=make_config(),
        freq="1d",
        show_progress=False,
    )


if __name__ == "__main__":
    main()
