#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期权链、Greeks 与 Delta 对冲：做多平值跨式并每日对冲（gamma scalping）.

演示策略内的期权方法：

- ``option_chain()``：期权链，含最新价、远期、隐含波动率与 Greeks；
- ``select_option()``：按平值档位或目标 Delta 选合约；
- ``get_greeks()`` / ``portfolio_greeks()``：单合约与组合的 Greeks；
- ``hedge_delta()``：用标的期货把组合现金 Delta 对冲回 0。

策略：开始时买入平值认购和平值认沽各 ``QTY`` 张（做多波动率），此后每个交易日
收盘用豆粕期货对冲 Delta（对冲单在下一根日线成交），最后一天全部平仓。标的的
实际波动高于买入时的隐含波动率时，反复对冲“高抛低吸”赚到的钱能超过期权的
时间价值损耗。

数据是本地合成的，不需要联网：豆粕期货 ``m2401`` 按 30% 年化波动随机游走，
9 档行权价的认购、认沽按 Black-76 定价，隐含波动率带微笑（越虚值越高，平值
约 20%）。日线打 15:00 戳，与期权链“到期日 15:00 到期”的期限口径一致，所以
期权链反推出的隐含波动率就是生成数据时用的微笑。

其他品种的用法：商品期权的远期取标的期货价（本例）；股指期权把指数登记为
``underlying_symbol``，远期按同一到期的认购、认沽价格平价反推，对冲时传
``hedge_delta("000300", hedge_symbol="IF2412")``。有买一、卖一报价的 tick 数据
可把类属性 ``option_price_source`` 设为 ``"mid"``，按买卖中间价求隐含波动率。

运行方式::

    python examples/80_option_chain_greeks.py

输出：开仓时的期权链（微笑）与所选合约、每周的组合 Greeks 与对冲持仓、最终收益。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from akquant import (
    BacktestConfig,
    BarBatch,
    ChinaOptionsConfig,
    InstrumentConfig,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.option import black76_price

UNDERLYING = "m2401"
EXPIRY = 20231215
STRIKES = [2800, 2850, 2900, 2950, 3000, 3050, 3100, 3150, 3200]
MULTIPLIER = 10.0
RATE = 0.02
QTY = 10
DATES = pd.date_range(
    "2023-10-09 15:00", "2023-11-24 15:00", freq="B", tz="Asia/Shanghai"
)


def smile(forward: float, strike: float) -> float:
    """隐含波动率微笑: 平值约 20%, 越虚值越高."""
    return float(0.20 + 0.8 * np.log(strike / forward) ** 2)


def option_symbol(kind: str, strike: int) -> str:
    """大商所期权代码格式, 如 m2401-C-3000."""
    return f"{UNDERLYING}-{kind[0]}-{strike}"


def make_data(seed: int = 1) -> Dict[str, pd.DataFrame]:
    """合成期货与期权日线(收盘 15:00, 上海时区)."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.30 / np.sqrt(252)
    futures = 3000.0 * np.exp(np.cumsum(rng.normal(0.0, daily_vol, len(DATES))))
    futures[0] = 3000.0
    expiry = pd.Timestamp(f"{EXPIRY} 15:00", tz="Asia/Shanghai")
    t = np.array([(expiry - d).total_seconds() / (365 * 86400) for d in DATES])

    def frame(symbol: str, close: np.ndarray) -> pd.DataFrame:
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

    data = {UNDERLYING: frame(UNDERLYING, futures)}
    for strike in STRIKES:
        sigma = np.array([smile(f, strike) for f in futures])
        for kind in ("CALL", "PUT"):
            price = np.asarray(black76_price(futures, strike, t, RATE, sigma, kind))
            data[option_symbol(kind, strike)] = frame(
                option_symbol(kind, strike), price
            )
    return data


def make_config() -> BacktestConfig:
    """期货与 18 张期权的合约配置; 商品期权按张收 2 元佣金."""
    instruments: List[InstrumentConfig] = [
        InstrumentConfig(
            symbol=UNDERLYING,
            asset_type="FUTURES",
            multiplier=MULTIPLIER,
            margin_ratio=0.1,
        )
    ]
    for strike in STRIKES:
        for kind in ("CALL", "PUT"):
            instruments.append(
                InstrumentConfig(
                    symbol=option_symbol(kind, strike),
                    asset_type="OPTION",
                    multiplier=MULTIPLIER,
                    option_type="CALL" if kind == "CALL" else "PUT",
                    strike_price=float(strike),
                    expiry_date=EXPIRY,
                    underlying_symbol=UNDERLYING,
                )
            )
    return BacktestConfig(
        strategy_config=StrategyConfig(initial_cash=1_000_000.0),
        instruments_config=instruments,
        china_options=ChinaOptionsConfig(
            commission_per_contract=2.0,
            exchange_fee_per_contract=0.0,
            clearing_fee_per_contract=0.0,
            sell_open_exempt=False,
        ),
    )


class GammaScalping(Strategy):
    """做多平值跨式, 每日收盘用期货对冲 Delta."""

    option_rate = RATE

    def __init__(self) -> None:
        """初始化持仓腿与对冲次数."""
        super().__init__()
        self.legs: Optional[List[str]] = None
        self.hedges = 0

    def on_bars(self, batch: BarBatch) -> None:
        """每个交易日所有合约的日线到齐后执行."""
        now = self.now
        if self.legs is None:
            self.open_straddle()
        elif now is not None and now == DATES[-1]:
            for symbol in [*self.legs, UNDERLYING]:
                self.close_position(symbol)
        else:
            if now is not None and now.day_of_week == 4:
                self.report()
            # 对冲单在下一根日线成交, 所以周五打印的是对冲前的 Greeks
            if self.hedge_delta(UNDERLYING) is not None:
                self.hedges += 1

    def open_straddle(self) -> None:
        """打印期权链微笑, 选平值认购、认沽并买入."""
        chain = self.option_chain(UNDERLYING)
        smile_table = chain.pivot(index="strike", columns="option_type", values="iv")
        print(f"开仓日期权链(远期 {chain['forward'].iloc[0]:.0f}), 隐含波动率:")
        print(smile_table.round(4).to_string())
        call = self.select_option(UNDERLYING, "CALL")
        put = self.select_option(UNDERLYING, "PUT")
        wing = self.select_option(UNDERLYING, "PUT", delta=0.25)
        assert call is not None and put is not None
        greeks = self.get_greeks(call)
        print(
            f"平值认购 {call}: 价格 {greeks['price']:.2f}, iv {greeks['iv']:.2%}, "
            f"delta {greeks['delta']:.3f}; 平值认沽 {put}; 25 Delta 认沽 {wing}"
        )
        self.buy(call, QTY)
        self.buy(put, QTY)
        self.legs = [call, put]

    def report(self) -> None:
        """打印对冲前的组合 Greeks 与期货持仓(Greeks 含期货)."""
        g = self.portfolio_greeks(UNDERLYING)
        now = self.now
        day = now.strftime("%Y-%m-%d") if now is not None else "-"
        print(
            f"{day} 对冲前 期货 {self.get_position(UNDERLYING):+.0f} 手 | "
            f"delta {g['delta']:+8.2f} gamma {g['gamma']:.4f} "
            f"vega {g['vega'] / 100:8.1f}/1% theta {g['theta'] / 365:8.1f}/天"
        )


def main() -> Any:
    """运行回测并打印结果."""
    strategy = GammaScalping()
    result = run_backtest(
        data=make_data(),
        strategy=strategy,
        config=make_config(),
        freq="1d",
        show_progress=False,
    )
    print(f"对冲调仓 {strategy.hedges} 次")
    print(f"总收益率: {result.metrics.total_return_pct:.2f}%")
    return result


if __name__ == "__main__":
    main()
