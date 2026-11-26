"""内置 sizer 在期货上按占用保证金与可用资金定量, 股票、基金结果不变.

``PercentSizer(p)``: 用可用资金的 p% 作为这笔开仓占用的资金; ``AllInSizer``: 用全部
可用资金。每单位占用资金 = 价格 × 合约乘数 × 保证金率, 保证金率只对期货生效。期货上
传给 sizer 的 ``cash`` 是可用资金(权益 − 占用保证金 − 冻结资金), 其余品种仍是
``ctx.cash``。
"""

import math
from typing import Any, Dict, List, Optional, Tuple, cast

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.config import RiskConfig
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import BrokerOrderSubmitter
from akquant.sizer import AllInSizer, FixedSize, PercentSizer, Sizer

_FUT = "RB2310"
_STOCK = "000001.SZ"
_FUND = "510300.SH"
_TZ = "Asia/Shanghai"
_CASH = 100_000.0
_PRICE = 10.0
_SAFETY = 0.0001  # 风控缺省安全垫

# 两组(合约乘数, 保证金率): 乘数 × 保证金率都不等于 1, 按价格除和按保证金除结果不同
_FUTURES_SPECS = [(10.0, 0.08), (300.0, 0.12)]
_SPEC_IDS = ["mult10_ratio0.08", "mult300_ratio0.12"]

Action = Tuple[str, Tuple[Any, ...]]
OrderRow = Tuple[str, float, float, str]


def _daily(prices: List[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-02 15:00", periods=len(prices), freq="B"
    ).tz_localize(_TZ)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1e9] * len(prices),
        },
        index=index,
    )


class _Scripted(Strategy):
    """第 n 根 K 线执行 script[n]; ("sizer", (obj,)) 换 sizer, 其余调同名下单方法."""

    script: Dict[int, List[Action]] = {}
    target = _FUT

    def __init__(self) -> None:
        super().__init__()
        self.n = 0
        self.max_buy: List[float] = []

    def on_bar(self, bar: Bar) -> None:
        """按序号执行."""
        if bar.symbol != self.target:
            return
        for name, args in self.script.get(self.n, []):
            if name == "sizer":
                self.set_sizer(args[0])
            elif name == "max_buy":
                self.max_buy.append(self._calculate_max_buy_qty(self.target, *args))
            else:
                getattr(self, name)(self.target, *args)
        self.n += 1


def _strategy(script: Dict[int, List[Action]], target: str) -> _Scripted:
    cls = type("_Run", (_Scripted,), {"script": script, "target": target})
    return cast(_Scripted, cls())


def _run_futures(
    script: Dict[int, List[Action]],
    multiplier: float,
    margin_ratio: float,
    *,
    account_mode: str = "cash",
    prices: Optional[List[float]] = None,
) -> Tuple[Any, Any]:
    strategy = _strategy(script, _FUT)
    risk = RiskConfig(account_mode=account_mode, enable_short_sell=True)
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(
            initial_cash=_CASH, commission_rate=0.0, risk=risk
        ),
        instruments_config=[
            aq.InstrumentConfig(
                symbol=_FUT,
                asset_type="FUTURES",
                multiplier=multiplier,
                margin_ratio=margin_ratio,
            )
        ],
    )
    result = aq.run_backtest(
        data={_FUT: _daily(prices or [_PRICE] * 5)},
        strategy=strategy,
        symbols=_FUT,
        freq="1d",
        fill_policy=aq.NextOpen(),
        config=config,
        show_progress=False,
    )
    return result, strategy


def _orders(result: Any) -> List[OrderRow]:
    """(方向, 数量, 成交量, 状态) 按提交顺序."""
    return [
        (
            str(row["side"]),
            float(row["quantity"]),
            float(row["filled_quantity"]),
            str(row["status"]),
        )
        for row in result.orders_df.sort_values("created_at").to_dict("records")
    ]


class _RecordingPercent(PercentSizer):
    """记下内置 PercentSizer 算出的请求手数(引擎缩量之前)."""

    def __init__(self, percents: float, sink: List[float]) -> None:
        super().__init__(percents)
        self.sink = sink

    def get_size(
        self,
        price: float,
        cash: float,
        context: Any,
        symbol: str,
        unit_cost: Optional[float] = None,
    ) -> float:
        extra = {} if unit_cost is None else {"unit_cost": unit_cost}
        size = super().get_size(price, cash, context, symbol, **extra)
        self.sink.append(size)
        return size


class _RecordingAllIn(AllInSizer):
    def __init__(self, sink: List[float]) -> None:
        self.sink = sink

    def get_size(
        self,
        price: float,
        cash: float,
        context: Any,
        symbol: str,
        unit_cost: Optional[float] = None,
    ) -> float:
        extra = {} if unit_cost is None else {"unit_cost": unit_cost}
        size = super().get_size(price, cash, context, symbol, **extra)
        self.sink.append(size)
        return size


# ---------------------------------------------------------------------------
# 期货: 按占用保证金定量
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("multiplier", "margin_ratio"), _FUTURES_SPECS, ids=_SPEC_IDS)
@pytest.mark.parametrize("percent", [25.0, 50.0])
def test_futures_percent_sizer_commits_percent_of_funds_as_margin(
    multiplier: float, margin_ratio: float, percent: float
) -> None:
    """PercentSizer(p): 占用保证金 = 资金 × p%, 不被引擎缩量."""
    unit = _PRICE * multiplier * margin_ratio
    expected = math.floor(_CASH * percent / 100.0 / unit)
    requested: List[float] = []
    result, _ = _run_futures(
        {0: [("sizer", (_RecordingPercent(percent, requested),)), ("buy", ())]},
        multiplier,
        margin_ratio,
    )

    assert requested == [expected]
    assert _orders(result) == [("buy", expected, expected, "filled")]


@pytest.mark.parametrize(("multiplier", "margin_ratio"), _FUTURES_SPECS, ids=_SPEC_IDS)
@pytest.mark.parametrize("kind", ["percent100", "all_in"])
def test_futures_full_allocation_requests_funds_over_margin(
    multiplier: float, margin_ratio: float, kind: str
) -> None:
    """PercentSizer(100) / AllInSizer: 请求 资金 / 每手保证金, 引擎只缩掉安全垫零头."""
    unit = _PRICE * multiplier * margin_ratio
    requested_expected = math.floor(_CASH / unit)
    filled_expected = math.floor(_CASH * (1 - _SAFETY) / unit)
    requested: List[float] = []
    sizer: Sizer = (
        _RecordingPercent(100.0, requested)
        if kind == "percent100"
        else _RecordingAllIn(requested)
    )
    result, _ = _run_futures(
        {0: [("sizer", (sizer,)), ("buy", ())]}, multiplier, margin_ratio
    )

    assert requested == [requested_expected]
    assert _orders(result) == [("buy", filled_expected, filled_expected, "filled")]


@pytest.mark.parametrize("account_mode", ["cash", "margin"])
@pytest.mark.parametrize(("multiplier", "margin_ratio"), _FUTURES_SPECS, ids=_SPEC_IDS)
def test_futures_second_position_sized_from_available_funds(
    multiplier: float, margin_ratio: float, account_mode: str
) -> None:
    """持有 100 手后再 PercentSizer(50): 按可用资金(资金 − 已占保证金)的一半定量.

    此前传给 sizer 的是 ctx.cash: 现金账户里它按名义市值扣减, 持仓后为 0 或负数(算出
    0 手); 保证金账户里它不扣保证金, 算出的手数超过可用资金。
    """
    unit = _PRICE * multiplier * margin_ratio
    available = _CASH - 100 * unit
    expected = math.floor(available * 0.5 / unit)
    result, _ = _run_futures(
        {
            0: [("buy", (100,))],
            2: [("sizer", (PercentSizer(50),)), ("buy", ())],
        },
        multiplier,
        margin_ratio,
        account_mode=account_mode,
    )

    assert _orders(result) == [
        ("buy", 100.0, 100.0, "filled"),
        ("buy", expected, expected, "filled"),
    ]


def test_futures_sell_without_quantity_does_not_use_sizer() -> None:
    """Sell / short 不给数量不走 sizer: 空仓时不下单, 有多头时全平(口径不变)."""
    flat, _ = _run_futures(
        {
            0: [("sizer", (PercentSizer(50),)), ("sell", ()), ("short", ())],
        },
        10.0,
        0.08,
    )
    assert _orders(flat) == []

    holding, _ = _run_futures(
        {0: [("buy", (10,))], 2: [("sizer", (PercentSizer(50),)), ("sell", ())]},
        10.0,
        0.08,
    )
    assert _orders(holding) == [
        ("buy", 10.0, 10.0, "filled"),
        ("sell", 10.0, 10.0, "filled"),
    ]


# ---------------------------------------------------------------------------
# 自定义 sizer: 只实现 get_size
# ---------------------------------------------------------------------------


class _Spy(Sizer):
    """只实现 get_size: 记下收到的 (price, cash), 固定买 3 手/股."""

    def __init__(self, seen: List[Tuple[float, float]], size: float = 300.0) -> None:
        self.seen = seen
        self.size = size

    def get_size(self, price: float, cash: float, context: Any, symbol: str) -> float:
        self.seen.append((price, cash))
        return self.size


def test_custom_sizer_on_futures_receives_price_and_available_funds() -> None:
    """自定义 sizer 收到的 price 仍是价格; 期货上 cash 是可用资金而非 ctx.cash."""
    seen: List[Tuple[float, float]] = []
    multiplier, margin_ratio = 10.0, 0.08
    result, _ = _run_futures(
        {0: [("buy", (100,))], 2: [("sizer", (_Spy(seen, 3.0),)), ("buy", ())]},
        multiplier,
        margin_ratio,
    )

    assert seen == [(_PRICE, _CASH - 100 * _PRICE * multiplier * margin_ratio)]
    assert _orders(result)[-1] == ("buy", 3.0, 3.0, "filled")


# ---------------------------------------------------------------------------
# _calculate_max_buy_qty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("multiplier", "margin_ratio"), _FUTURES_SPECS, ids=_SPEC_IDS)
def test_calculate_max_buy_qty_on_futures_uses_margin(
    multiplier: float, margin_ratio: float
) -> None:
    """期货: 资金扣安全垫后按每手保证金折算(费率为 0)."""
    _, strategy = _run_futures(
        {0: [("max_buy", (_PRICE, _CASH))]}, multiplier, margin_ratio
    )
    unit = _PRICE * multiplier * margin_ratio
    assert strategy.max_buy == [float(math.floor(_CASH * (1 - _SAFETY) / unit))]


# ---------------------------------------------------------------------------
# 股票(现金账户、融资账户)与基金: 逐笔不变
# ---------------------------------------------------------------------------

_STOCK_PRICES = [10.0, 10.3, 9.8, 10.1, 10.6, 10.2, 9.9, 10.4]


def _stock_script() -> Dict[int, List[Action]]:
    """先 30%、再全仓、全卖、再 50%, 中间夹一次固定数量和一次自定义 sizer."""
    return {
        0: [("sizer", (PercentSizer(30),)), ("buy", ())],
        1: [("sizer", (AllInSizer(),)), ("buy", ())],
        3: [("sell", ())],
        4: [("sizer", (PercentSizer(50),)), ("buy", ())],
        5: [("sizer", (FixedSize(200),)), ("buy", ())],
        6: [("sizer", (_Spy([], 700.0),)), ("buy", ()), ("max_buy", (10.0, _CASH))],
    }


def _run_stock(target: str, policy: Any, *, margin: bool) -> Tuple[Any, Any]:
    strategy = _strategy(_stock_script(), target)
    kwargs: Dict[str, Any] = {}
    if margin:
        kwargs["risk_config"] = RiskConfig(
            account_mode="margin", enable_short_sell=True, initial_margin_ratio=0.5
        )
    if target == _FUND:
        kwargs["config"] = aq.BacktestConfig(
            strategy_config=aq.StrategyConfig(initial_cash=_CASH),
            instruments_config=[aq.InstrumentConfig(symbol=_FUND, asset_type="FUND")],
        )
    result = aq.run_backtest(
        data={target: _daily(_STOCK_PRICES)},
        strategy=strategy,
        symbols=target,
        initial_cash=_CASH,
        freq="1d",
        fill_policy=policy,
        t_plus_one=False,
        commission_rate=0.0003,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.0,
        lot_size=100,
        show_progress=False,
        **kwargs,
    )
    return result, strategy


def _fills(result: Any) -> List[Tuple[str, float, float]]:
    return [
        (str(row["side"]), float(row["quantity"]), float(row["price"]))
        for row in result.executions_df.to_dict("records")
    ]


# 在修改前的提交(ed21e77d)上跑出来的结果, 修改后必须逐笔相同
_STOCK_BASELINE: Dict[str, Any] = {
    "000001.SZ|False|NextOpen": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 5000.0, 5000.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.3),
            ("buy", 6700.0, 9.8),
            ("sell", 9700.0, 10.6),
            ("buy", 5000.0, 10.2),
            ("buy", 200.0, 9.9),
            ("buy", 700.0, 10.4),
        ],
        "max_buy": [9900.0],
    },
    "000001.SZ|False|CurrentClose": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 4600.0, 4600.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.0),
            ("buy", 6700.0, 10.3),
            ("sell", 9700.0, 10.1),
            ("buy", 4600.0, 10.6),
            ("buy", 200.0, 10.2),
            ("buy", 700.0, 9.9),
        ],
        "max_buy": [9900.0],
    },
    "000001.SZ|True|NextOpen": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 5000.0, 5000.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.3),
            ("buy", 6700.0, 9.8),
            ("sell", 9700.0, 10.6),
            ("buy", 5000.0, 10.2),
            ("buy", 200.0, 9.9),
            ("buy", 700.0, 10.4),
        ],
        "max_buy": [9900.0],
    },
    "000001.SZ|True|CurrentClose": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 4600.0, 4600.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.0),
            ("buy", 6700.0, 10.3),
            ("sell", 9700.0, 10.1),
            ("buy", 4600.0, 10.6),
            ("buy", 200.0, 10.2),
            ("buy", 700.0, 9.9),
        ],
        "max_buy": [9900.0],
    },
    "510300.SH|False|NextOpen": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 5000.0, 5000.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.3),
            ("buy", 6700.0, 9.8),
            ("sell", 9700.0, 10.6),
            ("buy", 5000.0, 10.2),
            ("buy", 200.0, 9.9),
            ("buy", 700.0, 10.4),
        ],
        "max_buy": [9900.0],
    },
    "510300.SH|False|CurrentClose": {
        "orders": [
            ("buy", 3000.0, 3000.0, "filled"),
            ("buy", 6700.0, 6700.0, "filled"),
            ("sell", 9700.0, 9700.0, "filled"),
            ("buy", 4600.0, 4600.0, "filled"),
            ("buy", 200.0, 200.0, "filled"),
            ("buy", 700.0, 700.0, "filled"),
        ],
        "fills": [
            ("buy", 3000.0, 10.0),
            ("buy", 6700.0, 10.3),
            ("sell", 9700.0, 10.1),
            ("buy", 4600.0, 10.6),
            ("buy", 200.0, 10.2),
            ("buy", 700.0, 9.9),
        ],
        "max_buy": [9900.0],
    },
}


@pytest.mark.parametrize(
    ("target", "margin"),
    [(_STOCK, False), (_STOCK, True), (_FUND, False)],
    ids=["stock_cash", "stock_margin", "fund_cash"],
)
@pytest.mark.parametrize(
    "policy", [aq.NextOpen, aq.CurrentClose], ids=["next_open", "current_close"]
)
def test_stock_and_fund_results_unchanged(
    target: str, margin: bool, policy: Any
) -> None:
    """股票(现金/融资账户)、基金: sizer 定量的订单与成交和修改前逐笔相同."""
    result, strategy = _run_stock(target, policy(), margin=margin)
    key = f"{target}|{margin}|{policy.__name__}"
    expected = _STOCK_BASELINE[key]
    assert _orders(result) == expected["orders"]
    assert _fills(result) == expected["fills"]
    assert strategy.max_buy == expected["max_buy"]


def test_custom_sizer_on_stock_receives_price_and_ctx_cash() -> None:
    """股票上自定义 sizer 收到的 price 是参考价, cash 仍是 ctx.cash."""
    seen: List[Tuple[float, float]] = []

    class _Run(_Scripted):
        script = {0: [("sizer", (_Spy(seen),)), ("buy", ())]}
        target = _STOCK

        def on_bar(self, bar: Bar) -> None:
            if self.n == 0:
                assert self.ctx is not None
                self.ctx_cash = float(self.ctx.cash)
            super().on_bar(bar)

    strategy = _Run()
    result = aq.run_backtest(
        data={_STOCK: _daily([_PRICE] * 3)},
        strategy=strategy,
        symbols=_STOCK,
        initial_cash=_CASH,
        freq="1d",
        fill_policy=aq.NextOpen(),
        show_progress=False,
    )
    assert seen == [(_PRICE, strategy.ctx_cash)]
    assert _orders(result) == [("buy", 300.0, 300.0, "filled")]


# ---------------------------------------------------------------------------
# paper 实盘(run_live 模拟成交): 与回测同一口径
# ---------------------------------------------------------------------------


def test_paper_futures_sizer_uses_margin_and_available_funds() -> None:
    """paper: 首单 PercentSizer(50) 按保证金定量, 持仓后再开仓按可用资金定量.

    paper 期货沿用默认费率(不能设为 0), 第二单的可用资金在策略里读出来再按公式算。
    """
    multiplier, margin_ratio = 10.0, 0.08
    unit = _PRICE * multiplier * margin_ratio
    first = math.floor(_CASH * 0.5 / unit)

    index = pd.date_range(
        "2024-01-02 09:31:00", periods=6, freq="1min", tz="Asia/Shanghai"
    )
    bars = [
        Bar(int(ts.value), _PRICE, _PRICE, _PRICE, _PRICE, 1e9, _FUT) for ts in index
    ]
    strategy = _strategy(
        {
            0: [("sizer", (PercentSizer(50),)), ("buy", ())],
            2: [("buy", ())],
        },
        _FUT,
    )
    positions: List[float] = []
    available: List[float] = []
    on_bar = strategy.on_bar

    def recording_on_bar(bar: Bar) -> None:
        positions.append(strategy.get_position(_FUT))
        available.append(float(strategy.get_account()["available_cash"]))
        on_bar(bar)

    strategy.on_bar = recording_on_bar  # type: ignore[method-assign]
    aq.run_live(
        strategy_cls=strategy,
        instruments=[
            aq.Instrument(
                symbol=_FUT,
                asset_type=aq.AssetType.Futures,
                multiplier=multiplier,
                margin_ratio=margin_ratio,
                tick_size=0.01,
                lot_size=1,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=_CASH,
        show_progress=False,
        duration="60s",
    )

    assert positions[2] == first
    # 第二单前可用资金 = 资金 − 首单保证金 − 手续费(ctx.cash 已按名义额扣成负数)
    assert _CASH - first * unit - 100.0 < available[2] < _CASH - first * unit
    second = math.floor(available[2] * 0.5 / unit)
    assert strategy.get_position(_FUT) == first + second


# ---------------------------------------------------------------------------
# broker_live: 合约快照可用, 资金仍取柜台现金
# ---------------------------------------------------------------------------


class _BrokerExecution:
    def __init__(self, cash: float) -> None:
        self._cash = cash

    def get_cash(self) -> float:
        return self._cash


class _FuturesSnapshot:
    asset_type = "FUTURES"
    multiplier = 10.0
    margin_ratio = 0.08
    lot_size = 1


class _BrokerStrategy:
    _owner_strategy_id = "_default"
    broker_ready = True
    ctx = None
    current_bar = None
    current_tick = None

    def __init__(self, sizer: Any) -> None:
        self.sizer = sizer
        self.execution = _BrokerExecution(_CASH)
        self._last_prices: Dict[str, float] = {}
        self._instrument_snapshots: Dict[str, Any] = {_FUT: _FuturesSnapshot()}


def _broker_submit(strategy: _BrokerStrategy) -> List[float]:
    captured: List[UnifiedOrderRequest] = []

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            captured.append(req)
            return f"b{len(captured)}"

    cap = BrokerCapability(broker_name="middleware")
    BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    ).submit_order(symbol=_FUT, side="Buy", price=_PRICE)
    return [float(r.quantity) for r in captured]


def test_broker_live_builtin_sizer_divides_by_margin_per_lot() -> None:
    """broker_live: 内置 sizer 按快照里的乘数与保证金率定量, 资金仍取 get_cash()."""
    unit = _PRICE * _FuturesSnapshot.multiplier * _FuturesSnapshot.margin_ratio
    strategy = _BrokerStrategy(PercentSizer(50))
    assert _broker_submit(strategy) == [float(math.floor(_CASH * 0.5 / unit))]


def test_broker_live_custom_sizer_receives_price_and_cash() -> None:
    """broker_live: 自定义 sizer 收到价格与 get_cash(), 与修改前相同."""
    seen: List[Tuple[float, float]] = []
    strategy = _BrokerStrategy(_Spy(seen, 4.0))
    assert _broker_submit(strategy) == [4.0]
    assert seen == [(_PRICE, _CASH)]


# ---------------------------------------------------------------------------
# 期权买方按全额权利金; 非期货标的不看合约乘数与保证金率
# ---------------------------------------------------------------------------

_OPT = "CALL_OPT"
_UNDERLYING = "UL"


def _run_single(
    symbol: str, instruments: List[Any], sizer: Sizer, prices: List[float]
) -> Any:
    strategy = _strategy({0: [("sizer", (sizer,)), ("buy", ())]}, symbol)
    config = aq.BacktestConfig(
        strategy_config=aq.StrategyConfig(initial_cash=_CASH, commission_rate=0.0),
        instruments_config=instruments,
    )
    return aq.run_backtest(
        data={symbol: _daily(prices)},
        strategy=strategy,
        symbols=symbol,
        freq="1d",
        fill_policy=aq.NextOpen(),
        config=config,
        show_progress=False,
    )


@pytest.mark.parametrize("kind", ["percent50", "all_in"])
def test_option_buyer_sized_by_full_premium(kind: str) -> None:
    """期权买方: 每张占用 = 权利金 × 合约乘数(全额), 不按保证金率打折.

    权利金 0.5、乘数 10000, 每张 5000 元; 10 万资金全仓 20 张、一半 10 张。
    此前按价格折算, 请求 20 万张 / 10 万张。
    """
    premium, multiplier = 0.5, 10000.0
    sink: List[float] = []
    sizer: Sizer = (
        _RecordingPercent(50.0, sink) if kind == "percent50" else _RecordingAllIn(sink)
    )
    instruments = [
        aq.InstrumentConfig(
            symbol=_OPT,
            asset_type="OPTION",
            multiplier=multiplier,
            margin_ratio=0.35,
            option_type="CALL",
            strike_price=10,
            expiry_date=20301231,
            underlying_symbol=_UNDERLYING,
        ),
        aq.InstrumentConfig(symbol=_UNDERLYING, asset_type="STOCK"),
    ]
    _run_single(_OPT, instruments, sizer, [premium] * 5)

    share = 0.5 if kind == "percent50" else 1.0
    expected = float(int(_CASH * share / (premium * multiplier)))
    assert expected == (10.0 if kind == "percent50" else 20.0)
    assert sink == [expected]


@pytest.mark.parametrize(
    ("asset_type", "symbol"),
    [("STOCK", _STOCK), ("FUND", _FUND)],
    ids=["stock", "fund"],
)
def test_non_futures_ignore_registered_multiplier_and_margin_ratio(
    asset_type: Any, symbol: str
) -> None:
    """股票、基金登记了乘数与保证金率也按价格全额折算(引擎对它们的乘数恒为 1)."""
    sink: List[float] = []
    instruments = [
        aq.InstrumentConfig(
            symbol=symbol, asset_type=asset_type, multiplier=10.0, margin_ratio=0.5
        )
    ]
    _run_single(symbol, instruments, _RecordingPercent(50.0, sink), [_PRICE] * 5)

    assert sink == [float(int(_CASH * 0.5 / _PRICE))] == [5000.0]
