"""组合持仓申请/拆分: 策略侧校验(回测)与 CTP 组合申请 ``ReqCombActionInsert``."""

import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import akquant
import pandas as pd
import pytest
from akquant import BarBatch, Strategy
from akquant.gateway import ctp_native
from akquant.gateway.ctp_native import CTPTraderGateway

DAYS = pd.date_range("2026-11-17 15:00", periods=3, freq="B", tz="Asia/Shanghai")
NEAR, FAR = "m2601", "m2605"
CALL, PUT = "m2605-C-3000", "m2605-P-3000"


def _bars(symbol: str, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": DAYS,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
            "symbol": symbol,
        }
    )


class _Spread(Strategy):
    """第 1 天卖近买远各 1 手, 第 2 天逐个尝试组合申请并记录结果."""

    def __init__(self) -> None:
        super().__init__()
        self.day = -1
        self.results: List[str] = []

    def on_bars(self, batch: BarBatch) -> None:
        self.day += 1
        if self.day == 0:
            self.sell(NEAR, 1)
            self.buy(FAR, 1)
        elif self.day == 1:
            cases: List[Any] = [
                ("SP", [(NEAR, -1), (FAR, 1)], 1),
                ("SP", [(NEAR, -1), (FAR, 1)], 2),
                ("SP", [(NEAR, 1), (FAR, 1)], 1),
                ("S P", [(NEAR, -1), (FAR, 1)], 1),
                ("SP", [(NEAR, -1)], 1),
                ("SP", [(NEAR, -1), (NEAR, -1)], 1),
                ("SP", [(NEAR, -0.5), (FAR, 1)], 1),
                ("SP", [(NEAR, -1), (FAR, 1)], 1.5),
            ]
            for code, legs, quantity in cases:
                try:
                    self.combine_positions(code, legs, quantity)
                    self.uncombine_positions(code, legs, quantity)
                    self.results.append("ok")
                except ValueError as exc:
                    self.results.append(str(exc))


def test_backtest_combine_validates_positions() -> None:
    """回测只做校验: 持仓足够时成功, 否则报出确切原因."""
    strategy = _Spread()
    akquant.run_backtest(
        data={NEAR: _bars(NEAR, 3000.0), FAR: _bars(FAR, 3050.0)},
        strategy=strategy,
        config=akquant.BacktestConfig(
            strategy_config=akquant.StrategyConfig(initial_cash=1_000_000.0),
            instruments_config=[
                akquant.InstrumentConfig(
                    symbol=s, asset_type="FUTURES", multiplier=10.0, margin_ratio=0.1
                )
                for s in (NEAR, FAR)
            ],
        ),
        freq="1d",
        show_progress=False,
    )
    assert strategy.results == [
        "ok",
        f"{NEAR} 空头持仓 1 手, 不足组合所需 2 手",
        f"{NEAR} 多头持仓 0 手, 不足组合所需 1 手",
        "组合代码须为不含空格和 & 的非空字符串, 实际: 'S P'",
        "组合持仓须恰好 2 条腿, 实际 1 条",
        f"组合持仓的两条腿相同: {NEAR}",
        f"组合持仓的腿 {NEAR} 每组手数须为非零整数, 实际: -0.5",
        "quantity(组数) 须为正整数, 实际: 1.5",
    ]


def _gateway(
    monkeypatch: pytest.MonkeyPatch, respond: Any
) -> tuple[CTPTraderGateway, List[Any]]:
    """组合申请用的网关替身; ``respond(gateway, request)`` 模拟柜台回报."""
    for name, value in {
        "THOST_FTDC_D_Buy": "0",
        "THOST_FTDC_D_Sell": "1",
        "THOST_FTDC_CMDR_Comb": "0",
        "THOST_FTDC_CMDR_UnComb": "1",
        "THOST_FTDC_HF_Speculation": "1",
    }.items():
        monkeypatch.setattr(ctp_native.tdapi, name, value, raising=False)
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcInputCombActionField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    requests: List[Any] = []
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)

    def insert(request: Any, req_id: int) -> int:
        requests.append(request)
        respond(gateway, request)
        return 0

    gateway.__dict__.update(
        broker_id="9999",
        user_id="tester",
        api=SimpleNamespace(ReqCombActionInsert=insert),
        req_id=0,
        connected=True,
        login_status=True,
        ready_to_trade=True,
        front_id=1,
        session_id=7,
        _comb_action_timeout_sec=0.05,
        _comb_action_ref=1,
        _comb_action_waiters={},
    )
    return gateway, requests


def _rtn(request: Any, status: str, **extra: Any) -> Any:
    fields: Dict[str, Any] = {
        "FrontID": 1,
        "SessionID": 7,
        "CombActionRef": request.CombActionRef,
        "ActionStatus": status,
        "StatusMsg": "",
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def test_ctp_combine_waits_for_exchange_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """组合合约代码与方向、手数按第一腿; 收到交易所受理才返回."""

    def respond(gateway: CTPTraderGateway, request: Any) -> None:
        # 回报在另一线程到达, 先报已提交, 再报已受理
        def push() -> None:
            gateway.OnRtnCombAction(_rtn(request, "a"))
            gateway.OnRtnCombAction(_rtn(request, "b"))

        threading.Thread(target=push).start()

    gateway, requests = _gateway(monkeypatch, respond)
    gateway._comb_action_timeout_sec = 2.0
    gateway.combine_positions("STD", [(CALL, -2.0), (PUT, -2.0)])
    gateway.combine_positions("BLS", [(CALL, 1.0), (PUT, -1.0)], uncombine=True)
    assert [
        (r.InstrumentID, r.Direction, r.Volume, r.CombDirection, r.HedgeFlag)
        for r in requests
    ] == [
        (f"STD {CALL}&{PUT}", "1", 2, "0", "1"),
        (f"BLS {CALL}&{PUT}", "0", 1, "1", "1"),
    ]
    assert [r.CombActionRef for r in requests] == ["1", "2"]
    assert gateway._comb_action_waiters == {}


def test_ctp_combine_reports_exact_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    """柜台拒绝、交易所拒绝、超时分别报出确切原因."""
    legs = [(CALL, -1.0), (PUT, -1.0)]
    label = f"申请组合 STD {CALL}&{PUT}"

    def rsp_error(gateway: CTPTraderGateway, request: Any) -> None:
        gateway.OnRspCombActionInsert(
            request, SimpleNamespace(ErrorID=16, ErrorMsg="CTP:找不到合约"), 1, True
        )

    gateway, _ = _gateway(monkeypatch, rsp_error)
    with pytest.raises(RuntimeError, match=f"^{label} 被拒绝: 16 CTP:找不到合约$"):
        gateway.combine_positions("STD", legs)

    def exchange_reject(gateway: CTPTraderGateway, request: Any) -> None:
        gateway.OnRtnCombAction(_rtn(request, "c", StatusMsg="持仓不足"))

    gateway, _ = _gateway(monkeypatch, exchange_reject)
    with pytest.raises(RuntimeError, match=f"^{label} 被拒绝: 持仓不足$"):
        gateway.combine_positions("STD", legs)

    def other_session(gateway: CTPTraderGateway, request: Any) -> None:
        gateway.OnRtnCombAction(_rtn(request, "b", SessionID=8))

    gateway, _ = _gateway(monkeypatch, other_session)
    with pytest.raises(RuntimeError, match="结果不可知"):
        gateway.combine_positions("STD", legs)
    assert gateway._comb_action_waiters == {}
