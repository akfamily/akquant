"""实盘期权行权指令(CTP 执行宣告): 按交易所与今昨仓拆腿、等柜台受理、拒绝原因上抛.

拆腿规则来自 SimNow 实测: 上期所、能源中心今仓须报"平今"、昨仓报"平仓"; 大商所、
郑商所、广期所只接受"平仓"。这里用假的 CTP API 同步回调受理/拒绝。
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from akquant.gateway import ctp_native
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.ctp_native import CTPTraderGateway

CONSTANTS = {
    "THOST_FTDC_OF_Close": "1",
    "THOST_FTDC_OF_CloseToday": "3",
    "THOST_FTDC_ACTP_Exec": "1",
    "THOST_FTDC_PD_Long": "2",
    "THOST_FTDC_HF_Speculation": "1",
    "THOST_FTDC_EOPF_UnReserve": "1",
    "THOST_FTDC_EOCF_NotToClose": "1",
}


def _gateway(
    monkeypatch: pytest.MonkeyPatch,
    exchange: str,
    today: float,
    yesterday: float,
    reject: Optional[Dict[int, str]] = None,
    silent: bool = False,
) -> tuple[CTPTraderGateway, List[Any]]:
    """假柜台: 第 i 条指令按 ``reject[i]`` 拒绝, 否则受理(未执行); silent 时不回报."""
    for name, value in CONSTANTS.items():
        monkeypatch.setattr(ctp_native.tdapi, name, value, raising=False)
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcInputExecOrderField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.broker_id, gateway.user_id = "9999", "tester"
    gateway.connected = gateway.login_status = gateway.ready_to_trade = True
    gateway.front_id, gateway.session_id = 1, 7
    gateway.order_ref, gateway.req_id = 100, 0
    gateway._exercise_timeout_sec = 0.05
    gateway._exercise_waiters = {}
    gateway.query_positions = lambda instrument_id=None: [  # type: ignore[method-assign]
        {
            "symbol": "ag2702C10300",
            "direction": "Buy",
            "available_today_quantity": today,
            "available_yesterday_quantity": yesterday,
            "exchange_id": exchange,
        }
    ]
    sent: List[Any] = []

    def insert(request: Any, req_id: int) -> int:
        sent.append(request)
        if silent:
            return 0
        message = (reject or {}).get(len(sent))
        if message is not None:
            gateway.OnRspExecOrderInsert(
                request, SimpleNamespace(ErrorID=74, ErrorMsg=message), req_id, True
            )
        else:
            gateway.OnRtnExecOrder(
                SimpleNamespace(
                    FrontID=1,
                    SessionID=7,
                    ExecOrderRef=request.ExecOrderRef,
                    InstrumentID=request.InstrumentID,
                    ExecResult="n",
                    OrderSubmitStatus="3",
                    StatusMsg="未执行",
                )
            )
        return 0

    gateway.api = SimpleNamespace(ReqExecOrderInsert=insert)
    return gateway, sent


def _legs(sent: List[Any]) -> List[tuple]:
    return [(r.ExchangeID, r.OffsetFlag, r.Volume, r.CloseFlag) for r in sent]


def test_shfe_splits_previous_and_today_positions() -> None:
    """上期所: 先昨仓"平仓"再今仓"平今", 行权后不自动平仓."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "SHFE", today=1, yesterday=2)
        legs = gateway.exercise("ag2702C10300", 3)
    assert _legs(sent) == [("SHFE", "1", 2, "1"), ("SHFE", "3", 1, "1")]
    assert [(leg["offset"], leg["quantity"]) for leg in legs] == [("1", 2), ("3", 1)]
    assert all(r.PosiDirection == "2" and r.ActionType == "1" for r in sent)


def test_dce_sends_single_close_order() -> None:
    """大商所(郑商所、广期所同): 今仓也只报一条"平仓"."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "DCE", today=2, yesterday=0)
        gateway.exercise("ag2702C10300", 2)
    assert _legs(sent) == [("DCE", "1", 2, "1")]


def test_exercise_reports_exact_causes() -> None:
    """超过柜台可用持仓不发指令; 柜台拒绝、部分受理、超时各自写明."""
    with pytest.MonkeyPatch.context() as mp:
        gateway, sent = _gateway(mp, "SHFE", today=1, yesterday=1)
        with pytest.raises(ValueError) as over:
            gateway.exercise("ag2702C10300", 3)
        assert sent == []
    assert str(over.value) == (
        "ag2702C10300 申请行权 3 手, 超过柜台可用多头持仓 2 手(今仓 1, 昨仓 1)"
    )

    with pytest.MonkeyPatch.context() as mp:
        gateway, _ = _gateway(
            mp, "SHFE", today=1, yesterday=1, reject={2: "CTP:执行仓位不足"}
        )
        with pytest.raises(RuntimeError) as partial:
            gateway.exercise("ag2702C10300", 2)
    assert str(partial.value) == (
        "ag2702C10300 行权指令 101(1 手)被拒: 74 CTP:执行仓位不足; 指令 100 1 手已受理"
    )

    with pytest.MonkeyPatch.context() as mp:
        gateway, _ = _gateway(mp, "DCE", today=1, yesterday=0, silent=True)
        with pytest.raises(RuntimeError) as timeout:
            gateway.exercise("ag2702C10300", 1)
    assert str(timeout.value) == (
        "ag2702C10300 行权指令 100(1 手) 0.05 秒内柜台未确认, "
        "状态未知, 请在交易终端核对"
    )
    assert gateway._exercise_waiters == {}


def test_broker_execution_routes_exercise_to_gateway() -> None:
    """实盘执行后端把行权交给柜台网关; 网关没有实现时报出网关类型."""
    calls: List[tuple] = []
    gateway = SimpleNamespace(exercise=lambda s, q: calls.append((s, q)))
    BrokerExecution(object(), gateway, None, None).exercise("m2701-C-3000", 2)  # type: ignore[arg-type]
    assert calls == [("m2701-C-3000", 2)]

    with pytest.raises(RuntimeError, match="柜台网关 SimpleNamespace 没有实现"):
        BrokerExecution(object(), SimpleNamespace(), None, None).exercise(  # type: ignore[arg-type]
            "m2701-C-3000", 1
        )
