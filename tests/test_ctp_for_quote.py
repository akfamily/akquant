"""询价(ForQuote): 实盘经 CTP ReqForQuoteInsert 发出、拒绝记日志; 回测不产生报价.

SimNow 实测: 询价成功时柜台不回报(查询为"已接受"); 合约不存在时 OnRspForQuoteInsert
与 OnErrRtnForQuoteInsert 各回报一次"16 CTP:找不到合约"; 交易所留空可受理。
"""

import logging
from types import SimpleNamespace
from typing import Any, List

import akquant
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.gateway import ctp_native
from akquant.gateway.broker_execution import BrokerExecution
from akquant.gateway.ctp_native import CTPTraderGateway


def _gateway(monkeypatch: pytest.MonkeyPatch, ret: int = 0) -> tuple[Any, List[Any]]:
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcInputForQuoteField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.broker_id, gateway.user_id = "9999", "tester"
    gateway.connected = gateway.login_status = gateway.ready_to_trade = True
    gateway.order_ref, gateway.req_id = 100, 0
    gateway._for_quote_rejected = set()
    sent: List[Any] = []

    def insert(request: Any, req_id: int) -> int:
        sent.append(request)
        return ret

    gateway.api = SimpleNamespace(ReqForQuoteInsert=insert)
    return gateway, sent


def test_request_quote_sends_for_quote_insert_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """发出询价返回询价引用, 引用与报单共用递增序号, 交易所留空."""
    gateway, sent = _gateway(monkeypatch)
    assert gateway.request_quote("m2701-C-3000") == "100"
    assert gateway.request_quote("m2701-P-2800") == "101"
    assert [
        (r.BrokerID, r.InvestorID, r.UserID, r.InstrumentID, r.ForQuoteRef)
        for r in sent
    ] == [
        ("9999", "tester", "tester", "m2701-C-3000", "100"),
        ("9999", "tester", "tester", "m2701-P-2800", "101"),
    ]
    assert not hasattr(sent[0], "ExchangeID")


def test_request_quote_reports_send_failure_and_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """请求函数返回非 0 时报出返回码; 柜台未就绪时不发."""
    gateway, _ = _gateway(monkeypatch, ret=-3)
    with pytest.raises(RuntimeError, match="ReqForQuoteInsert 返回 -3"):
        gateway.request_quote("m2701-C-3000")
    gateway.ready_to_trade = False
    with pytest.raises(RuntimeError, match="not ready for quote requests"):
        gateway.request_quote("m2701-C-3000")


def test_rejected_quote_request_is_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """柜台对同一笔拒绝回报两次, 只记一条 ERROR, 写明合约、引用与原因."""
    gateway, _ = _gateway(monkeypatch)
    field = SimpleNamespace(InstrumentID="xx9999-C-1", ForQuoteRef="7")
    info = SimpleNamespace(ErrorID=16, ErrorMsg="CTP:找不到合约")
    with caplog.at_level(logging.INFO, logger="akquant.gateway.ctp_native"):
        gateway.OnRspForQuoteInsert(field, info, 1, True)
        gateway.OnErrRtnForQuoteInsert(field, info)
        gateway.OnRspForQuoteInsert(field, SimpleNamespace(ErrorID=0), 2, True)
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [
        "CTP quote request rejected: symbol=xx9999-C-1 ref=7 error_id=16 "
        "msg=CTP:找不到合约"
    ]


def test_broker_execution_routes_quote_request_to_gateway() -> None:
    """实盘执行后端把询价交给柜台网关; 网关没有实现时报出网关类型."""
    gateway = SimpleNamespace(request_quote=lambda symbol: f"ref-{symbol}")
    execution = BrokerExecution(object(), gateway, None, None)  # type: ignore[arg-type]
    assert execution.request_quote("m2701-C-3000") == "ref-m2701-C-3000"
    with pytest.raises(RuntimeError, match="没有实现 request_quote"):
        BrokerExecution(object(), SimpleNamespace(), None, None).request_quote(  # type: ignore[arg-type]
            "m2701-C-3000"
        )


class _QuoteEveryBar(Strategy):
    """每根 bar 询价一次, 记录返回值与未登记合约的报错."""

    results: List[Any] = []
    errors: List[str] = []

    def on_bar(self, bar: Bar) -> None:
        self.results.append(self.request_quote(bar.symbol))
        with pytest.raises(KeyError) as exc:
            self.request_quote("nope")
        self.errors.append(str(exc.value))


def test_backtest_quote_request_warns_once_and_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """回测没有做市商: 询价返回 None, 只在首次记一条警告; 未登记合约报 KeyError."""
    _QuoteEveryBar.results, _QuoteEveryBar.errors = [], []
    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-05", periods=3, freq="D", tz="Asia/Shanghai"
            ),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100,
            "symbol": "AAA",
        }
    )
    with caplog.at_level(logging.WARNING, logger="akquant.execution.sim"):
        akquant.run_backtest(data=data, strategy=_QuoteEveryBar, show_progress=False)
    assert _QuoteEveryBar.results == [None, None, None]
    assert len(_QuoteEveryBar.errors) == 3
    warnings = [
        r.getMessage() for r in caplog.records if r.name == "akquant.execution.sim"
    ]
    assert warnings == [
        "回测与 paper 没有做市商, 询价(AAA)不产生报价; 本会话不再重复提示"
    ]
