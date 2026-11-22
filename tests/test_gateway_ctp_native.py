import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from akquant.gateway import ctp_native
from akquant.gateway.ctp_native import CTPMarketGateway, CTPTraderGateway


def _build_native_gateway(monkeypatch: pytest.MonkeyPatch) -> CTPTraderGateway:
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcQryTradingAccountField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcQryInvestorPositionField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcQryTradeField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(ctp_native.tdapi, "THOST_FTDC_PD_Long", "2", raising=False)
    monkeypatch.setattr(ctp_native.tdapi, "THOST_FTDC_PD_Short", "3", raising=False)
    monkeypatch.setattr(ctp_native.tdapi, "THOST_FTDC_PSD_Today", "1", raising=False)
    monkeypatch.setattr(ctp_native.tdapi, "THOST_FTDC_PSD_History", "2", raising=False)

    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.broker_id = "9999"
    gateway.user_id = "tester"
    gateway.password = ""
    gateway.auth_code = ""
    gateway.app_id = "app"
    gateway.front_url = "tcp://test"
    gateway.api = SimpleNamespace()
    gateway.req_id = 0
    gateway.connected = True
    gateway.login_status = True
    gateway.ready_to_trade = True
    gateway.front_id = 0
    gateway.session_id = 0
    gateway.order_ref = 1
    gateway.order_callback = None
    gateway.trade_callback = None
    gateway.error_callback = None
    gateway.order_ref_to_client_order_id = {}
    gateway.order_ref_to_symbol = {}
    gateway._account_query_timeout_sec = 0.1
    gateway._account_query_event = threading.Event()
    gateway._account_query_result = None
    gateway._account_query_error = ""
    gateway._position_query_timeout_sec = 0.1
    gateway._position_query_event = threading.Event()
    gateway._position_query_results = {}
    gateway._position_query_error = ""
    gateway._trade_query_timeout_sec = 0.1
    gateway._trade_query_event = threading.Event()
    gateway._trade_query_results = []
    gateway._trade_query_error = ""
    return gateway


def test_ctp_native_query_account_returns_unified_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native CTP account query should normalize account fields."""
    gateway = _build_native_gateway(monkeypatch)

    def _req_qry_trading_account(request: object, req_id: int) -> int:
        assert getattr(request, "BrokerID") == "9999"
        assert getattr(request, "InvestorID") == "tester"
        gateway.OnRspQryTradingAccount(
            SimpleNamespace(
                AccountID="acct-001",
                Balance=120000.5,
                Available=98000.25,
                CurrMargin=15000.0,
                FrozenMargin=4000.0,
                FrozenCash=500.0,
                FrozenCommission=12.5,
                PositionProfit=-320.0,
            ),
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            True,
        )
        return 0

    gateway.api.ReqQryTradingAccount = _req_qry_trading_account

    account = gateway.query_account()

    assert account is not None
    assert account["account_id"] == "acct-001"
    assert account["equity"] == 120000.5
    assert account["cash"] == 120000.5
    assert account["available_cash"] == 98000.25
    assert account["margin"] == 15000.0
    # 挂单冻结 = 冻结保证金 + 冻结资金(期权权利金等) + 冻结手续费
    assert account["frozen_cash"] == 4512.5
    assert account["unrealized_pnl"] == -320.0


def test_ctp_native_query_positions_aggregates_today_yesterday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native CTP position query should aggregate today/yesterday quantities."""
    gateway = _build_native_gateway(monkeypatch)

    def _req_qry_investor_position(request: object, req_id: int) -> int:
        assert getattr(request, "BrokerID") == "9999"
        assert getattr(request, "InvestorID") == "tester"
        gateway.OnRspQryInvestorPosition(
            SimpleNamespace(
                InstrumentID="au2606",
                PosiDirection="2",
                Position=5,
                TodayPosition=2,
                YdPosition=3,
                ShortFrozen=2,
                PositionPrice=510.5,
            ),
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            False,
        )
        gateway.OnRspQryInvestorPosition(
            None,
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            True,
        )
        return 0

    gateway.api.ReqQryInvestorPosition = _req_qry_investor_position

    positions = gateway.query_positions()

    assert positions == [
        {
            "symbol": "au2606",
            "direction": "Buy",
            "quantity": 5.0,
            "available_quantity": 3.0,
            "today_quantity": 2.0,
            "yesterday_quantity": 3.0,
            "available_today_quantity": 2.0,
            "available_yesterday_quantity": 1.0,
            "avg_price": 510.5,
            "timestamp_ns": positions[0]["timestamp_ns"],
        }
    ]


def test_ctp_native_query_positions_raises_on_rsp_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native CTP position query should surface broker query errors."""
    gateway = _build_native_gateway(monkeypatch)

    def _req_qry_investor_position(request: object, req_id: int) -> int:
        _ = request
        gateway.OnRspQryInvestorPosition(
            None,
            SimpleNamespace(ErrorID=7, ErrorMsg="position query failed"),
            req_id,
            True,
        )
        return 0

    gateway.api.ReqQryInvestorPosition = _req_qry_investor_position

    with pytest.raises(RuntimeError, match="position query failed"):
        gateway.query_positions()


def test_ctp_native_query_trades_today_returns_trade_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native CTP trade query should normalize queried trades."""
    gateway = _build_native_gateway(monkeypatch)
    gateway.order_ref_to_client_order_id["42"] = "coid-42"

    def _req_qry_trade(request: object, req_id: int) -> int:
        assert getattr(request, "BrokerID") == "9999"
        assert getattr(request, "InvestorID") == "tester"
        gateway.OnRspQryTrade(
            SimpleNamespace(
                TradeID="t42",
                OrderRef="42",
                OrderSysID="sys42",
                InstrumentID="au2606",
                Direction="0",
                Volume=2,
                Price=520.0,
                OffsetFlag="3",
            ),
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            False,
        )
        gateway.OnRspQryTrade(
            None,
            SimpleNamespace(ErrorID=0, ErrorMsg=""),
            req_id,
            True,
        )
        return 0

    gateway.api.ReqQryTrade = _req_qry_trade

    trades = gateway.query_trades_today()

    assert len(trades) == 1
    trade = trades[0]
    assert trade["trade_id"] == "t42"
    assert trade["client_order_id"] == "coid-42"
    assert trade["symbol"] == "au2606"
    assert trade["side"] == "Buy"
    assert trade["quantity"] == 2.0
    assert trade["price"] == 520.0
    assert trade["position_effect"] == "close_today"


def test_ctp_native_trader_logs_connection_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Native trader gateway should log connect and auth lifecycle events."""
    gateway = _build_native_gateway(monkeypatch)
    monkeypatch.setattr(
        ctp_native.tdapi,
        "CThostFtdcReqAuthenticateField",
        lambda: SimpleNamespace(),
        raising=False,
    )
    gateway.api = SimpleNamespace(ReqAuthenticate=lambda request, req_id: 0)

    with caplog.at_level("INFO", logger="akquant.gateway.ctp_native"):
        gateway.OnFrontConnected()

    connected_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "CTP trader front connected: tcp://test"
    )
    typed_connected = connected_record
    assert cast(Any, typed_connected).phase == "gateway"

    auth_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Requesting CTP trader authentication (req_id=1)"
    )
    typed_auth = auth_record
    assert cast(Any, typed_auth).phase == "gateway"


def test_ctp_native_market_logs_subscribe_failures_with_symbol(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Native market gateway should log subscribe failures with symbol context."""
    gateway = CTPMarketGateway.__new__(CTPMarketGateway)

    with caplog.at_level("ERROR", logger="akquant.gateway.ctp_native"):
        gateway.OnRspSubMarketData(
            SimpleNamespace(InstrumentID="au2606"),
            SimpleNamespace(ErrorID=7, ErrorMsg="subscribe failed"),
            1,
            True,
        )

    record = next(
        record
        for record in caplog.records
        if record.getMessage()
        == "CTP market subscribe failed for [au2606]: subscribe failed"
    )
    typed_record = cast(Any, record)
    assert typed_record.phase == "gateway"
    assert typed_record.symbol == "au2606"


def test_ctp_native_position_rows_sum_margin_and_position_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一合约同一方向的今仓行、昨仓行: 占用保证金与持仓盈亏相加."""
    gateway = _build_native_gateway(monkeypatch)
    gateway._position_query_results = {}
    for use_margin, profit in ((10000.0, -120.0), (15000.0, 300.0)):
        gateway._accumulate_position_query_row(
            SimpleNamespace(
                InstrumentID="au2606",
                PosiDirection="2",
                Position=1,
                UseMargin=use_margin,
                PositionProfit=profit,
            )
        )

    (row,) = gateway._position_query_results.values()
    assert row["margin"] == 25000.0
    assert row["unrealized_pnl"] == 180.0


def test_ctp_native_order_return_carries_side_quantity_and_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """委托回报带上买卖方向、委托数量、委托价."""
    gateway = _build_native_gateway(monkeypatch)
    seen: list[dict[str, object]] = []
    gateway.order_callback = seen.append
    gateway.OnRtnOrder(
        SimpleNamespace(
            OrderRef="7",
            InstrumentID="au2606",
            OrderStatus="3",
            Direction="1",
            VolumeTotalOriginal=3,
            LimitPrice=512.5,
        )
    )

    (payload,) = seen
    assert (payload["side"], payload["quantity"], payload["price"]) == (
        "Sell",
        3.0,
        512.5,
    )


def test_ctp_native_market_updates_subscriptions_after_login() -> None:
    """登录后替换订阅集: 新增的立即订阅、移除的立即退订; 登录前只记下."""
    calls: list[tuple[str, list[bytes], int]] = []
    gateway = CTPMarketGateway.__new__(CTPMarketGateway)
    gateway.symbols = ["au2606", "ag2606"]

    def recorder(kind: str) -> Any:
        def call(symbols: list[bytes], count: int) -> int:
            calls.append((kind, symbols, count))
            return 0

        return call

    gateway.api = SimpleNamespace(
        SubscribeMarketData=recorder("sub"), UnSubscribeMarketData=recorder("unsub")
    )

    gateway.logged_in = False
    gateway.update_subscriptions(["au2606", "cu2606"])
    assert gateway.symbols == ["au2606", "cu2606"]
    assert calls == []

    gateway.logged_in = True
    gateway.update_subscriptions(["au2606", "m2701-C-3000"])
    assert gateway.symbols == ["au2606", "m2701-C-3000"]
    assert calls == [
        ("sub", [b"m2701-C-3000"], 1),
        ("unsub", [b"cu2606"], 1),
    ]
