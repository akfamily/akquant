"""CTP 下单路径: 本地持仓账拆平今/平昨、查询流控重试、市价单保护价、昨仓口径."""

import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from akquant.gateway import ctp_native
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.brokers.ctp.offset import OffsetBook
from akquant.gateway.ctp_native import CTPTraderGateway
from akquant.gateway.order_submitter import BrokerOrderSubmitter


def _row(symbol: str, direction: str, today: float, yd: float, exch: str) -> Dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "today_quantity": today,
        "yesterday_quantity": yd,
        "exchange_id": exch,
    }


def test_offset_book_requires_sync() -> None:
    """持仓账未由持仓查询同步时拒绝拆分, 不猜测."""
    with pytest.raises(ValueError) as exc:
        OffsetBook().split_close("rb2701", "sell", 1)
    assert (
        str(exc.value) == "rb2701 平仓无法拆分今昨仓: 本地持仓账尚未从柜台持仓查询同步"
    )


def test_offset_book_shfe_today_first_with_frozen() -> None:
    """上期所先平今再平昨, 扣除平仓挂单冻结; 不足时写明今昨仓与冻结量."""
    book = OffsetBook()
    book.apply_snapshot([_row("rb2701", "Buy", 2, 3, "SHFE")])
    assert book.split_close("rb2701", "sell", 4) == [
        ("close_today", 2),
        ("close_yesterday", 2),
    ]
    book.update_order("1:1:7", "rb2701", "SHFE", "sell", "close_today", 1, True)
    assert book.split_close("rb2701", "sell", 4) == [
        ("close_today", 1),
        ("close_yesterday", 3),
    ]
    with pytest.raises(ValueError) as exc:
        book.split_close("rb2701", "sell", 5)
    assert str(exc.value) == (
        "rb2701 平多头 5 手, 超过可平 4 手(今仓 2、昨仓 3, 平仓挂单冻结今仓 1、昨仓 0)"
    )
    # 挂单成交: 冻结解除, 今仓减少
    book.update_order("1:1:7", "rb2701", "SHFE", "sell", "close_today", 0, False)
    book.update_trade("rb2701", "SHFE", "sell", "close_today", 1)
    # 普通平仓标志在上期所只平昨仓
    book.update_trade("rb2701", "SHFE", "sell", "close", 1)
    book.update_trade("rb2701", "SHFE", "buy", "open", 2)
    assert book.split_close("rb2701", "sell", 5) == [
        ("close_today", 3),
        ("close_yesterday", 2),
    ]
    # 买入平空头: 没有空头持仓
    with pytest.raises(ValueError, match="^rb2701 平空头 1 手, 超过可平 0 手"):
        book.split_close("rb2701", "buy", 1)


def test_offset_book_other_exchanges_use_plain_close() -> None:
    """不区分今昨仓的交易所整笔普通平仓, 按合计可平数量校验."""
    book = OffsetBook()
    book.apply_snapshot([_row("m2701", "Sell", 1, 2, "DCE")])
    assert book.split_close("m2701", "buy", 3) == [("close", 3)]
    book.update_order("1:1:9", "m2701", "DCE", "buy", "close", 2, True)
    with pytest.raises(ValueError, match="超过可平 1 手"):
        book.split_close("m2701", "buy", 2)
    # 只重置单个合约的查询不影响其他合约
    book.apply_snapshot([_row("MA701", "Buy", 1, 0, "CZCE")], symbol="MA701")
    assert book.split_close("MA701", "sell", 1) == [("close", 1)]
    assert book.split_close("m2701", "buy", 1) == [("close", 1)]


def _gateway(monkeypatch: pytest.MonkeyPatch) -> CTPTraderGateway:
    for name, value in {
        "THOST_FTDC_D_Buy": "0",
        "THOST_FTDC_D_Sell": "1",
        "THOST_FTDC_OF_Open": "0",
        "THOST_FTDC_OF_Close": "1",
        "THOST_FTDC_OF_CloseToday": "3",
        "THOST_FTDC_OF_CloseYesterday": "4",
        "THOST_FTDC_HF_Speculation": "1",
        "THOST_FTDC_OPT_AnyPrice": "1",
        "THOST_FTDC_OPT_LimitPrice": "2",
        "THOST_FTDC_TC_IOC": "1",
        "THOST_FTDC_TC_GFD": "3",
        "THOST_FTDC_VC_AV": "1",
        "THOST_FTDC_VC_CV": "3",
        "THOST_FTDC_CC_Immediately": "1",
        "THOST_FTDC_FCC_NotForceClose": "0",
        "THOST_FTDC_PD_Long": "2",
        "THOST_FTDC_PD_Short": "3",
        "THOST_FTDC_PSD_Today": "1",
        "THOST_FTDC_PSD_History": "2",
    }.items():
        monkeypatch.setattr(ctp_native.tdapi, name, value, raising=False)
    for field in (
        "CThostFtdcInputOrderField",
        "CThostFtdcQryTradingAccountField",
        "CThostFtdcQryInvestorPositionField",
        "CThostFtdcQryDepthMarketDataField",
    ):
        monkeypatch.setattr(ctp_native.tdapi, field, SimpleNamespace, raising=False)
    gateway = CTPTraderGateway.__new__(CTPTraderGateway)
    gateway.__dict__.update(
        broker_id="9999",
        user_id="tester",
        api=SimpleNamespace(),
        req_id=0,
        connected=True,
        login_status=True,
        ready_to_trade=True,
        front_id=1,
        session_id=7,
        order_ref=1,
        order_callback=None,
        trade_callback=None,
        error_callback=None,
        order_ref_to_client_order_id={},
        order_ref_to_symbol={},
        _account_query_timeout_sec=0.5,
        _account_query_event=threading.Event(),
        _account_query_result=None,
        _account_query_error="",
        _position_query_timeout_sec=0.5,
        _position_query_event=threading.Event(),
        _position_query_results={},
        _position_query_error="",
        _depth_query_timeout_sec=0.5,
        _depth_query_event=threading.Event(),
        _depth_query_rows=[],
        _depth_query_error="",
        _query_lock=threading.RLock(),
        _query_retry_interval_sec=0.0,
        offset_book=OffsetBook(),
        _symbol_exchange={},
    )
    return gateway


OK = SimpleNamespace(ErrorID=0, ErrorMsg="")


def test_yesterday_position_excludes_closed_yesterday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YdPosition 是上日持仓; 当前昨仓按 Position − TodayPosition 计."""
    gateway = _gateway(monkeypatch)

    def query(request: Any, req_id: int) -> int:
        gateway.OnRspQryInvestorPosition(
            SimpleNamespace(
                InstrumentID="m2701",
                ExchangeID="DCE",
                PosiDirection="2",
                Position=3,
                TodayPosition=1,
                YdPosition=5,
            ),
            OK,
            req_id,
            True,
        )
        return 0

    gateway.api.ReqQryInvestorPosition = query
    [row] = gateway.query_positions()
    assert (row["today_quantity"], row["yesterday_quantity"]) == (1.0, 2.0)
    assert row["exchange_id"] == "DCE"
    # 全量持仓查询同步了本地持仓账
    assert gateway.split_close("m2701", "Sell", 3) == [("close", 3.0)]


def test_query_retries_on_flow_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """查询被流控(-3)时重发, 超过次数后报出返回码含义."""
    gateway = _gateway(monkeypatch)
    calls: List[int] = []

    def query(request: Any, req_id: int) -> int:
        calls.append(req_id)
        if len(calls) < 3:
            return -3
        gateway.OnRspQryTradingAccount(
            SimpleNamespace(Balance=100.0, Available=90.0), OK, req_id, True
        )
        return 0

    gateway.api.ReqQryTradingAccount = query
    account = gateway.query_account()
    assert account is not None and account["available_cash"] == 90.0
    assert len(calls) == 3

    gateway.api.ReqQryTradingAccount = lambda request, req_id: -3
    with pytest.raises(RuntimeError) as exc:
        gateway.query_account()
    assert str(exc.value) == (
        "ReqQryTradingAccount failed with code=-3(每秒发送请求数超过许可数), "
        "已重发 5 次"
    )


def _depth(gateway: CTPTraderGateway, rows: List[Any], counter: List[str]) -> None:
    def query(request: Any, req_id: int) -> int:
        counter.append(request.InstrumentID)
        for i, row in enumerate(rows):
            gateway.OnRspQryDepthMarketData(row, OK, req_id, i == len(rows) - 1)
        return 0

    gateway.api.ReqQryDepthMarketData = query


def _market(gateway: CTPTraderGateway, symbol: str, side: str) -> Any:
    sent: List[Any] = []

    def insert(request: Any, req_id: int) -> int:
        sent.append(request)
        return 0

    gateway.api.ReqOrderInsert = insert
    gateway.insert_order(
        client_order_id="c",
        symbol=symbol,
        side=side,
        quantity=1,
        price=None,
        order_type="market",
        time_in_force="IOC",
        position_effect="open",
    )
    return sent[0]


def test_market_order_protection_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """上期所/能源中心市价单以涨跌停价为保护价; 其他交易所不带价格且只查一次交易所."""
    gateway = _gateway(monkeypatch)
    queried: List[str] = []
    rb = SimpleNamespace(
        InstrumentID="rb2701",
        ExchangeID="SHFE",
        UpperLimitPrice=3132.0,
        LowerLimitPrice=2833.0,
    )
    # 深度行情按前缀匹配, 期权行先到也只取代码完全相同的一行
    rb_option = SimpleNamespace(
        InstrumentID="rb2701C3000",
        ExchangeID="SHFE",
        UpperLimitPrice=1.0,
        LowerLimitPrice=0.5,
    )
    _depth(gateway, [rb_option, rb], queried)
    buy = _market(gateway, "rb2701", "Buy")
    assert (buy.OrderPriceType, buy.LimitPrice, buy.TimeCondition) == ("1", 3132.0, "1")
    assert _market(gateway, "rb2701", "Sell").LimitPrice == 2833.0

    m = SimpleNamespace(
        InstrumentID="m2701",
        ExchangeID="DCE",
        UpperLimitPrice=3249.0,
        LowerLimitPrice=2883.0,
    )
    _depth(gateway, [m], queried)
    assert _market(gateway, "m2701", "Buy").LimitPrice == 0.0
    assert _market(gateway, "m2701", "Sell").LimitPrice == 0.0
    assert queried == ["rb2701", "rb2701", "m2701"]


def test_market_order_without_limit_price_is_rejected_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取不到有效涨停价时本地拒绝, 不报 0 价, 也不占用报单引用."""
    gateway = _gateway(monkeypatch)
    sc = SimpleNamespace(
        InstrumentID="sc2701",
        ExchangeID="INE",
        UpperLimitPrice=1.7976931348623157e308,
        LowerLimitPrice=473.0,
    )
    _depth(gateway, [sc], [])
    with pytest.raises(ValueError) as exc:
        _market(gateway, "sc2701", "Buy")
    assert str(exc.value) == (
        "sc2701 行情中没有有效的涨停价, 不能报INE市价单(须以涨跌停价为保护价)"
    )
    assert gateway.order_ref == 1


def test_order_and_trade_returns_update_offset_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """委托回报计入挂单冻结, 成交回报增减今昨仓."""
    gateway = _gateway(monkeypatch)
    gateway.offset_book.apply_snapshot([_row("rb2701", "Buy", 0, 2, "SHFE")])
    gateway.OnRtnTrade(
        SimpleNamespace(
            InstrumentID="rb2701",
            ExchangeID="SHFE",
            Direction="0",
            OffsetFlag="0",
            Volume=3,
            Price=3000.0,
        )
    )
    gateway.OnRtnOrder(
        SimpleNamespace(
            InstrumentID="rb2701",
            ExchangeID="SHFE",
            OrderRef="5",
            FrontID=1,
            SessionID=7,
            Direction="1",
            CombOffsetFlag="4",
            VolumeTotalOriginal=2,
            VolumeTraded=0,
            VolumeTotal=2,
            OrderStatus="3",
        )
    )
    assert gateway.split_close("rb2701", "Sell", 3) == [("close_today", 3.0)]
    with pytest.raises(ValueError, match="平仓挂单冻结今仓 0、昨仓 2"):
        gateway.split_close("rb2701", "Sell", 4)


def test_submitter_rejects_locally_when_split_fails() -> None:
    """柜台网关拆分平仓失败时本地拒单, on_reject 收到原因, 不报单."""
    placed: List[UnifiedOrderRequest] = []
    rejects: List[str] = []

    class _Gw:
        def split_close(self, symbol: str, side: str, quantity: float) -> Any:
            raise ValueError("rb2701 平多头 5 手, 超过可平 4 手")

        def place_order(self, req: UnifiedOrderRequest) -> str:
            placed.append(req)
            return "b1"

    class _Strategy:
        _owner_strategy_id = "_default"
        broker_ready = True

        def on_order(self, order: Any) -> None:
            pass

        def on_reject(self, order: Any) -> None:
            rejects.append(order.reject_reason)

    cap = BrokerCapability(
        broker_name="ctp",
        position_effect=True,
        position_details=True,
        supported_position_effects=("open", "close", "close_today", "close_yesterday"),
    )
    submitter = BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=_Strategy(),
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )
    receipt = submitter.submit_order(
        symbol="rb2701",
        side="Sell",
        quantity=5,
        price=3000.0,
        order_type="Limit",
        position_effect="close",
    )
    assert receipt.failure == "rejected"
    assert placed == []
    assert rejects == ["rb2701 平多头 5 手, 超过可平 4 手"]


def test_auto_and_reduce_only_use_offset_book() -> None:
    """开平自动判断: 先平到可平数量再开仓, 只减仓不开仓; 不在下单路径上查询持仓."""
    from akquant.gateway.order_submitter import resolve_live_order_legs

    book = OffsetBook()
    book.apply_snapshot([_row("rb2701", "Buy", 1, 1, "SHFE")])

    class _Gw:
        def query_positions(self) -> Any:
            raise AssertionError("下单路径不应查询持仓")

        def closable_quantity(self, symbol: str, side: str) -> float:
            return book.closable(symbol, side.lower())

        def split_close(self, symbol: str, side: str, quantity: float) -> Any:
            return book.split_close(symbol, side.lower(), quantity)

    cap = BrokerCapability(
        broker_name="ctp",
        position_effect=True,
        position_details=True,
        supported_position_effects=("auto", "open", "close"),
    )

    def legs(side: str, qty: float, effect: str, reduce_only: bool = False) -> Any:
        return resolve_live_order_legs(
            _Gw(), cap, "rb2701", side, qty, effect, reduce_only, getattr
        )

    # 卖出不超过可平: 只平不开(此前原样报 auto, CTP 映射成开仓)
    assert legs("Sell", 1, "auto") == [("close_today", 1)]
    assert legs("Sell", 3, "auto") == [
        ("close_today", 1),
        ("close_yesterday", 1),
        ("open", 1),
    ]
    assert legs("Buy", 2, "auto") == [("open", 2)]
    # 只减仓: close 照样拆今昨, auto 不开仓, 没有可平持仓时拒绝
    assert legs("Sell", 2, "close", reduce_only=True) == [
        ("close_today", 1),
        ("close_yesterday", 1),
    ]
    assert legs("Sell", 3, "auto", reduce_only=True) == [
        ("close_today", 1),
        ("close_yesterday", 1),
    ]
    with pytest.raises(ValueError) as exc:
        legs("Buy", 1, "auto", reduce_only=True)
    assert str(exc.value) == "rb2701 只减仓委托没有可平持仓"
