# -*- coding: utf-8 -*-
"""
CTP Gateway Implementation for AKQuant.

This module provides CTP (China Futures) connectivity using openctp-ctp library.
"""

import threading
import time
from typing import Any, Callable, List, Optional

from ....akquant import BarAggregator, DataFeed, Tick
from ....log import build_log_extra, get_logger

try:
    from openctp_ctp import thostmduserapi as mdapi  # type: ignore
    from openctp_ctp import thosttraderapi as tdapi  # type: ignore

    HAS_OPENCTP = True
except ImportError:
    HAS_OPENCTP = False

    class MockSpi:
        """Mock SPI class for when openctp-ctp is not installed."""

        pass

    class MockMdApi:
        """Mock market API namespace."""

        CThostFtdcMdSpi = MockSpi

    class MockTdApi:
        """Mock trader API namespace."""

        CThostFtdcTraderSpi = MockSpi

    mdapi = MockMdApi  # type: ignore
    tdapi = MockTdApi  # type: ignore


logger = get_logger("gateway.ctp_native")


class CTPTraderGateway(tdapi.CThostFtdcTraderSpi):  # type: ignore
    """
    CTP Trading Gateway.

    Handles connection, authentication, login, and settlement confirmation.
    """

    def __init__(
        self,
        front_url: str,
        broker_id: str = "9999",
        user_id: str = "",
        password: str = "",
        auth_code: str = "0000000000000000",
        app_id: str = "simnow_client_test",
    ):
        """
        Initialize the CTP Trader Gateway.

        :param front_url: CTP Front URL
        :param broker_id: Broker ID
        :param user_id: User ID
        :param password: Password
        :param auth_code: Authentication Code
        :param app_id: App ID
        """
        if not HAS_OPENCTP:
            raise ImportError(
                "openctp-ctp is not installed. Please run `pip install openctp-ctp`."
            )

        tdapi.CThostFtdcTraderSpi.__init__(self)
        self.front_url = front_url
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.auth_code = auth_code
        self.app_id = app_id

        self.api: Any = None
        self.req_id = 0
        self.connected = False
        self.authenticated = False
        self.login_status = False
        self.ready_to_trade = False
        self.front_id = 0
        self.session_id = 0
        self.order_ref = 1
        self.order_callback: Callable[[dict[str, Any]], None] | None = None
        self.trade_callback: Callable[[dict[str, Any]], None] | None = None
        self.error_callback: Callable[[dict[str, Any]], None] | None = None
        self.order_ref_to_client_order_id: dict[str, str] = {}
        self.order_ref_to_symbol: dict[str, str] = {}
        self._account_query_timeout_sec = 2.0
        self._account_query_event = threading.Event()
        self._account_query_result: dict[str, Any] | None = None
        self._account_query_error = ""
        self._position_query_timeout_sec = 2.0
        self._position_query_event = threading.Event()
        self._position_query_results: dict[tuple[str, str], dict[str, Any]] = {}
        self._position_query_error = ""
        self._trade_query_timeout_sec = 2.0
        self._trade_query_event = threading.Event()
        self._trade_query_results: list[dict[str, Any]] = []
        self._trade_query_error = ""

    def _log_extra(self, *, symbol: str | None = None) -> dict[str, Any]:
        return build_log_extra(phase="gateway", symbol=symbol)

    def start(self) -> None:
        """Start the CTP Trader API in a blocking way (should be run in a thread)."""
        logger.info(
            "Connecting native CTP trader gateway to %s",
            self.front_url,
            extra=self._log_extra(),
        )
        try:
            self.api = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi()
            self.api.RegisterFront(self.front_url)
            self.api.RegisterSpi(self)
            self.api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
            self.api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
            self.api.Init()
            logger.info(
                "Native CTP trader API initialized; joining event loop",
                extra=self._log_extra(),
            )
            self.api.Join()
        except Exception:
            logger.exception(
                "Failed to start native CTP trader gateway",
                extra=self._log_extra(),
            )

    def set_order_handler(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Set native order event handler."""
        self.order_callback = callback

    def set_trade_handler(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Set native trade event handler."""
        self.trade_callback = callback

    def set_error_handler(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Set native error event handler."""
        self.error_callback = callback

    def can_trade(self) -> bool:
        """Return whether the trader channel is ready for order requests."""
        return self.connected and self.login_status and self.ready_to_trade

    def insert_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None,
        order_type: str,
        time_in_force: str,
        position_effect: str = "auto",
    ) -> dict[str, Any]:
        """Send CTP order insert request."""
        if not self.can_trade():
            raise RuntimeError("CTP trader is not ready for trading")
        if self.api is None:
            raise RuntimeError("CTP trader API is not initialized")

        order_ref_text = str(self.order_ref)
        self.order_ref += 1
        self.order_ref_to_client_order_id[order_ref_text] = client_order_id
        self.order_ref_to_symbol[order_ref_text] = symbol

        request = tdapi.CThostFtdcInputOrderField()
        request.BrokerID = self.broker_id
        request.InvestorID = self.user_id
        request.ExchangeID = ""
        request.InstrumentID = symbol
        request.OrderRef = order_ref_text
        request.UserID = self.user_id
        request.Direction = (
            tdapi.THOST_FTDC_D_Buy
            if str(side).strip().lower() == "buy"
            else tdapi.THOST_FTDC_D_Sell
        )
        normalized_effect = self._normalize_position_effect(position_effect)
        request.CombOffsetFlag = self._map_position_effect_to_ctp_offset(
            normalized_effect
        )
        request.CombHedgeFlag = tdapi.THOST_FTDC_HF_Speculation
        request.VolumeTotalOriginal = int(max(1, round(quantity)))

        order_type_key = str(order_type).strip().lower()
        tif_key = str(time_in_force).strip().upper()
        if order_type_key == "market":
            request.OrderPriceType = tdapi.THOST_FTDC_OPT_AnyPrice
            request.LimitPrice = 0.0
            request.TimeCondition = tdapi.THOST_FTDC_TC_IOC
            request.VolumeCondition = tdapi.THOST_FTDC_VC_AV
        else:
            if price is None or price <= 0:
                raise ValueError("limit price must be positive for non-market orders")
            request.OrderPriceType = tdapi.THOST_FTDC_OPT_LimitPrice
            request.LimitPrice = float(price)
            if tif_key in {"IOC", "FAK", "FOK"}:
                request.TimeCondition = tdapi.THOST_FTDC_TC_IOC
                if tif_key == "FOK":
                    request.VolumeCondition = tdapi.THOST_FTDC_VC_CV
                else:
                    request.VolumeCondition = tdapi.THOST_FTDC_VC_AV
            else:
                request.TimeCondition = tdapi.THOST_FTDC_TC_GFD
                request.VolumeCondition = tdapi.THOST_FTDC_VC_AV

        request.MinVolume = 1
        request.ContingentCondition = tdapi.THOST_FTDC_CC_Immediately
        request.ForceCloseReason = tdapi.THOST_FTDC_FCC_NotForceClose
        request.IsAutoSuspend = 0
        request.UserForceClose = 0

        self.req_id += 1
        ret = self.api.ReqOrderInsert(request, self.req_id)
        if ret != 0:
            self.order_ref_to_client_order_id.pop(order_ref_text, None)
            self.order_ref_to_symbol.pop(order_ref_text, None)
            raise RuntimeError(f"ReqOrderInsert failed with code={ret}")
        return {
            "broker_order_id": self._make_broker_order_id(
                front_id=self.front_id,
                session_id=self.session_id,
                order_ref=order_ref_text,
                order_sys_id="",
            ),
            "order_ref": order_ref_text,
            "front_id": self.front_id,
            "session_id": self.session_id,
            "timestamp_ns": time.time_ns(),
        }

    def cancel_order(self, broker_order_id: str) -> None:
        """Send CTP cancel order request."""
        if not self.can_trade():
            raise RuntimeError("CTP trader is not ready for cancel requests")
        if self.api is None:
            raise RuntimeError("CTP trader API is not initialized")
        parsed = self._parse_broker_order_id(broker_order_id)
        order_ref = parsed.get("order_ref", "")
        if not order_ref:
            raise ValueError(f"invalid broker_order_id={broker_order_id}")
        request = tdapi.CThostFtdcInputOrderActionField()
        request.BrokerID = self.broker_id
        request.InvestorID = self.user_id
        request.OrderRef = order_ref
        request.FrontID = int(parsed.get("front_id") or self.front_id)
        request.SessionID = int(parsed.get("session_id") or self.session_id)
        request.ActionFlag = tdapi.THOST_FTDC_AF_Delete
        request.InstrumentID = self.order_ref_to_symbol.get(order_ref, "")
        self.req_id += 1
        ret = self.api.ReqOrderAction(request, self.req_id)
        if ret != 0:
            raise RuntimeError(f"ReqOrderAction failed with code={ret}")

    def query_account(self) -> dict[str, Any] | None:
        """Query the CTP trading account snapshot synchronously."""
        if not self.can_trade():
            raise RuntimeError("CTP trader is not ready for account queries")
        if self.api is None:
            raise RuntimeError("CTP trader API is not initialized")

        self._account_query_result = None
        self._account_query_error = ""
        self._account_query_event.clear()

        request = tdapi.CThostFtdcQryTradingAccountField()
        request.BrokerID = self.broker_id
        request.InvestorID = self.user_id

        self.req_id += 1
        ret = self.api.ReqQryTradingAccount(request, self.req_id)
        if ret != 0:
            raise RuntimeError(f"ReqQryTradingAccount failed with code={ret}")
        if not self._account_query_event.wait(timeout=self._account_query_timeout_sec):
            raise RuntimeError("CTP trading account query timed out")
        if self._account_query_error:
            raise RuntimeError(self._account_query_error)
        return self._account_query_result

    def query_positions(self, instrument_id: str | None = None) -> list[dict[str, Any]]:
        """Query and aggregate CTP investor positions synchronously."""
        if not self.can_trade():
            raise RuntimeError("CTP trader is not ready for position queries")
        if self.api is None:
            raise RuntimeError("CTP trader API is not initialized")

        self._position_query_results = {}
        self._position_query_error = ""
        self._position_query_event.clear()

        request = tdapi.CThostFtdcQryInvestorPositionField()
        request.BrokerID = self.broker_id
        request.InvestorID = self.user_id
        if instrument_id:
            request.InstrumentID = instrument_id

        self.req_id += 1
        ret = self.api.ReqQryInvestorPosition(request, self.req_id)
        if ret != 0:
            raise RuntimeError(f"ReqQryInvestorPosition failed with code={ret}")
        if not self._position_query_event.wait(
            timeout=self._position_query_timeout_sec
        ):
            raise RuntimeError("CTP investor position query timed out")
        if self._position_query_error:
            raise RuntimeError(self._position_query_error)
        return list(self._position_query_results.values())

    def query_trades_today(
        self, instrument_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Query today's trade fills from CTP synchronously."""
        if not self.can_trade():
            raise RuntimeError("CTP trader is not ready for trade queries")
        if self.api is None:
            raise RuntimeError("CTP trader API is not initialized")

        self._trade_query_results = []
        self._trade_query_error = ""
        self._trade_query_event.clear()

        request = tdapi.CThostFtdcQryTradeField()
        request.BrokerID = self.broker_id
        request.InvestorID = self.user_id
        if instrument_id:
            request.InstrumentID = instrument_id

        self.req_id += 1
        ret = self.api.ReqQryTrade(request, self.req_id)
        if ret != 0:
            raise RuntimeError(f"ReqQryTrade failed with code={ret}")
        if not self._trade_query_event.wait(timeout=self._trade_query_timeout_sec):
            raise RuntimeError("CTP trade query timed out")
        if self._trade_query_error:
            raise RuntimeError(self._trade_query_error)
        return list(self._trade_query_results)

    def OnFrontConnected(self) -> None:
        """Handle front connection event."""
        logger.info(
            "CTP trader front connected: %s",
            self.front_url,
            extra=self._log_extra(),
        )
        self.connected = True

        self.req_id += 1
        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID = self.broker_id
        req.UserID = self.user_id
        req.AppID = self.app_id
        req.AuthCode = self.auth_code
        logger.info(
            "Requesting CTP trader authentication (req_id=%s)",
            self.req_id,
            extra=self._log_extra(),
        )
        if self.api:
            self.api.ReqAuthenticate(req, self.req_id)

    def OnRspAuthenticate(
        self,
        pRspAuthenticateField: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle authentication response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.error(
                "CTP trader authentication failed: error_id=%s msg=%s",
                pRspInfo.ErrorID,
                pRspInfo.ErrorMsg,
                extra=self._log_extra(),
            )
        else:
            logger.info(
                "CTP trader authentication succeeded",
                extra=self._log_extra(),
            )

        self.authenticated = True

        self.req_id += 1
        req = tdapi.CThostFtdcReqUserLoginField()
        req.BrokerID = self.broker_id
        req.UserID = self.user_id
        req.Password = self.password
        logger.info(
            "Requesting CTP trader user login (req_id=%s)",
            self.req_id,
            extra=self._log_extra(),
        )
        if self.api:
            self.api.ReqUserLogin(req, self.req_id)

    def OnRspUserLogin(
        self, pRspUserLogin: Any, pRspInfo: Any, nRequestID: int, bIsLast: bool
    ) -> None:
        """Handle login response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.error(
                "CTP trader login failed: error_id=%s msg=%s",
                pRspInfo.ErrorID,
                pRspInfo.ErrorMsg,
                extra=self._log_extra(),
            )
            return

        logger.info(
            "CTP trader login succeeded: trading_day=%s broker_id=%s user_id=%s",
            pRspUserLogin.TradingDay,
            pRspUserLogin.BrokerID,
            pRspUserLogin.UserID,
            extra=self._log_extra(),
        )
        self.login_status = True
        self.front_id = int(getattr(pRspUserLogin, "FrontID", 0) or 0)
        self.session_id = int(getattr(pRspUserLogin, "SessionID", 0) or 0)
        max_order_ref = str(getattr(pRspUserLogin, "MaxOrderRef", "")).strip()
        if max_order_ref.isdigit():
            self.order_ref = int(max_order_ref) + 1

        self.req_id += 1
        req = tdapi.CThostFtdcSettlementInfoConfirmField()
        req.BrokerID = self.broker_id
        req.InvestorID = self.user_id
        logger.info(
            "Confirming CTP settlement info",
            extra=self._log_extra(),
        )
        if self.api:
            self.api.ReqSettlementInfoConfirm(req, self.req_id)

    def OnRspSettlementInfoConfirm(
        self,
        pSettlementInfoConfirm: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle settlement info confirmation response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.error(
                "CTP settlement confirmation failed: %s",
                pRspInfo.ErrorMsg,
                extra=self._log_extra(),
            )
            self.ready_to_trade = False
        else:
            logger.info(
                "CTP settlement info confirmed",
                extra=self._log_extra(),
            )
            self.ready_to_trade = True

    def OnFrontDisconnected(self, nReason: int) -> None:
        """Handle front disconnection event."""
        # 系统级致命: 交易前置断开后无法下单/撤单, 实盘失去执行能力, 需人工立即介入
        # (CTP 会自动重连, 但重连成功前的委托意图无法送达)。
        logger.critical(
            "CTP trader front disconnected: reason=%s",
            nReason,
            extra=self._log_extra(),
        )
        self.connected = False
        self.login_status = False
        self.ready_to_trade = False

    def OnRspOrderInsert(
        self, pInputOrder: Any, pRspInfo: Any, nRequestID: int, bIsLast: bool
    ) -> None:
        """Handle order insert response."""
        if pRspInfo is None or pRspInfo.ErrorID == 0:
            return
        self._emit_rejected_order(
            input_order=pInputOrder,
            error_id=getattr(pRspInfo, "ErrorID", 0),
            error_msg=self._to_text(getattr(pRspInfo, "ErrorMsg", "")),
            source="OnRspOrderInsert",
        )

    def OnErrRtnOrderInsert(self, pInputOrder: Any, pRspInfo: Any) -> None:
        """Handle asynchronous order insert reject."""
        self._emit_rejected_order(
            input_order=pInputOrder,
            error_id=getattr(pRspInfo, "ErrorID", 0),
            error_msg=self._to_text(getattr(pRspInfo, "ErrorMsg", "")),
            source="OnErrRtnOrderInsert",
        )

    def OnRspQryTradingAccount(
        self,
        pTradingAccount: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle trading account query response."""
        _ = nRequestID
        error_id = (
            int(getattr(pRspInfo, "ErrorID", 0) or 0) if pRspInfo is not None else 0
        )
        if error_id:
            error_msg = self._to_text(getattr(pRspInfo, "ErrorMsg", ""))
            self._account_query_error = (
                f"CTP trading account query failed: {error_id} {error_msg}".strip()
            )
        if pTradingAccount is not None:
            self._account_query_result = {
                "account_id": self._to_text(
                    getattr(pTradingAccount, "AccountID", self.user_id)
                )
                or self.user_id,
                "equity": float(getattr(pTradingAccount, "Balance", 0.0) or 0.0),
                "cash": float(getattr(pTradingAccount, "Balance", 0.0) or 0.0),
                "available_cash": float(
                    getattr(pTradingAccount, "Available", 0.0) or 0.0
                ),
                "timestamp_ns": time.time_ns(),
            }
        if bIsLast:
            self._account_query_event.set()

    def OnRspQryInvestorPosition(
        self,
        pInvestorPosition: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle investor position query response."""
        _ = nRequestID
        error_id = (
            int(getattr(pRspInfo, "ErrorID", 0) or 0) if pRspInfo is not None else 0
        )
        if error_id:
            error_msg = self._to_text(getattr(pRspInfo, "ErrorMsg", ""))
            self._position_query_error = (
                f"CTP investor position query failed: {error_id} {error_msg}".strip()
            )
        if pInvestorPosition is not None:
            self._accumulate_position_query_row(pInvestorPosition)
        if bIsLast:
            self._position_query_event.set()

    def OnRspQryTrade(
        self,
        pTrade: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle trade query response."""
        _ = nRequestID
        error_id = (
            int(getattr(pRspInfo, "ErrorID", 0) or 0) if pRspInfo is not None else 0
        )
        if error_id:
            error_msg = self._to_text(getattr(pRspInfo, "ErrorMsg", ""))
            self._trade_query_error = (
                f"CTP trade query failed: {error_id} {error_msg}".strip()
            )
        if pTrade is not None:
            self._trade_query_results.append(self._trade_payload_from_ctp_trade(pTrade))
        if bIsLast:
            self._trade_query_event.set()

    def OnRtnOrder(self, pOrder: Any) -> None:
        """Handle order return event."""
        order_ref = self._to_text(getattr(pOrder, "OrderRef", ""))
        order_sys_id = self._to_text(getattr(pOrder, "OrderSysID", "")).strip()
        front_id = int(getattr(pOrder, "FrontID", 0) or self.front_id)
        session_id = int(getattr(pOrder, "SessionID", 0) or self.session_id)
        payload = {
            "client_order_id": self.order_ref_to_client_order_id.get(order_ref, ""),
            "broker_order_id": self._make_broker_order_id(
                front_id=front_id,
                session_id=session_id,
                order_ref=order_ref,
                order_sys_id=order_sys_id,
            ),
            "symbol": self._to_text(getattr(pOrder, "InstrumentID", "")),
            "status": self._map_order_status(getattr(pOrder, "OrderStatus", "")),
            "filled_quantity": float(getattr(pOrder, "VolumeTraded", 0.0) or 0.0),
            "avg_fill_price": float(getattr(pOrder, "LimitPrice", 0.0) or 0.0),
            "reject_reason": self._to_text(getattr(pOrder, "StatusMsg", "")),
            "timestamp_ns": time.time_ns(),
            "order_ref": order_ref,
            "position_effect": self._map_ctp_offset_to_position_effect(
                getattr(pOrder, "CombOffsetFlag", "")
            ),
        }
        if self.order_callback is not None:
            self.order_callback(payload)

    def OnRtnTrade(self, pTrade: Any) -> None:
        """Handle trade return event."""
        payload = self._trade_payload_from_ctp_trade(pTrade)
        if self.trade_callback is not None:
            self.trade_callback(payload)

    def _emit_rejected_order(
        self,
        *,
        input_order: Any,
        error_id: Any,
        error_msg: str,
        source: str,
    ) -> None:
        order_ref = self._to_text(getattr(input_order, "OrderRef", ""))
        broker_order_id = self._make_broker_order_id(
            front_id=self.front_id,
            session_id=self.session_id,
            order_ref=order_ref,
            order_sys_id="",
        )
        payload = {
            "client_order_id": self.order_ref_to_client_order_id.get(order_ref, ""),
            "broker_order_id": broker_order_id,
            "symbol": self._to_text(getattr(input_order, "InstrumentID", "")),
            "status": "rejected",
            "filled_quantity": 0.0,
            "avg_fill_price": float(getattr(input_order, "LimitPrice", 0.0) or 0.0),
            "reject_reason": error_msg,
            "timestamp_ns": time.time_ns(),
            "order_ref": order_ref,
            "error_code": str(error_id),
            "position_effect": self._map_ctp_offset_to_position_effect(
                getattr(input_order, "CombOffsetFlag", "")
            ),
        }
        if self.order_callback is not None:
            self.order_callback(payload)
        if self.error_callback is not None:
            self.error_callback(
                {
                    "event_type": "order_reject",
                    "source": source,
                    "broker_order_id": broker_order_id,
                    "client_order_id": payload["client_order_id"],
                    "symbol": payload["symbol"],
                    "error_code": str(error_id),
                    "error_message": error_msg,
                    "timestamp_ns": payload["timestamp_ns"],
                    "order_ref": order_ref,
                }
            )

    def _make_broker_order_id(
        self,
        *,
        front_id: int,
        session_id: int,
        order_ref: str,
        order_sys_id: str,
    ) -> str:
        base = f"ctp-{front_id}-{session_id}-{order_ref}"
        if order_sys_id:
            return f"{base}-{order_sys_id}"
        return base

    def _parse_broker_order_id(self, broker_order_id: str) -> dict[str, Any]:
        text = str(broker_order_id).strip()
        parts = text.split("-")
        if len(parts) < 4 or parts[0] != "ctp":
            return {}
        return {
            "front_id": int(parts[1]) if parts[1].isdigit() else self.front_id,
            "session_id": int(parts[2]) if parts[2].isdigit() else self.session_id,
            "order_ref": parts[3],
            "order_sys_id": "-".join(parts[4:]) if len(parts) > 4 else "",
        }

    def _map_order_status(self, raw_status: Any) -> str:
        key = self._to_text(raw_status)
        status_map = {
            "0": "filled",
            "1": "partially_filled",
            "2": "partially_filled",
            "3": "submitted",
            "4": "submitted",
            "5": "cancelled",
            "a": "submitted",
            "b": "submitted",
            "c": "submitted",
        }
        if key in status_map:
            return status_map[key]
        return key.lower() if key else "submitted"

    def _map_direction(self, raw_direction: Any) -> str:
        key = self._to_text(raw_direction)
        if key in {"0", "buy", "b"}:
            return "Buy"
        if key in {"1", "sell", "s"}:
            return "Sell"
        return str(raw_direction)

    def _map_position_direction(self, raw_direction: Any) -> str:
        key = self._to_text(raw_direction).lower()
        long_key = self._to_text(getattr(tdapi, "THOST_FTDC_PD_Long", "2")).lower()
        short_key = self._to_text(getattr(tdapi, "THOST_FTDC_PD_Short", "3")).lower()
        if key in {long_key, "2", "long", "buy"}:
            return "Buy"
        if key in {short_key, "3", "short", "sell"}:
            return "Sell"
        return self._map_direction(raw_direction)

    def _map_position_date(self, raw_position_date: Any) -> str:
        key = self._to_text(raw_position_date).lower()
        today_key = self._to_text(getattr(tdapi, "THOST_FTDC_PSD_Today", "1")).lower()
        history_key = self._to_text(
            getattr(tdapi, "THOST_FTDC_PSD_History", "2")
        ).lower()
        if key in {today_key, "1", "today"}:
            return "today"
        if key in {history_key, "2", "history", "yesterday"}:
            return "yesterday"
        return ""

    def _trade_payload_from_ctp_trade(self, trade: Any) -> dict[str, Any]:
        order_ref = self._to_text(getattr(trade, "OrderRef", ""))
        order_sys_id = self._to_text(getattr(trade, "OrderSysID", "")).strip()
        front_id = int(getattr(trade, "FrontID", 0) or self.front_id)
        session_id = int(getattr(trade, "SessionID", 0) or self.session_id)
        return {
            "trade_id": self._to_text(getattr(trade, "TradeID", "")),
            "broker_order_id": self._make_broker_order_id(
                front_id=front_id,
                session_id=session_id,
                order_ref=order_ref,
                order_sys_id=order_sys_id,
            ),
            "client_order_id": self.order_ref_to_client_order_id.get(order_ref, ""),
            "symbol": self._to_text(getattr(trade, "InstrumentID", "")),
            "side": self._map_direction(getattr(trade, "Direction", "")),
            "quantity": float(getattr(trade, "Volume", 0.0) or 0.0),
            "price": float(getattr(trade, "Price", 0.0) or 0.0),
            "timestamp_ns": time.time_ns(),
            "order_ref": order_ref,
            "position_effect": self._map_ctp_offset_to_position_effect(
                getattr(trade, "OffsetFlag", "")
            ),
        }

    def _accumulate_position_query_row(self, position: Any) -> None:
        symbol = self._to_text(getattr(position, "InstrumentID", ""))
        direction = self._map_position_direction(getattr(position, "PosiDirection", ""))
        if not symbol:
            return

        key = (symbol, direction)
        snapshot = self._position_query_results.get(
            key,
            {
                "symbol": symbol,
                "direction": direction,
                "quantity": 0.0,
                "available_quantity": 0.0,
                "today_quantity": 0.0,
                "yesterday_quantity": 0.0,
                "available_today_quantity": 0.0,
                "available_yesterday_quantity": 0.0,
                "avg_price": 0.0,
                "timestamp_ns": time.time_ns(),
            },
        )

        position_qty = float(getattr(position, "Position", 0.0) or 0.0)
        today_qty = float(getattr(position, "TodayPosition", 0.0) or 0.0)
        yesterday_qty = float(getattr(position, "YdPosition", 0.0) or 0.0)
        if today_qty <= 0.0 and yesterday_qty <= 0.0:
            position_date = self._map_position_date(
                getattr(position, "PositionDate", "")
            )
            if position_date == "today":
                today_qty = position_qty
            elif position_date == "yesterday":
                yesterday_qty = position_qty
        if yesterday_qty <= 0.0:
            yesterday_qty = max(position_qty - today_qty, 0.0)
        if today_qty <= 0.0:
            today_qty = max(position_qty - yesterday_qty, 0.0)

        direction_frozen = float(
            getattr(
                position,
                "ShortFrozen" if direction == "Buy" else "LongFrozen",
                0.0,
            )
            or 0.0
        )
        today_frozen = float(getattr(position, "TodayFrozen", 0.0) or 0.0)
        yesterday_frozen = float(getattr(position, "YdFrozen", 0.0) or 0.0)
        if today_frozen <= 0.0 and yesterday_frozen <= 0.0 and direction_frozen > 0.0:
            allocated_yesterday_frozen = min(yesterday_qty, direction_frozen)
            allocated_today_frozen = min(
                today_qty,
                max(direction_frozen - allocated_yesterday_frozen, 0.0),
            )
            yesterday_frozen = allocated_yesterday_frozen
            today_frozen = allocated_today_frozen

        snapshot["quantity"] += position_qty
        snapshot["today_quantity"] += today_qty
        snapshot["yesterday_quantity"] += yesterday_qty
        snapshot["available_today_quantity"] += max(today_qty - today_frozen, 0.0)
        snapshot["available_yesterday_quantity"] += max(
            yesterday_qty - yesterday_frozen, 0.0
        )
        snapshot["available_quantity"] = (
            snapshot["available_today_quantity"]
            + snapshot["available_yesterday_quantity"]
        )

        price_candidates = (
            getattr(position, "PositionPrice", 0.0),
            getattr(position, "OpenPrice", 0.0),
            getattr(position, "SettlementPrice", 0.0),
        )
        for candidate in price_candidates:
            price_value = float(candidate or 0.0)
            if price_value > 0.0:
                snapshot["avg_price"] = price_value
                break

        snapshot["timestamp_ns"] = time.time_ns()
        self._position_query_results[key] = snapshot

    def _normalize_position_effect(self, position_effect: str | None) -> str:
        normalized = str(position_effect or "auto").strip().lower()
        if normalized not in {
            "auto",
            "open",
            "close",
            "close_today",
            "close_yesterday",
        }:
            raise ValueError(
                "position_effect must be one of: auto, open, close, "
                "close_today, close_yesterday"
            )
        return normalized

    def _map_position_effect_to_ctp_offset(self, position_effect: str) -> Any:
        mapping = {
            "auto": getattr(tdapi, "THOST_FTDC_OF_Open", "0"),
            "open": getattr(tdapi, "THOST_FTDC_OF_Open", "0"),
            "close": getattr(tdapi, "THOST_FTDC_OF_Close", "1"),
            "close_today": getattr(tdapi, "THOST_FTDC_OF_CloseToday", "3"),
            "close_yesterday": getattr(tdapi, "THOST_FTDC_OF_CloseYesterday", "4"),
        }
        return mapping[position_effect]

    def _map_ctp_offset_to_position_effect(self, raw_offset: Any) -> str:
        key = self._to_text(raw_offset)
        mapping = {
            self._to_text(getattr(tdapi, "THOST_FTDC_OF_Open", "0")): "open",
            self._to_text(getattr(tdapi, "THOST_FTDC_OF_Close", "1")): "close",
            self._to_text(
                getattr(tdapi, "THOST_FTDC_OF_CloseToday", "3")
            ): "close_today",
            self._to_text(
                getattr(tdapi, "THOST_FTDC_OF_CloseYesterday", "4")
            ): "close_yesterday",
            "open": "open",
            "close": "close",
            "closetoday": "close_today",
            "close_today": "close_today",
            "closeyesterday": "close_yesterday",
            "close_yesterday": "close_yesterday",
        }
        return mapping.get(key.lower(), "auto")

    def _to_text(self, value: Any) -> str:
        if isinstance(value, bytes):
            try:
                return value.decode("gbk", errors="ignore").strip()
            except Exception:
                return value.decode("utf-8", errors="ignore").strip()
        return str(value).strip()


class CTPMarketGateway(mdapi.CThostFtdcMdSpi):  # type: ignore
    """
    CTP Market Data Gateway.

    Subscribes to market data and pushes Bars and/or Ticks to AKQuant DataFeed,
    depending on ``emit_ticks``/``emit_bars`` (see :meth:`__init__`).
    """

    def __init__(
        self,
        feed: DataFeed,
        front_url: str,
        symbols: List[str],
        use_aggregator: bool = True,
        emit_ticks: Optional[bool] = None,
        emit_bars: Optional[bool] = None,
    ):
        """
        Initialize the Market Gateway.

        :param feed: AKQuant DataFeed instance
        :param front_url: CTP Front URL (e.g., tcp://180.168.146.187:10131)
        :param symbols: List of symbols to subscribe (e.g., ["au2310", "rb2310"])
        :param use_aggregator: Legacy either-or switch, kept as a compat alias:
            True (default) is equivalent to bar-only, False is equivalent to
            tick-only. Superseded by ``emit_ticks``/``emit_bars`` when those are
            given explicitly — the two can then both be True (dual stream).
        :param emit_ticks: Whether to push raw Ticks into the feed (drives
            ``on_tick``). ``None`` falls back to the ``use_aggregator`` alias
            *per-parameter* (see note below), so passing only ``emit_bars``
            does not silently flip this one to False.
        :param emit_bars: Whether to push aggregated Bars into the feed (drives
            ``on_bar``). ``None`` falls back to the ``use_aggregator`` alias the
            same way as ``emit_ticks``.

            The per-parameter fallback (rather than "fall back only when both
            are None") matters: if a caller passes ``emit_ticks=True`` alone,
            wanting to *add* tick delivery on top of the existing bar stream,
            resolving the untouched ``emit_bars`` from ``bool(None)`` would
            silently turn off ``on_bar`` — the opposite of what they asked for.
        :raises ImportError: openctp-ctp is not installed.
        :raises ValueError: ``emit_ticks`` and ``emit_bars`` both resolve to
            False — that configuration would push no market data at all.
        """
        if not HAS_OPENCTP:
            raise ImportError(
                "openctp-ctp is not installed. Please run `pip install openctp-ctp`."
            )

        mdapi.CThostFtdcMdSpi.__init__(self)
        self.feed = feed
        self.front_url = front_url
        self.symbols = symbols
        self.use_aggregator = use_aggregator

        self.api: Any = None
        self.req_id = 0
        self.connected = False

        # use_aggregator 是旧的二选一开关, 保留为兼容别名: True(默认) -> 只出
        # bar, False -> 只出 tick。新参数 emit_ticks / emit_bars 显式且可同时
        # 为真(双流), 优先级更高。
        # 逐参数回退(而非"两者都 None 才回退"): 只显式表态一个参数时, 另一个
        # 仍按 use_aggregator 别名推导, 不会被 bool(None) 静默压成 False——
        # 否则单独传 emit_ticks=True 会悄悄关掉 on_bar, 而用户的意图是
        # "加上 tick"而非"关掉 bar"。
        if emit_ticks is None:
            emit_ticks = not use_aggregator
        if emit_bars is None:
            emit_bars = bool(use_aggregator)
        emit_ticks = bool(emit_ticks)
        emit_bars = bool(emit_bars)
        if not emit_ticks and not emit_bars:
            raise ValueError(
                "emit_ticks 与 emit_bars 不能同时为 False: 那样网关不会推任何行情"
            )
        self.emit_ticks = emit_ticks
        self.emit_bars = emit_bars
        # 双流下 bar 必须打区间结束戳, 否则与 tick 混推会时间戳倒退(实盘无
        # sort 兜底, 见 OnRtnDepthMarketData 里的顺序注释)。
        self.stamp_bar_at_interval_end = emit_ticks and emit_bars

        self.aggregator: Optional[BarAggregator] = None
        if self.emit_bars:
            self.aggregator = BarAggregator(
                feed, stamp_bar_at_interval_end=self.stamp_bar_at_interval_end
            )

    def _log_extra(self, *, symbol: str | None = None) -> dict[str, Any]:
        return build_log_extra(phase="gateway", symbol=symbol)

    def start(self) -> None:
        """Start the CTP Market API in a blocking way (should be run in a thread)."""
        logger.info(
            "Connecting native CTP market gateway to %s",
            self.front_url,
            extra=self._log_extra(),
        )
        try:
            self.api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi()
            self.api.RegisterFront(self.front_url)
            self.api.RegisterSpi(self)
            self.api.Init()
            logger.info(
                "Native CTP market API initialized; joining event loop",
                extra=self._log_extra(),
            )
            self.api.Join()
        except Exception:
            logger.exception(
                "Failed to start native CTP market gateway",
                extra=self._log_extra(),
            )

    def OnFrontConnected(self) -> None:
        """Handle front connection event."""
        logger.info(
            "CTP market front connected: %s",
            self.front_url,
            extra=self._log_extra(),
        )
        self.connected = True

        req = mdapi.CThostFtdcReqUserLoginField()
        self.req_id += 1
        logger.info(
            "Requesting CTP market user login (req_id=%s)",
            self.req_id,
            extra=self._log_extra(),
        )
        if self.api:
            self.api.ReqUserLogin(req, self.req_id)

    def OnFrontDisconnected(self, nReason: int) -> None:
        """Handle front disconnection event."""
        logger.warning(
            "CTP market front disconnected: reason=%s",
            nReason,
            extra=self._log_extra(),
        )
        self.connected = False

    def OnRspUserLogin(
        self, pRspUserLogin: Any, pRspInfo: Any, nRequestID: int, bIsLast: bool
    ) -> None:
        """Handle login response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.error(
                "CTP market login failed: error_id=%s msg=%s",
                pRspInfo.ErrorID,
                pRspInfo.ErrorMsg,
                extra=self._log_extra(),
            )
            return

        logger.info(
            "CTP market login succeeded: trading_day=%s broker_id=%s user_id=%s",
            pRspUserLogin.TradingDay,
            pRspUserLogin.BrokerID,
            pRspUserLogin.UserID,
            extra=self._log_extra(),
        )

        logger.info(
            "Subscribing CTP market data for symbols=%s",
            self.symbols,
            extra=self._log_extra(),
        )
        self.req_id += 1
        if self.api:
            ret = self.api.SubscribeMarketData(
                [s.encode("utf-8") for s in self.symbols], len(self.symbols)
            )
            if ret == 0:
                logger.info(
                    "CTP market subscribe request sent successfully",
                    extra=self._log_extra(),
                )
            else:
                logger.error(
                    "CTP market subscribe request failed: code=%s",
                    ret,
                    extra=self._log_extra(),
                )

    def OnRspSubMarketData(
        self,
        pSpecificInstrument: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle subscribe response."""
        symbol = str(getattr(pSpecificInstrument, "InstrumentID", "")).strip() or None
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.error(
                "CTP market subscribe failed for [%s]: %s",
                getattr(pSpecificInstrument, "InstrumentID", ""),
                pRspInfo.ErrorMsg,
                extra=self._log_extra(symbol=symbol),
            )
            return
        logger.info(
            "CTP market subscribe succeeded for [%s]",
            getattr(pSpecificInstrument, "InstrumentID", ""),
            extra=self._log_extra(symbol=symbol),
        )

    def OnRtnDepthMarketData(self, pDepthMarketData: Any) -> None:
        """Process market data tick."""
        symbol = pDepthMarketData.InstrumentID
        price = pDepthMarketData.LastPrice
        volume = pDepthMarketData.Volume

        if price > 1e7 or price <= 0:
            return

        now_ns = time.time_ns()

        # 顺序: 先喂聚合器(可能吐出上一区间已收盘的 bar), 再推当前 tick,
        # 保证时间戳单调。实盘没有回测那层 feed.sort() 兜底
        # (RealtimeDataClient::sort 是空实现, "Live data cannot be sorted"),
        # 推入顺序即引擎所见顺序——反了会让 bar 排在更晚的 tick 之后造成时间戳
        # 倒退, 且这个倒退是静默的, 不会报错, 只会破坏批次边界。
        if self.aggregator is not None:
            self.aggregator.on_tick(symbol, price, float(volume), now_ns)

        # 引入真 add_tick 之前, use_aggregator=False 下曾把每笔 tick 包装成
        # 退化 bar(open=high=low=close=price)推入 feed, 作为 on_bar 的替代品
        # ——那正是 fill_missing_bars 曾用成交价合成假 bar 污染 bar 序列的
        # 同一类问题(见本计划 Task 5)。现在逐笔以真 Tick 形式推送, 该替代
        # 路径已删除; emit_ticks/emit_bars 都不出的组合在构造期就被
        # ValueError 拒绝, 不存在"两者都不出"需要兜底的状态。
        if self.emit_ticks:
            self.feed.add_tick(
                Tick(timestamp=now_ns, symbol=symbol, price=price, volume=float(volume))
            )
