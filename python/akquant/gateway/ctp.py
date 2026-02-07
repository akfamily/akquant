# -*- coding: utf-8 -*-
"""
CTP Gateway Implementation for AKQuant.

This module provides CTP (China Futures) connectivity using openctp-ctp library.
"""

import time
from typing import Any, Dict, List, Optional

from akquant import Bar, BarAggregator, DataFeed

# Optional dependency
try:
    from openctp_ctp import thostmduserapi as mdapi  # type: ignore
    from openctp_ctp import thosttraderapi as tdapi  # type: ignore

    HAS_OPENCTP = True
except ImportError:
    HAS_OPENCTP = False
    # Define dummy base classes to avoid NameError during class definition
    mdapi = object  # type: ignore
    tdapi = object  # type: ignore

    # Mocking Spi classes for type hinting/inheritance purposes if library is missing
    # In reality, the classes below won't work without the library
    class MockSpi:
        """Mock SPI class for when openctp-ctp is not installed."""

        pass

    mdapi.CThostFtdcMdSpi = MockSpi  # type: ignore
    tdapi.CThostFtdcTraderSpi = MockSpi  # type: ignore


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

    def start(self) -> None:
        """Start the CTP Trader API in a blocking way (should be run in a thread)."""
        print(f"[CTP-Trade] Connecting to {self.front_url}...")
        try:
            self.api = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi()
            self.api.RegisterFront(self.front_url)
            self.api.RegisterSpi(self)
            self.api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
            self.api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
            self.api.Init()

            # Keep thread alive
            print("[CTP-Trade] API Initialized, joining thread...")
            self.api.Join()
        except Exception as e:
            print(f"[CTP-Trade] ERROR: Failed to start CTP Trader API: {e}")

    def OnFrontConnected(self) -> None:
        """Handle front connection event."""
        print(
            f"[CTP-Trade] OnFrontConnected: Successfully connected to {self.front_url}"
        )
        self.connected = True

        # Authenticate
        self.req_id += 1
        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID = self.broker_id
        req.UserID = self.user_id
        req.AppID = self.app_id
        req.AuthCode = self.auth_code
        print(f"[CTP-Trade] Requesting Authentication (ReqID={self.req_id})...")
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
            print(
                f"[CTP-Trade] Authentication failed. ErrorID={pRspInfo.ErrorID}, "
                f"Msg={pRspInfo.ErrorMsg}"
            )
            # Even if auth fails, some sim envs might allow login, but usually not.
        else:
            print("[CTP-Trade] Authentication succeed.")

        self.authenticated = True

        # Login
        self.req_id += 1
        req = tdapi.CThostFtdcReqUserLoginField()
        req.BrokerID = self.broker_id
        req.UserID = self.user_id
        req.Password = self.password
        print(f"[CTP-Trade] Requesting User Login (ReqID={self.req_id})...")
        if self.api:
            self.api.ReqUserLogin(req, self.req_id)

    def OnRspUserLogin(
        self, pRspUserLogin: Any, pRspInfo: Any, nRequestID: int, bIsLast: bool
    ) -> None:
        """Handle login response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(
                f"[CTP-Trade] Login failed. ErrorID={pRspInfo.ErrorID}, "
                f"Msg={pRspInfo.ErrorMsg}"
            )
            return

        print(
            f"[CTP-Trade] Login succeed. TradingDay: {pRspUserLogin.TradingDay}, "
            f"BrokerID: {pRspUserLogin.BrokerID}, UserID: {pRspUserLogin.UserID}"
        )
        self.login_status = True

        # Settlement Confirm
        self.req_id += 1
        req = tdapi.CThostFtdcSettlementInfoConfirmField()
        req.BrokerID = self.broker_id
        req.InvestorID = self.user_id
        print("[CTP-Trade] Confirming Settlement Info...")
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
            print(f"[CTP-Trade] Settlement Confirm failed: {pRspInfo.ErrorMsg}")
        else:
            print("[CTP-Trade] Settlement Info Confirmed.")

    def OnFrontDisconnected(self, nReason: int) -> None:
        """Handle front disconnection event."""
        print(f"[CTP-Trade] OnFrontDisconnected. [nReason={nReason}]")
        self.connected = False
        self.login_status = False


class CTPMarketGateway(mdapi.CThostFtdcMdSpi):  # type: ignore
    """
    CTP Market Data Gateway.

    Subscribes to market data and pushes Ticks (as Bars) to AKQuant DataFeed.
    """

    def __init__(
        self,
        feed: DataFeed,
        front_url: str,
        symbols: List[str],
        use_aggregator: bool = True,
    ):
        """
        Initialize the Market Gateway.

        :param feed: AKQuant DataFeed instance
        :param front_url: CTP Front URL (e.g., tcp://180.168.146.187:10131)
        :param symbols: List of symbols to subscribe (e.g., ["au2310", "rb2310"])
        :param use_aggregator: If True, uses BarAggregator to generate 1-min bars.
                               If False, pushes each tick as a Bar immediately.
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

        # Helper for Tick-as-Bar mode
        self.last_volume: Dict[str, float] = {}

        # Helper for Aggregator mode
        self.aggregator: Optional[BarAggregator]
        if self.use_aggregator:
            self.aggregator = BarAggregator(feed)
        else:
            self.aggregator = None

    def start(self) -> None:
        """Start the CTP Market API in a blocking way (should be run in a thread)."""
        print(f"[CTP] Connecting to {self.front_url}...")
        try:
            self.api = mdapi.CThostFtdcMdApi.CreateFtdcMdApi()
            self.api.RegisterFront(self.front_url)
            self.api.RegisterSpi(self)
            self.api.Init()

            # Keep the CTP thread alive
            print("[CTP] API Initialized, joining thread...")
            self.api.Join()
        except Exception as e:
            print(f"[CTP] ERROR: Failed to start CTP API: {e}")

    def OnFrontConnected(self) -> None:
        """Handle front connection event."""
        print(f"[CTP] OnFrontConnected: Successfully connected to {self.front_url}")
        self.connected = True

        # Login (SimNow market data usually doesn't need specific user/pass)
        req = mdapi.CThostFtdcReqUserLoginField()
        self.req_id += 1
        print(f"[CTP] Requesting User Login (ReqID={self.req_id})...")
        if self.api:
            self.api.ReqUserLogin(req, self.req_id)

    def OnFrontDisconnected(self, nReason: int) -> None:
        """Handle front disconnection event."""
        print(f"[CTP] OnFrontDisconnected. [nReason={nReason}]")
        self.connected = False

    def OnRspUserLogin(
        self, pRspUserLogin: Any, pRspInfo: Any, nRequestID: int, bIsLast: bool
    ) -> None:
        """Handle login response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(
                f"[CTP] Login failed. ErrorID={pRspInfo.ErrorID}, "
                f"Msg={pRspInfo.ErrorMsg}"
            )
            return

        print(
            f"[CTP] Login succeed. TradingDay: {pRspUserLogin.TradingDay}, "
            f"BrokerID: {pRspUserLogin.BrokerID}, UserID: {pRspUserLogin.UserID}"
        )

        # Subscribe
        print(f"[CTP] Subscribing to {self.symbols}...")
        self.req_id += 1
        # CTP API expects list of bytes
        if self.api:
            ret = self.api.SubscribeMarketData(
                [s.encode("utf-8") for s in self.symbols], len(self.symbols)
            )
            if ret == 0:
                print("[CTP] Subscribe request sent successfully.")
            else:
                print(f"[CTP] Subscribe request failed with code {ret}")

    def OnRspSubMarketData(
        self,
        pSpecificInstrument: Any,
        pRspInfo: Any,
        nRequestID: int,
        bIsLast: bool,
    ) -> None:
        """Handle subscribe response."""
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(
                f"[CTP] Subscribe failed for [{pSpecificInstrument.InstrumentID}]: "
                f"{pRspInfo.ErrorMsg}"
            )
            return
        print(f"[CTP] Subscribe succeed for [{pSpecificInstrument.InstrumentID}]")

    def OnRtnDepthMarketData(self, pDepthMarketData: Any) -> None:
        """Process market data tick."""
        # Received Tick Data
        symbol = pDepthMarketData.InstrumentID
        price = pDepthMarketData.LastPrice
        volume = pDepthMarketData.Volume
        # update_time = pDepthMarketData.UpdateTime
        # millis = pDepthMarketData.UpdateMillisec

        # Filter invalid data
        if price > 1e7 or price <= 0:
            return

        # Note: In production logging every tick might be too verbose
        # print(
        #     f"[CTP] TICK -> {symbol} | Time: {update_time}.{millis:03d} | "
        #     f"Price: {price} | Vol: {volume}"
        # )

        # Use system time for timestamp as CTP time parsing can be complex
        now_ns = time.time_ns()

        if self.use_aggregator and self.aggregator:
            # Use Rust Aggregator
            self.aggregator.on_tick(symbol, price, float(volume), now_ns)
        else:
            # Manual Tick-as-Bar Logic
            # Calculate Delta Volume (CTP Volume is cumulative)
            last_vol = self.last_volume.get(symbol, volume)
            delta_vol = volume - last_vol
            self.last_volume[symbol] = volume

            # Avoid negative volume if reset happens (e.g. next trading day)
            if delta_vol < 0:
                delta_vol = volume

            # Construct akquant.Bar
            bar = Bar(
                timestamp=now_ns,
                symbol=symbol,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=float(delta_vol),
                extra={},
            )

            # Push to AkQuant Engine (Thread-safe)
            self.feed.add_bar(bar)  # type: ignore
