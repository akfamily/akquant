"""内置 broker builder 及注册入口（把内置 broker 变成 registry 插件）."""

from __future__ import annotations

from typing import Any, Sequence

from ...akquant import DataFeed
from ..protocols import GatewayBundle, MarketGateway, TraderGateway
from ..registry import register_broker
from .ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from .miniqmt.stub import MiniQMTMarketGateway, MiniQMTTraderGateway
from .ptrade.stub import PTradeMarketGateway, PTradeTraderGateway


def _resolve_trader_capabilities(trader_gateway: TraderGateway | None) -> Any:
    if trader_gateway is None:
        return None
    get_capabilities = getattr(trader_gateway, "get_capabilities", None)
    return get_capabilities() if callable(get_capabilities) else None


def _build_ctp(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    md_front = kwargs.get("md_front", "")
    if not md_front:
        raise ValueError("md_front is required when broker='ctp'")
    market_gateway: MarketGateway = CTPMarketAdapter(
        feed=feed,
        front_url=md_front,
        symbols=list(symbols),
        use_aggregator=use_aggregator,
    )
    trader_gateway: TraderGateway | None = None
    td_front = kwargs.get("td_front")
    user_id = kwargs.get("user_id")
    if td_front and user_id:
        trader_gateway = CTPTraderAdapter(
            front_url=td_front,
            broker_id=kwargs.get("broker_id", "9999"),
            user_id=user_id,
            password=kwargs.get("password", ""),
            auth_code=kwargs.get("auth_code", "0000000000000000"),
            app_id=kwargs.get("app_id", "simnow_client_test"),
            execution_semantics_mode=kwargs.get("execution_semantics_mode", "strict"),
        )
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "ctp"},
    )


def _build_miniqmt(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    market_gateway = MiniQMTMarketGateway(feed=feed, symbols=list(symbols), **kwargs)
    trader_gateway: TraderGateway | None = MiniQMTTraderGateway(**kwargs)
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "miniqmt"},
    )


def _build_ptrade(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    market_gateway = PTradeMarketGateway(feed=feed, symbols=list(symbols), **kwargs)
    trader_gateway: TraderGateway | None = PTradeTraderGateway(**kwargs)
    return GatewayBundle(
        market_gateway=market_gateway,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "ptrade"},
    )


def _build_qmf(
    feed: DataFeed, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
) -> GatewayBundle:
    _ = (feed, symbols, use_aggregator)
    try:
        from .qmf.adapter import QMFTraderGateway
        from .qmf.client import QMFClientConfig, QMFHttpClient
    except ImportError as exc:
        raise ValueError(
            "broker='qmf' 需要额外依赖，请安装: pip install 'akquant[qmf]'"
        ) from exc
    ws_url = kwargs.get("ws_url")
    if not ws_url:
        raise ValueError("ws_url is required when broker='qmf'")
    required = (
        "base_url",
        "qmf_user_id",
        "account_content",
        "password",
        "input_content",
        "content_type",
        "password_key",
    )
    missing = [k for k in required if not kwargs.get(k)]
    if missing:
        raise ValueError(f"broker='qmf' 缺少必填项: {', '.join(missing)}")
    config = QMFClientConfig(
        base_url=kwargs["base_url"],
        qmf_user_id=kwargs["qmf_user_id"],
        account_content=kwargs["account_content"],
        password=kwargs["password"],
        input_content=kwargs["input_content"],
        content_type=kwargs["content_type"],
        password_key=kwargs["password_key"],
        password_type=kwargs.get("password_type", "2"),
        asset_prop=kwargs.get("asset_prop", "0"),
        timeout=float(kwargs.get("timeout", 10.0)),
    )
    enable_options = bool(kwargs.get("enable_options", False))
    option_client = None
    if enable_options:
        option_config = QMFClientConfig(
            base_url=kwargs["base_url"],
            qmf_user_id=kwargs.get("option_qmf_user_id", kwargs["qmf_user_id"]),
            account_content=kwargs.get(
                "option_account_content", kwargs["account_content"]
            ),
            password=kwargs.get("option_password", kwargs["password"]),
            input_content=kwargs["input_content"],
            content_type=kwargs["content_type"],
            password_key=kwargs["password_key"],
            password_type=kwargs.get("password_type", "2"),
            asset_prop="B",
            timeout=float(kwargs.get("timeout", 10.0)),
        )
        option_client = QMFHttpClient(option_config)
    trader_gateway: TraderGateway | None = QMFTraderGateway(
        client=QMFHttpClient(config), ws_url=ws_url, option_client=option_client
    )
    return GatewayBundle(
        market_gateway=None,
        trader_gateway=trader_gateway,
        trader_capabilities=_resolve_trader_capabilities(trader_gateway),
        metadata={"broker": "qmf"},
    )


def register_builtin_brokers() -> None:
    """Register all built-in brokers into the shared registry (idempotent)."""
    register_broker("ctp", _build_ctp)
    register_broker("miniqmt", _build_miniqmt)
    register_broker("ptrade", _build_ptrade)
    register_broker("qmf", _build_qmf)
