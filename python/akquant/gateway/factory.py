from typing import Any, Sequence

from ..akquant import DataFeed
from .brokers.ctp.adapter import CTPMarketAdapter, CTPTraderAdapter
from .brokers.miniqmt.stub import MiniQMTMarketGateway, MiniQMTTraderGateway
from .brokers.ptrade.stub import PTradeMarketGateway, PTradeTraderGateway
from .protocols import GatewayBundle, MarketGateway, TraderGateway
from .registry import create_registered_gateway_bundle, list_registered_brokers


def _resolve_trader_capabilities(trader_gateway: TraderGateway | None) -> Any:
    if trader_gateway is None:
        return None
    get_capabilities = getattr(trader_gateway, "get_capabilities", None)
    if callable(get_capabilities):
        return get_capabilities()
    return None


def create_gateway_bundle(
    broker: str,
    feed: DataFeed,
    symbols: Sequence[str],
    use_aggregator: bool = True,
    **kwargs: Any,
) -> GatewayBundle:
    """Create market/trader gateway bundle by broker key."""
    broker_key = broker.strip().lower()
    registered_bundle = create_registered_gateway_bundle(
        name=broker_key,
        feed=feed,
        symbols=symbols,
        use_aggregator=use_aggregator,
        **kwargs,
    )
    if registered_bundle is not None:
        return registered_bundle

    if broker_key == "ctp":
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
                execution_semantics_mode=kwargs.get(
                    "execution_semantics_mode", "strict"
                ),
            )
        return GatewayBundle(
            market_gateway=market_gateway,
            trader_gateway=trader_gateway,
            trader_capabilities=_resolve_trader_capabilities(trader_gateway),
            metadata={"broker": "ctp"},
        )

    if broker_key == "miniqmt":
        market_gateway = MiniQMTMarketGateway(
            feed=feed,
            symbols=list(symbols),
            **kwargs,
        )
        miniqmt_trader_gateway: TraderGateway | None = MiniQMTTraderGateway(**kwargs)
        return GatewayBundle(
            market_gateway=market_gateway,
            trader_gateway=miniqmt_trader_gateway,
            trader_capabilities=_resolve_trader_capabilities(miniqmt_trader_gateway),
            metadata={"broker": "miniqmt"},
        )

    if broker_key == "ptrade":
        market_gateway = PTradeMarketGateway(
            feed=feed,
            symbols=list(symbols),
            **kwargs,
        )
        ptrade_trader_gateway: TraderGateway | None = PTradeTraderGateway(**kwargs)
        return GatewayBundle(
            market_gateway=market_gateway,
            trader_gateway=ptrade_trader_gateway,
            trader_capabilities=_resolve_trader_capabilities(ptrade_trader_gateway),
            metadata={"broker": "ptrade"},
        )

    if broker_key == "qmf":
        try:
            from .brokers.qmf.adapter import QMFMarketGateway, QMFTraderGateway
            from .brokers.qmf.client import QMFClientConfig, QMFHttpClient
        except ImportError as exc:  # 缺可选依赖
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
        qmf_trader_gateway: TraderGateway | None = QMFTraderGateway(
            client=QMFHttpClient(config), ws_url=ws_url
        )
        return GatewayBundle(
            market_gateway=QMFMarketGateway(),
            trader_gateway=qmf_trader_gateway,
            trader_capabilities=_resolve_trader_capabilities(qmf_trader_gateway),
            metadata={"broker": "qmf"},
        )

    builtins = ["ctp", "miniqmt", "ptrade", "qmf"]
    registered = list_registered_brokers()
    all_brokers = builtins + [name for name in registered if name not in builtins]
    supported = ", ".join(all_brokers)
    raise ValueError(f"broker must be one of: {supported}")
