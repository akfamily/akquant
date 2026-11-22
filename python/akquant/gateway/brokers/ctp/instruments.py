"""从 CTP 查询期权合约并转为 ``Instrument``, 按交易时段缓存, 供实盘登记与运行中刷新.

交易所在夜盘、日盘两个时段都会新挂期权合约, 因此每个时段开盘前(默认 08:30、20:30)
各全量查询一次。查询走独立的短连接交易会话(登录、查询、断开), 不占用下单网关的查询
流控; 不确认结算单, 不下单。

- 商品期权: 标的期货一并登记, 期货保证金率取 ``ReqQryInstrumentMarginRate`` 的投资者
  按金额卖方保证金率, 期权保证金模型 ``CHINA_COMMODITY``。
- 中金所股指期权(``ProductClass='6'``): CTP 的标的字段是期权系列(如 ``IO2612``), 行情
  不推送指数, 标的指数代码由调用方按品种指定; 保证金调整系数由
  ``ReqQryOptionInstrTradeCost`` 按两个标的价格反推并相互核对, 期权保证金模型
  ``CHINA_INDEX``。
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ....akquant import AssetType, Instrument, OptionMarginModel, OptionType
from ....log import get_logger

try:
    from openctp_ctp import thosttraderapi as tdapi  # type: ignore

    HAS_OPENCTP = True
except ImportError:
    HAS_OPENCTP = False

    class _MockTdApi:
        CThostFtdcTraderSpi = object

    tdapi = _MockTdApi  # type: ignore

logger = get_logger("gateway.ctp_instruments")

FUTURES_CLASS = "1"
COMMODITY_OPTION_CLASS = "2"
INDEX_OPTION_CLASS = "6"
CALL_TYPE = "1"
PUT_TYPE = "2"
CONNECTION_KEYS = (
    "td_front",
    "broker_id",
    "user_id",
    "password",
    "app_id",
    "auth_code",
)
DEFAULT_REFRESH_TIMES = ("08:30", "20:30")
DEFAULT_TIMEZONE = "Asia/Shanghai"
# 反推中金所调整系数用的两个标的价格(相对行权价): 认购实值(虚值额 0)与认购虚值
# (保证金取下限 0.5 × 系数 × 标的), 两个分支各得一个系数, 相互核对。
INDEX_PROBE_ITM = 1.2
INDEX_PROBE_OTM = 0.8
COEFFICIENT_TOLERANCE = 1e-6
RATIO_DIGITS = 8
QUERY_INTERVAL_SEC = 1.05  # CTP 查询流控 1 次/秒
QUERY_TIMEOUT_SEC = 120.0
LOGIN_TIMEOUT_SEC = 60.0
CACHE_VERSION = 1


def _row(field: Any) -> Dict[str, Any]:
    return {key: getattr(field, key) for key in dir(field) if key[:1].isupper()}


class _QuerySession(tdapi.CThostFtdcTraderSpi):  # type: ignore
    """只做查询的同步交易会话: 认证、登录后按 1 次/秒限频发查询并等全部回报."""

    def __init__(self, connection: Mapping[str, str], flow_dir: Path) -> None:
        if not HAS_OPENCTP:
            raise ImportError(
                "查询 CTP 合约需要 openctp-ctp, 请先 `pip install openctp-ctp`"
            )
        tdapi.CThostFtdcTraderSpi.__init__(self)
        self.conn = dict(connection)
        self.req_id = 0
        self.pending: Dict[int, Dict[str, Any]] = {}
        self.ready = threading.Event()
        self.error = ""
        self.trading_day = ""
        self.last_query = 0.0
        flow_dir.mkdir(parents=True, exist_ok=True)
        self.api = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi(str(flow_dir) + "/")
        self.api.RegisterSpi(self)
        self.api.RegisterFront(self.conn["td_front"])
        self.api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
        self.api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
        self.api.Init()
        if not self.ready.wait(LOGIN_TIMEOUT_SEC):
            self.close()
            raise RuntimeError(
                f"CTP 交易前置 {self.conn['td_front']} "
                f"{LOGIN_TIMEOUT_SEC:g} 秒内未完成登录"
            )
        if self.error:
            self.close()
            raise RuntimeError(f"CTP 查询会话登录失败: {self.error}")

    def _next(self) -> int:
        self.req_id += 1
        return self.req_id

    def _fail(self, where: str, info: Any) -> bool:
        if info is not None and info.ErrorID != 0:
            self.error = f"{where}: {info.ErrorID} {info.ErrorMsg}"
            self.ready.set()
            return True
        return False

    def OnFrontConnected(self) -> None:
        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID, req.UserID = self.conn["broker_id"], self.conn["user_id"]
        req.AppID, req.AuthCode = self.conn["app_id"], self.conn["auth_code"]
        self.api.ReqAuthenticate(req, self._next())

    def OnFrontDisconnected(self, nReason: int) -> None:
        if not self.ready.is_set():
            self.error = f"前置断开: {nReason}"
            self.ready.set()

    def OnRspAuthenticate(self, field: Any, info: Any, req_id: int, last: bool) -> None:
        if self._fail("认证", info):
            return
        req = tdapi.CThostFtdcReqUserLoginField()
        req.BrokerID, req.UserID = self.conn["broker_id"], self.conn["user_id"]
        req.Password = self.conn["password"]
        self.api.ReqUserLogin(req, self._next())

    def OnRspUserLogin(self, field: Any, info: Any, req_id: int, last: bool) -> None:
        if self._fail("登录", info):
            return
        self.trading_day = str(field.TradingDay)
        self.ready.set()

    def query(self, method: str, field: Any) -> List[Dict[str, Any]]:
        """发出查询并等待全部回报; 流控(返回 -2/-3)时间隔后重发."""
        for _ in range(10):
            wait = QUERY_INTERVAL_SEC - (time.monotonic() - self.last_query)
            if wait > 0:
                time.sleep(wait)
            self.last_query = time.monotonic()
            rid = self._next()
            slot: Dict[str, Any] = {"rows": [], "done": threading.Event(), "error": ""}
            self.pending[rid] = slot
            ret = getattr(self.api, method)(field, rid)
            if ret == 0:
                break
            del self.pending[rid]
        else:
            raise RuntimeError(f"{method} 连续 10 次被流控拒绝(返回 {ret})")
        if not slot["done"].wait(QUERY_TIMEOUT_SEC):
            raise RuntimeError(f"{method} {QUERY_TIMEOUT_SEC:g} 秒内未收齐回报")
        if slot["error"]:
            raise RuntimeError(f"{method} 失败: {slot['error']}")
        return list(slot["rows"])

    def _on_rsp(self, field: Any, info: Any, req_id: int, last: bool) -> None:
        slot = self.pending.get(req_id)
        if slot is None:
            return
        if info is not None and info.ErrorID != 0:
            slot["error"] = f"{info.ErrorID} {info.ErrorMsg}"
        elif field is not None:
            slot["rows"].append(_row(field))
        if last:
            slot["done"].set()

    OnRspQryInstrument = _on_rsp
    OnRspQryInstrumentMarginRate = _on_rsp
    OnRspQryOptionInstrTradeCost = _on_rsp

    def OnRspError(self, info: Any, req_id: int, last: bool) -> None:
        self._on_rsp(None, info, req_id, True)

    def close(self) -> None:
        self.api.Release()


def _matches(
    entry: str, row: Dict[str, Any], futures: Dict[str, Dict[str, Any]]
) -> bool:
    """期权是否属于 ``entry`` 指定的标的: 标的合约代码或标的品种代码(不区分大小写)."""
    key = entry.strip().lower()
    underlying = str(row["UnderlyingInstrID"])
    if underlying.lower() == key:
        return True
    if row["ProductClass"] == INDEX_OPTION_CLASS:
        return str(row["ProductID"]).lower() == key
    underlying_row = futures.get(underlying)
    return (
        underlying_row is not None and str(underlying_row["ProductID"]).lower() == key
    )


def select_option_rows(
    rows: Sequence[Dict[str, Any]], underlyings: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """从全量合约里选出跟踪标的的期权与标的期货.

    :param rows: ``ReqQryInstrument`` 的全部回报
    :param underlyings: 标的合约代码(如 ``m2601``、``IO2612``)或标的品种代码
        (如 ``m``、``IO``)
    :return: (期权行, 标的期货行)
    """
    futures = {
        str(r["InstrumentID"]): r for r in rows if r["ProductClass"] == FUTURES_CLASS
    }
    options = [
        r
        for r in rows
        if r["ProductClass"] in (COMMODITY_OPTION_CLASS, INDEX_OPTION_CLASS)
        and any(_matches(entry, r, futures) for entry in underlyings)
    ]
    unmatched = [
        entry
        for entry in underlyings
        if not any(_matches(entry, r, futures) for r in options)
    ]
    if unmatched:
        raise ValueError(f"CTP 合约表里没有以下标的的期权: {', '.join(unmatched)}")
    wanted = sorted(
        {
            str(r["UnderlyingInstrID"])
            for r in options
            if r["ProductClass"] == COMMODITY_OPTION_CLASS
        }
    )
    missing = [symbol for symbol in wanted if symbol not in futures]
    if missing:
        raise ValueError(f"商品期权的标的期货不在 CTP 合约表里: {', '.join(missing)}")
    return options, [futures[symbol] for symbol in wanted]


def index_margin_coefficient(product: str, probes: Sequence[Dict[str, Any]]) -> float:
    """由两次交易成本查询反推中金所期权保证金调整系数, 两个分支的结果须一致.

    每次查询的每手保证金(不含权利金) = 乘数 × max(标的 × 系数 − 虚值额,
    0.5 × 系数 × 标的)(认购)。实值探测虚值额为 0, 系数 = 保证金 / (乘数 × 标的);
    虚值探测取下限, 系数 = 保证金 / (0.5 × 乘数 × 标的)。

    :param product: 期权品种(如 ``IO``)
    :param probes: ``{"label": "itm"|"otm", "underlying_price", "multiplier",
        "fixed_margin"}`` 两条
    """
    found: Dict[str, float] = {}
    for probe in probes:
        base = float(probe["multiplier"]) * float(probe["underlying_price"])
        if probe["label"] == "otm":
            base *= 0.5
        found[str(probe["label"])] = float(probe["fixed_margin"]) / base
    if set(found) != {"itm", "otm"}:
        raise ValueError(
            f"{product} 的交易成本探测须有实值、虚值各一条, 实际: {sorted(found)}"
        )
    if found["itm"] <= 0:
        raise ValueError(f"{product} 反推的保证金调整系数 {found['itm']:g} 不为正")
    if abs(found["itm"] - found["otm"]) > COEFFICIENT_TOLERANCE:
        raise ValueError(
            f"{product} 反推的保证金调整系数不一致: 实值分支 {found['itm']:.8f}, "
            f"虚值分支 {found['otm']:.8f}; 期货公司的保证金算法与 CHINA_INDEX 公式不同"
        )
    return round(found["itm"], RATIO_DIGITS)


def futures_margin_ratio(symbol: str, rates: Sequence[Dict[str, Any]]) -> float:
    """标的期货的投资者卖方保证金率(按金额); 按手数收取时报错."""
    if not rates:
        raise ValueError(f"期货公司没有返回 {symbol} 的保证金率")
    rate = rates[0]
    by_volume = float(rate["ShortMarginRatioByVolume"])
    if by_volume != 0:
        raise ValueError(
            f"{symbol} 的保证金按手数收取({by_volume:g} 元/手), "
            "CHINA_COMMODITY 只支持按金额比例"
        )
    ratio = float(rate["ShortMarginRatioByMoney"])
    if ratio <= 0:
        raise ValueError(f"{symbol} 的卖方保证金率 {ratio:g} 不为正")
    # CTP 以 double 传比例(如 0.14000000000000004), 去掉浮点尾差
    return round(ratio, RATIO_DIGITS)


def build_instruments(
    snapshot: Dict[str, Any], index_underlyings: Mapping[str, str]
) -> List[Instrument]:
    """把查询快照(见 ``fetch_snapshot``)转为标的期货与期权的 ``Instrument``.

    :param snapshot: 查询快照
    :param index_underlyings: 中金所期权品种 → 标的指数代码, 如 ``{"IO": "000300.SH"}``
    """
    instruments: List[Instrument] = []
    for row in snapshot["futures"]:
        symbol = str(row["InstrumentID"])
        instruments.append(
            Instrument(
                symbol=symbol,
                asset_type=AssetType.Futures,
                multiplier=float(row["VolumeMultiple"]),
                margin_ratio=futures_margin_ratio(
                    symbol, snapshot["margin_rates"][symbol]
                ),
                tick_size=float(row["PriceTick"]),
                expiry_date=int(row["ExpireDate"]),
            )
        )
    coefficients = {
        product: index_margin_coefficient(product, probes)
        for product, probes in snapshot["index_probes"].items()
    }
    for row in snapshot["options"]:
        symbol = str(row["InstrumentID"])
        kind = str(row["OptionsType"])
        if kind not in (CALL_TYPE, PUT_TYPE):
            raise ValueError(f"{symbol} 的期权类型 {kind!r} 不是认购('1')或认沽('2')")
        strike = float(row["StrikePrice"])
        if strike <= 0:
            raise ValueError(f"{symbol} 的行权价 {strike:g} 不为正")
        expiry = str(row["ExpireDate"]).strip()
        if len(expiry) != 8 or not expiry.isdigit():
            raise ValueError(f"{symbol} 的到期日 {expiry!r} 不是 YYYYMMDD")
        if row["ProductClass"] == INDEX_OPTION_CLASS:
            product = str(row["ProductID"])
            underlying = index_underlyings.get(product)
            if not underlying:
                raise ValueError(
                    f"{symbol} 是中金所 {product} 期权, index_underlyings 没有给出 "
                    f"{product} "
                    "对应的标的指数代码(CTP 不推送指数行情, 须与指数行情来源的代码一致)"
                )
            model, margin_ratio = OptionMarginModel.ChinaIndex, coefficients[product]
        else:
            underlying = str(row["UnderlyingInstrID"])
            model, margin_ratio = OptionMarginModel.ChinaCommodity, 1.0
        instruments.append(
            Instrument(
                symbol=symbol,
                asset_type=AssetType.Option,
                multiplier=float(row["VolumeMultiple"]),
                margin_ratio=margin_ratio,
                tick_size=float(row["PriceTick"]),
                option_type=OptionType.Call if kind == CALL_TYPE else OptionType.Put,
                strike_price=strike,
                expiry_date=int(expiry),
                underlying_symbol=underlying,
                option_margin_model=model,
            )
        )
    return instruments


def fetch_snapshot(
    connection: Mapping[str, str], underlyings: Sequence[str], flow_dir: Path
) -> Dict[str, Any]:
    """登录 CTP 查询一次: 全量合约、标的期货保证金率、中金所期权调整系数探测."""
    session = _QuerySession(connection, flow_dir)
    try:
        rows = session.query("ReqQryInstrument", tdapi.CThostFtdcQryInstrumentField())
        options, futures = select_option_rows(rows, underlyings)
        margin_rates: Dict[str, List[Dict[str, Any]]] = {}
        for row in futures:
            field = tdapi.CThostFtdcQryInstrumentMarginRateField()
            field.BrokerID, field.InvestorID = (
                connection["broker_id"],
                connection["user_id"],
            )
            field.InstrumentID, field.HedgeFlag = row["InstrumentID"], "1"
            margin_rates[str(row["InstrumentID"])] = session.query(
                "ReqQryInstrumentMarginRate", field
            )
        index_probes: Dict[str, List[Dict[str, Any]]] = {}
        for row in options:
            product = str(row["ProductID"])
            if (
                row["ProductClass"] != INDEX_OPTION_CLASS
                or row["OptionsType"] != CALL_TYPE
                or product in index_probes
            ):
                continue
            probes = []
            for label, factor in (("itm", INDEX_PROBE_ITM), ("otm", INDEX_PROBE_OTM)):
                field = tdapi.CThostFtdcQryOptionInstrTradeCostField()
                field.BrokerID, field.InvestorID = (
                    connection["broker_id"],
                    connection["user_id"],
                )
                field.InstrumentID, field.HedgeFlag = row["InstrumentID"], "1"
                field.ExchangeID = row["ExchangeID"]
                field.InputPrice = float(row["PriceTick"])
                field.UnderlyingPrice = float(row["StrikePrice"]) * factor
                costs = session.query("ReqQryOptionInstrTradeCost", field)
                if not costs:
                    raise ValueError(
                        f"期货公司没有返回 {row['InstrumentID']} 的交易成本"
                    )
                probes.append(
                    {
                        "label": label,
                        "instrument": row["InstrumentID"],
                        "underlying_price": field.UnderlyingPrice,
                        "multiplier": row["VolumeMultiple"],
                        "fixed_margin": costs[0]["FixedMargin"],
                    }
                )
            index_probes[product] = probes
        products = {
            str(r["ProductID"])
            for r in options
            if r["ProductClass"] == INDEX_OPTION_CLASS
        }
        without_call = sorted(products - set(index_probes))
        if without_call:
            raise ValueError(
                f"中金所 {', '.join(without_call)} 没有认购合约, 无法反推保证金调整系数"
            )
        return {
            "version": CACHE_VERSION,
            "trading_day": session.trading_day,
            "underlyings": list(underlyings),
            "options": options,
            "futures": futures,
            "margin_rates": margin_rates,
            "index_probes": index_probes,
        }
    finally:
        session.close()


def _parse_time(text: str) -> dt.time:
    try:
        return dt.datetime.strptime(text.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"刷新时刻 {text!r} 不是 HH:MM") from exc


class CTPOptionUniverse:
    """CTP 期权合约表: 按交易时段查询、缓存并转为 ``Instrument``.

    缓存按"最近一个已过的刷新时刻"区分: 同一时段内重启直接读缓存, 到了下一个
    刷新时刻(默认 08:30、20:30, 即日盘、夜盘开盘前 30 分钟)重新查询。

    :param connection: CTP 交易通道参数, 同 ``run_live(gateway_options=...)`` 的
        ``td_front`` / ``broker_id`` / ``user_id`` / ``password`` / ``app_id`` /
        ``auth_code``
    :param underlyings: 跟踪的标的合约代码或标的品种代码, 如 ``["m2601", "IO"]``
    :param index_underlyings: 中金所期权品种 → 标的指数代码, 跟踪中金所期权时必填
    :param cache_dir: 缓存目录(查询快照 JSON 与 CTP 流文件)
    :param refresh_times: 每日刷新时刻(本地时间 HH:MM)
    :param timezone: 刷新时刻所用时区
    """

    def __init__(
        self,
        connection: Mapping[str, str],
        underlyings: Sequence[str],
        index_underlyings: Optional[Mapping[str, str]] = None,
        cache_dir: str | Path = ".akquant_ctp",
        refresh_times: Sequence[str] = DEFAULT_REFRESH_TIMES,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        """校验参数; 不连接 CTP(查询在 ``load`` / ``refresh`` 时进行)."""
        missing = [key for key in CONNECTION_KEYS if not connection.get(key)]
        if missing:
            raise ValueError(f"CTP 连接参数缺少: {', '.join(missing)}")
        if not underlyings:
            raise ValueError("underlyings 不能为空")
        if not refresh_times:
            raise ValueError("refresh_times 不能为空")
        from zoneinfo import ZoneInfo

        self.connection = {key: str(connection[key]) for key in CONNECTION_KEYS}
        self.underlyings = [str(u).strip() for u in underlyings]
        self.index_underlyings = dict(index_underlyings or {})
        self.cache_dir = Path(cache_dir)
        self.refresh_times = sorted(_parse_time(t) for t in refresh_times)
        self.tz = ZoneInfo(timezone)

    def refresh_point(self, now: Optional[dt.datetime] = None) -> dt.datetime:
        """``now`` 之前(含)最近一个刷新时刻."""
        current = (now or dt.datetime.now(self.tz)).astimezone(self.tz)
        for days in range(2):
            day = current.date() - dt.timedelta(days=days)
            for at in reversed(self.refresh_times):
                point = dt.datetime.combine(day, at, tzinfo=self.tz)
                if point <= current:
                    return point
        raise AssertionError("两天内必有刷新时刻")

    def next_refresh(self, now: Optional[dt.datetime] = None) -> dt.datetime:
        """``now`` 之后最近一个刷新时刻."""
        current = (now or dt.datetime.now(self.tz)).astimezone(self.tz)
        for days in range(2):
            day = current.date() + dt.timedelta(days=days)
            for at in self.refresh_times:
                point = dt.datetime.combine(day, at, tzinfo=self.tz)
                if point > current:
                    return point
        raise AssertionError("两天内必有刷新时刻")

    def _cache_path(self, point: dt.datetime) -> Path:
        return self.cache_dir / f"ctp_options_{point:%Y%m%d_%H%M}.json"

    def load(self, now: Optional[dt.datetime] = None) -> List[Instrument]:
        """读本时段的缓存; 没有(或跟踪标的不同)时查询 CTP 并写缓存."""
        path = self._cache_path(self.refresh_point(now))
        if path.exists():
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if (
                snapshot.get("version") == CACHE_VERSION
                and snapshot.get("underlyings") == self.underlyings
            ):
                logger.info("CTP 期权合约表读取缓存 %s", path)
                return build_instruments(snapshot, self.index_underlyings)
        return self.refresh(now)

    def refresh(self, now: Optional[dt.datetime] = None) -> List[Instrument]:
        """查询 CTP, 写本时段缓存, 返回转换后的合约."""
        point = self.refresh_point(now)
        snapshot = fetch_snapshot(
            self.connection, self.underlyings, self.cache_dir / "flow"
        )
        instruments = build_instruments(snapshot, self.index_underlyings)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(point).write_text(
            json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "CTP 期权合约表已查询: 交易日 %s, 期权 %d 个, 标的期货 %d 个",
            snapshot["trading_day"],
            len(snapshot["options"]),
            len(snapshot["futures"]),
        )
        return instruments
