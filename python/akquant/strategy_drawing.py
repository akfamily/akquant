"""``Strategy.draw``：策略/指标脚本在 K 线上画线、框、标注、折线、斐波那契与表格.

身份键 ``(owner_strategy_id, symbol, drawing_id)``；同 id 再画即整体覆盖，
内容未变不发事件；
每个 ``(symbol, kind)`` 最多 ``strategy.max_drawings_per_kind`` 个，超出时淘汰最早创建的
（与 Pine 的 ``max_labels_count`` 行为一致）。见 spec 第 3、4 节。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Optional

from .chart import _drawing as D
from .chart import bar_end_ns, timestamp_ms_from_ns
from .log import get_logger as _get_logger

logger = _get_logger("drawing")

_CONTENT_KEYS = ("kind", "pane", "points", "text", "style", "table")


class DrawingApi:
    """绑定到单个策略实例（即单个 owner）的绘图入口."""

    def __init__(self, strategy: Any) -> None:
        """绑定策略实例，初始化存储与告警去重集合."""
        self._strategy = strategy
        # (symbol, drawing_id) -> 已发出的完整对象
        self._store: dict[tuple[str, str], dict[str, Any]] = {}
        # (symbol, kind) -> drawing_id 按创建先后排列（覆盖不改变位置）
        self._index: dict[tuple[str, str], OrderedDict[str, None]] = {}
        self._seq = 0
        self._warned: set[str] = set()

    # ---- 公共方法 -------------------------------------------------------
    def line(
        self,
        id: str,
        p1: Any,
        p2: Any,
        *,
        symbol: Optional[str] = None,
        extend: str = "none",
        color: Optional[str] = None,
        width: float = 1,
        style: str = "solid",
    ) -> str:
        """画线段 / 射线 / 直线；同 id 整体覆盖."""
        return self._draw(
            "line",
            id,
            symbol,
            lambda tz, end: D.build_line(
                p1,
                p2,
                tz=tz,
                to_end_ns=end,
                extend=extend,
                color=color,
                width=width,
                style=style,
            ),
        )

    def box(
        self,
        id: str,
        top_left: Any,
        bottom_right: Any,
        *,
        symbol: Optional[str] = None,
        border_color: Optional[str] = None,
        fill_color: Optional[str] = None,
        width: float = 1,
        text: str = "",
    ) -> str:
        """画矩形框."""
        return self._draw(
            "box",
            id,
            symbol,
            lambda tz, end: D.build_box(
                top_left,
                bottom_right,
                tz=tz,
                to_end_ns=end,
                border_color=border_color,
                fill_color=fill_color,
                width=width,
                text=text,
            ),
        )

    def label(
        self,
        id: str,
        point: Any,
        text: str,
        *,
        symbol: Optional[str] = None,
        style: str = "down",
        color: Optional[str] = None,
        text_color: Optional[str] = None,
        size: str = "normal",
    ) -> str:
        """画文字标注."""
        return self._draw(
            "label",
            id,
            symbol,
            lambda tz, end: D.build_label(
                point,
                text,
                tz=tz,
                to_end_ns=end,
                style=style,
                color=color,
                text_color=text_color,
                size=size,
            ),
        )

    def polyline(
        self,
        id: str,
        points: Any,
        *,
        symbol: Optional[str] = None,
        closed: bool = False,
        color: Optional[str] = None,
        width: float = 1,
        fill_color: Optional[str] = None,
    ) -> str:
        """画折线 / 多边形."""
        return self._draw(
            "polyline",
            id,
            symbol,
            lambda tz, end: D.build_polyline(
                points,
                tz=tz,
                to_end_ns=end,
                closed=closed,
                color=color,
                width=width,
                fill_color=fill_color,
            ),
        )

    def fib(
        self,
        id: str,
        p1: Any,
        p2: Any,
        *,
        symbol: Optional[str] = None,
        color: Optional[str] = None,
    ) -> str:
        """画斐波那契回撤."""
        return self._draw(
            "fib",
            id,
            symbol,
            lambda tz, end: D.build_fib(p1, p2, tz=tz, to_end_ns=end, color=color),
        )

    def table(
        self,
        id: str,
        rows: Any,
        *,
        header: Any = None,
        symbol: Optional[str] = None,
        position: str = "top_right",
        precision: int = 2,
        bg_color: Optional[str] = None,
        border_color: Optional[str] = None,
    ) -> str:
        """画屏幕固定位置的表格（symbol="*" 为全局表）."""
        return self._draw(
            "table",
            id,
            symbol,
            lambda tz, end: D.build_table(
                rows,
                header=header,
                position=position,
                precision=precision,
                bg_color=bg_color,
                border_color=border_color,
            ),
        )

    def delete(self, id: str, *, symbol: Optional[str] = None) -> None:
        """删除指定对象；不存在时静默."""
        drawing_id = self._check_id(id)
        key = (self._symbol(symbol, allow_global=True), drawing_id)
        existing = self._store.pop(key, None)
        if existing is not None:
            self._unindex(key, existing["kind"])
            self._emit("delete", self._identity(existing))

    def clear(
        self, kind: Optional[str] = None, *, symbol: Optional[str] = None
    ) -> None:
        """清除对象，可按 kind / symbol 过滤."""
        if kind is not None and kind not in D.DRAWING_KINDS:
            raise ValueError(f"kind 必须是 {', '.join(D.DRAWING_KINDS)} 之一")
        for key, existing in sorted(
            self._store.items(), key=lambda kv: kv[1]["created_seq"]
        ):
            if kind is not None and existing["kind"] != kind:
                continue
            if symbol is not None and key[0] != symbol:
                continue
            del self._store[key]
            self._unindex(key, existing["kind"])
            self._emit("delete", self._identity(existing))

    # ---- 内部 -----------------------------------------------------------
    def _check_id(self, id: Any) -> str:
        text = str(id or "").strip()
        if not text:
            raise ValueError("绘图对象 id 不能为空")
        return text

    def _symbol(self, symbol: Optional[str], *, allow_global: bool) -> str:
        if symbol == D.GLOBAL_SYMBOL:
            if not allow_global:
                raise ValueError("symbol='*'（全局）只适用于 draw.table")
            return D.GLOBAL_SYMBOL
        return str(self._strategy._resolve_symbol(symbol))

    def _tz(self) -> str:
        return str(getattr(self._strategy, "timezone", None) or "Asia/Shanghai")

    def _end_ns_fn(self) -> Callable[[int], int]:
        """当前 bar 的时间戳按 bar_end_ns 取区间终点，与指标点 timestamp_ms 同口径."""
        bar = getattr(self._strategy, "current_bar", None)
        if bar is None:
            return lambda ns: ns
        bar_ts = int(bar.timestamp)
        start = getattr(bar, "interval_start", None)
        return lambda ns: bar_end_ns(ns, start) if ns == bar_ts else ns

    def _owner(self) -> str:
        return str(
            getattr(self._strategy, "_owner_strategy_id", "_default") or "_default"
        )

    def _identity(self, d: dict[str, Any]) -> dict[str, Any]:
        """Delete 事件载荷：身份四字段 + updated_ts（流事件 ts，对外消息会去掉）."""
        out = {k: d[k] for k in ("owner_strategy_id", "drawing_id", "kind", "symbol")}
        out["updated_ts"] = self._updated_ts()
        return out

    def _draw(
        self,
        kind: str,
        id: Any,
        symbol: Optional[str],
        build: Callable[[str, Callable[[int], int]], dict[str, Any]],
    ) -> str:
        drawing_id = self._check_id(id)
        sym = self._symbol(symbol, allow_global=(kind == "table"))
        body = build(self._tz(), self._end_ns_fn())
        if body.pop("truncated", False):
            self._warn_once(
                f"truncate:{kind}",
                "绘图对象 %s(%s) 的文字超过 %d 字符，已截断",
                drawing_id,
                kind,
                D.MAX_TEXT_LEN,
            )
        key = (sym, drawing_id)
        existing = self._store.get(key)
        if existing is not None and existing["kind"] != kind:
            raise ValueError(
                f"id {drawing_id!r} 已被 {existing['kind']} 使用，"
                f"改画 {kind} 请先 draw.delete({drawing_id!r})"
            )
        if existing is not None and all(existing[k] == body[k] for k in _CONTENT_KEYS):
            return drawing_id
        if existing is None:
            self._evict_if_full(sym, kind)
            self._seq += 1
            created_seq = self._seq
        else:
            created_seq = existing["created_seq"]
        drawing = {
            "owner_strategy_id": self._owner(),
            "drawing_id": drawing_id,
            "symbol": sym,
            **body,
            "created_seq": created_seq,
            "updated_ts": self._updated_ts(),
            "phase": str(getattr(self._strategy, "_indicator_phase", "run") or "run"),
        }
        self._store[key] = drawing
        if existing is None:
            self._index.setdefault((sym, kind), OrderedDict())[drawing_id] = None
        self._emit("upsert", drawing)
        return drawing_id

    def _updated_ts(self) -> int:
        try:
            return timestamp_ms_from_ns(int(self._strategy._resolve_record_timestamp()))
        except RuntimeError:
            return 0

    def _unindex(self, key: tuple[str, str], kind: str) -> None:
        bucket = self._index.get((key[0], kind))
        if bucket is not None:
            bucket.pop(key[1], None)
            if not bucket:
                del self._index[(key[0], kind)]

    def _evict_if_full(self, sym: str, kind: str) -> None:
        limit = int(getattr(self._strategy, "max_drawings_per_kind", 500))
        bucket = self._index.get((sym, kind))
        while bucket and len(bucket) >= limit:
            oldest_id, _ = bucket.popitem(last=False)
            oldest = self._store.pop((sym, oldest_id))
            self._emit("delete", self._identity(oldest))
            self._warn_once(
                f"evict:{kind}",
                "%s 的 %s 对象超过 max_drawings_per_kind=%d，开始淘汰最早创建的",
                self._owner(),
                kind,
                limit,
            )
        if bucket is not None and not bucket:
            del self._index[(sym, kind)]

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(msg, *args)

    def _emit(self, op: str, drawing: dict[str, Any]) -> None:
        sink = getattr(self._strategy, "_indicator_recorder", None)
        if sink is None:
            return
        method = getattr(sink, "record_drawing", None)
        if not callable(method):
            if "no_sink" not in self._warned:
                self._warned.add("no_sink")
                logger.info(
                    "IndicatorSink %s 未实现 record_drawing，绘图对象不会送达",
                    type(sink).__name__,
                )
            return
        method(op=op, drawing=drawing)
