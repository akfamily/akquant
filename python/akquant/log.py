import json
import logging
import re
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Optional, Union

# Default format: Time | Level | Logger | Message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
OPTIMIZE_FORMAT = (
    "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOT_LOGGER_NAME = "akquant"
RUST_CONTEXT_MARKER = " [akq_ctx="
CONTEXT_FIELDS = (
    "phase",
    "event_time",
    "event_time_iso",
    "strategy_id",
    "slot",
    "symbol",
    "order_id",
    "client_order_id",
    # Correlation id spanning signal→submit→update→fill for one logical order
    # (RFC G5). In broker_live this is the group/root client_order_id.
    "trace_id",
    # Order-lifecycle audit dimensions (RFC G1). Present on every record but
    # only populated by build_order_audit_extra; None elsewhere (JSON skips it).
    "event",
    "side",
    "price",
    "quantity",
    "order_status",
    "order_type",
    "trade_id",
    "reason",
    # 拒单来源(RFC G1): "local"=本地校验/风控拒的, "broker"=柜台回绝的。
    # 两者的处置完全不同(本地拒单意味着单没发出去, 柜台拒单意味着发出去被回绝),
    # 对账时必须能分开筛。
    "origin",
)
# The dedicated order-audit logger namespace (child of the root logger).
ORDER_AUDIT_LOGGER_NAME = "audit.order"
# Records under this namespace carry self-contained human-readable messages, so
# the text console formatter omits the (redundant) structured context suffix.
AUDIT_NAME_PREFIX = f"{ROOT_LOGGER_NAME}.audit."


# Sensitive field masking (G2): keys whose values must never appear verbatim.
# FULL_MASK keys (secrets/credentials) are replaced wholesale; TAIL_MASK keys
# (account identifiers) keep only their last 4 chars for reconciliation.
FULL_MASK_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "app_key",
        "appkey",
        "app_secret",
        "auth_code",
        "authcode",
        "private_key",
    }
)
TAIL_MASK_KEYS = frozenset(
    {
        "user_id",
        "userid",
        "account",
        "account_id",
        "investor_id",
        "broker_id",
        "brokerid",
    }
)
_FULL_MASK = "****"
_MASK_TAIL_KEEP = 4
# Match `key=value`, `key: value`, `key="value"` in rendered messages. Value runs
# until whitespace, comma, or a closing quote — enough for id/secret tokens.
_INLINE_SENSITIVE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<sep>\s*[=:]\s*)"
    r"(?P<quote>[\"']?)"
    r"(?P<value>[^\s,;\"']+)"
    r"(?P=quote)"
)


def _mask_tail(value: str) -> str:
    """Mask a value keeping only its last few characters."""
    if len(value) <= _MASK_TAIL_KEEP:
        return _FULL_MASK
    return f"{_FULL_MASK}{value[-_MASK_TAIL_KEEP:]}"


def mask_sensitive_value(key: str, value: Any) -> Any:
    """Mask a single value when its key is sensitive; else return unchanged."""
    lowered = key.strip().lower()
    if lowered in FULL_MASK_KEYS:
        return _FULL_MASK
    if lowered in TAIL_MASK_KEYS:
        text = str(value).strip()
        return _mask_tail(text) if text else value
    return value


def mask_sensitive_text(message: str) -> str:
    """Mask inline `key=value` sensitive pairs embedded in a rendered message."""
    if not message:
        return message

    def _replace(match: "re.Match[str]") -> str:
        key = match.group("key")
        masked = mask_sensitive_value(key, match.group("value"))
        if masked == match.group("value"):
            return match.group(0)
        quote = match.group("quote")
        return f"{key}{match.group('sep')}{quote}{masked}{quote}"

    return _INLINE_SENSITIVE_RE.sub(_replace, message)


class SensitiveFilter(logging.Filter):
    """Redact sensitive credentials/account ids from records before emission.

    Masking runs at the handler layer so every log statement is covered by
    default — callers cannot leak a secret by forgetting to mask it themselves.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask sensitive structured extras and inline message pairs in place."""
        for key, value in list(vars(record).items()):
            if value is None:
                continue
            masked = mask_sensitive_value(key, value)
            if masked is not value and masked != value:
                setattr(record, key, masked)
        message = record.getMessage()
        redacted = mask_sensitive_text(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def _normalize_context_value(value: Any) -> Optional[str]:
    """Normalize structured logging context values to trimmed strings."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_rust_context_message(message: str) -> tuple[str, dict[str, Any] | None]:
    """Parse an AKQuant Rust log payload appended to the rendered message."""
    marker_index = message.rfind(RUST_CONTEXT_MARKER)
    if marker_index < 0 or not message.endswith("]"):
        return message, None
    payload_text = message[marker_index + len(RUST_CONTEXT_MARKER) : -1]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return message, None
    if not isinstance(payload, dict):
        return message, None
    return message[:marker_index], payload


def _extract_rust_context(record: logging.LogRecord) -> None:
    """Lift AKQuant Rust log context payloads into structured LogRecord fields."""
    if getattr(record, "_akquant_rust_context_parsed", False):
        return
    record._akquant_rust_context_parsed = True
    if not str(getattr(record, "name", "") or "").startswith(ROOT_LOGGER_NAME):
        return
    rendered_message = record.getMessage()
    stripped_message, payload = _parse_rust_context_message(rendered_message)
    if payload is None:
        return
    record.msg = stripped_message
    record.args = ()
    for field_name in CONTEXT_FIELDS:
        if field_name in payload:
            value = payload[field_name]
            if field_name == "event_time":
                setattr(record, field_name, value)
            else:
                setattr(record, field_name, _normalize_context_value(value))
        elif not hasattr(record, field_name):
            setattr(record, field_name, None)


def _install_log_record_factory() -> None:
    """Install a LogRecord factory that restores Rust logging context early."""
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_akquant_rust_context_factory", False):
        return

    def akquant_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = current_factory(*args, **kwargs)
        _extract_rust_context(record)
        return record

    akquant_record_factory._akquant_rust_context_factory = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(akquant_record_factory)


class AKQuantFormatter(logging.Formatter):
    """Formatter that can safely render AKQuant logging context."""

    def __init__(
        self,
        fmt: str,
        *,
        datefmt: str = DATE_FORMAT,
        include_context: bool = False,
        language: str = "en",
    ) -> None:
        """Initialize a formatter with optional AKQuant context rendering."""
        super().__init__(fmt, datefmt=datefmt)
        self.include_context = include_context
        self.language = language

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record and optionally append structured context text."""
        for field_name in CONTEXT_FIELDS:
            if not hasattr(record, field_name):
                setattr(record, field_name, None)

        is_audit = str(record.name).startswith(AUDIT_NAME_PREFIX)
        # Localized console rendering (RFC G6): rebuild the audit line from the
        # structured fields in the target language, then RESTORE the canonical
        # english msg so other handlers (file/JSON) still emit english.
        event = getattr(record, "event", None)
        if is_audit and self.language != "en" and event:
            original_msg, original_args = record.msg, record.args
            record.msg = render_audit_message(event, record.__dict__, self.language)
            record.args = ()
            try:
                return super().format(record)
            finally:
                record.msg, record.args = original_msg, original_args

        rendered = super().format(record)
        if not self.include_context:
            return rendered
        # Audit records are self-contained (message already carries the key
        # fields), so skip the redundant structured suffix in the console text.
        if is_audit:
            return rendered

        context_parts: list[str] = []
        for field_name in (
            "phase",
            "strategy_id",
            "slot",
            "symbol",
            "order_id",
            "client_order_id",
            "trace_id",
            "event_time_iso",
        ):
            value = getattr(record, field_name, None)
            if value is None or value == "":
                continue
            context_parts.append(f"{field_name}={value}")
        if not context_parts:
            return rendered
        return f"{rendered} | {' '.join(context_parts)}"


def _ensure_context_fields(record: logging.LogRecord) -> None:
    """Populate missing AKQuant context fields on a log record."""
    _extract_rust_context(record)
    for field_name in CONTEXT_FIELDS:
        if not hasattr(record, field_name):
            setattr(record, field_name, None)


def _to_jsonable(value: Any) -> Any:
    """Convert log values into JSON-safe payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            return str(value)
    return str(value)


class AKQuantJsonFormatter(logging.Formatter):
    """Formatter that renders AKQuant log records as JSON lines."""

    def __init__(self, *, datefmt: str = DATE_FORMAT) -> None:
        """Initialize a JSON formatter with the configured timestamp format."""
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as one JSON object per line."""
        _ensure_context_fields(record)
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": record.process,
            "process_name": record.processName,
        }
        for field_name in CONTEXT_FIELDS:
            value = getattr(record, field_name, None)
            if value is None or value == "":
                continue
            payload[field_name] = _to_jsonable(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "research": {
        "console_format": DEFAULT_FORMAT,
        "file_format": DEFAULT_FORMAT,
        "console_show_context": False,
        "file_show_context": True,
    },
    "optimize": {
        "console_format": OPTIMIZE_FORMAT,
        "file_format": OPTIMIZE_FORMAT,
        "console_show_context": False,
        "file_show_context": True,
    },
    "live": {
        "console_format": DEFAULT_FORMAT,
        "file_format": DEFAULT_FORMAT,
        "console_show_context": True,
        "file_show_context": True,
    },
}


def _normalize_level(level: Union[str, int]) -> int:
    """Normalize a logging level into its integer form."""
    if isinstance(level, int):
        return level
    normalized_name = level.strip().upper()
    if normalized_name == "WARN":
        normalized_name = "WARNING"
    normalized = getattr(logging, normalized_name, None)
    if isinstance(normalized, int):
        return normalized
    raise ValueError(f"Unknown log level: {level}")


def build_log_extra(
    *,
    phase: Optional[str] = None,
    event_time: Any = None,
    event_time_iso: Optional[str] = None,
    strategy_id: Optional[str] = None,
    slot: Optional[str] = None,
    symbol: Optional[str] = None,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a normalized AKQuant structured logging payload."""
    return {
        "phase": _normalize_context_value(phase),
        "event_time": event_time,
        "event_time_iso": _normalize_context_value(event_time_iso),
        "strategy_id": _normalize_context_value(strategy_id),
        "slot": _normalize_context_value(slot),
        "symbol": _normalize_context_value(symbol),
        "order_id": _normalize_context_value(order_id),
        "client_order_id": _normalize_context_value(client_order_id),
        "trace_id": _normalize_context_value(trace_id),
    }


def _coerce_number(value: Any) -> Any:
    """Keep numeric audit values numeric; drop empty/None; else stringify."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def build_order_audit_extra(
    *,
    event: str,
    strategy_id: Optional[str] = None,
    slot: Optional[str] = None,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    price: Any = None,
    quantity: Any = None,
    client_order_id: Optional[str] = None,
    order_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    order_status: Optional[str] = None,
    order_type: Optional[str] = None,
    trade_id: Optional[str] = None,
    reason: Optional[str] = None,
    origin: Optional[str] = None,
) -> dict[str, Any]:
    """Build a structured order-lifecycle audit payload (RFC G1).

    `order_id` carries the broker order id, matching the existing convention in
    the gateway event bridge. `trace_id` is the logical-order correlation id
    (group/root client_order_id) shared across a leg's submit→update→fill.
    Numeric fields (price/quantity) stay numeric so the JSON audit line is
    machine-parseable for reconciliation.
    """
    return {
        "phase": "gateway",
        "event": _normalize_context_value(event),
        "strategy_id": _normalize_context_value(strategy_id),
        "slot": _normalize_context_value(slot),
        "symbol": _normalize_context_value(symbol),
        "side": _normalize_context_value(side),
        "price": _coerce_number(price),
        "quantity": _coerce_number(quantity),
        "client_order_id": _normalize_context_value(client_order_id),
        "order_id": _normalize_context_value(order_id),
        "trace_id": _normalize_context_value(trace_id),
        "order_status": _normalize_context_value(order_status),
        "order_type": _normalize_context_value(order_type),
        "trade_id": _normalize_context_value(trade_id),
        "reason": _normalize_context_value(reason),
        "origin": _normalize_context_value(origin),
    }


# --- Order-audit message rendering (RFC G6) -------------------------------
# The audit *record* is language-neutral (english `event` code + structured
# fields). The human-readable message is a *rendering* concern: english is the
# canonical (files/JSON/grep), and a localized console renderer can rebuild the
# line from the same structured fields. This mirrors structlog's "swap the
# final renderer" and nautilus_trader's english-only structured logs — we never
# fork message prose across the business code.
SUPPORTED_LANGUAGES = ("en", "zh")
_AUDIT_MESSAGE_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "order_submit": "submit {side} {quantity} {symbol} {price} "
        "[{client_order_id}->{order_id}]",
        "order_fill": "fill {side} {quantity} {symbol} {price} "
        "[{client_order_id}->{order_id} {trade_id}]",
        "order_reject": "reject {symbol} [{client_order_id}] reason: {reason}",
        "order_cancel": "cancel {symbol} [{order_id}]",
        "order_update": "update {symbol} status={order_status} "
        "filled={quantity} [{client_order_id}->{order_id}]",
        "order_submit_unknown": "submit-unknown {side} {quantity} {symbol} "
        "{price} [{client_order_id}] reason: {reason}",
        "order_cancel_failed": "cancel-failed {symbol} [{order_id}] reason: {reason}",
    },
    "zh": {
        "order_submit": "下单 {side} {quantity} {symbol} {price} "
        "[{client_order_id}→{order_id}]",
        "order_fill": "成交 {side} {quantity} {symbol} {price} "
        "[{client_order_id}→{order_id} {trade_id}]",
        "order_reject": "拒单 {symbol} [{client_order_id}] 原因: {reason}",
        "order_cancel": "撤单请求 {symbol} [{order_id}]",
        "order_update": "订单更新 {symbol} 状态={order_status} "
        "已成={quantity} [{client_order_id}→{order_id}]",
        "order_submit_unknown": "报单状态未知 {side} {quantity} {symbol} "
        "{price} [{client_order_id}] 原因: {reason}",
        "order_cancel_failed": "撤单失败 {symbol} [{order_id}] 原因: {reason}",
    },
}
_MARKET_PRICE_TEXT = {"en": "market", "zh": "市价"}


def _compact_number(value: Any) -> str:
    """Render a numeric audit value compactly (drop trailing .0)."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_audit_message(
    event: Optional[str],
    fields: Any,
    language: str = "en",
) -> str:
    """Render one order-audit line from structured fields in the given language.

    `fields` is any mapping/record-dict carrying the audit fields (side, price,
    symbol, ids, …) — typically the `build_order_audit_extra` payload or a
    LogRecord's ``__dict__``. English is the canonical; other languages are a
    pure presentation over the same fields.
    """
    lang = language if language in _AUDIT_MESSAGE_TEMPLATES else "en"
    template = _AUDIT_MESSAGE_TEMPLATES[lang].get(str(event or ""))
    if template is None:
        return str(event or "")

    def _get(name: str) -> Any:
        if isinstance(fields, dict):
            return fields.get(name)
        return getattr(fields, name, None)

    price = _get("price")
    if str(event) in ("order_submit", "order_submit_unknown") and price in (None, ""):
        price_display = _MARKET_PRICE_TEXT[lang]
    elif price in (None, ""):
        price_display = ""
    else:
        price_display = f"@{_compact_number(price)}"

    display = {
        "side": _get("side") or "",
        "quantity": _compact_number(_get("quantity")),
        "symbol": _get("symbol") or "",
        "price": price_display,
        "client_order_id": _get("client_order_id") or "?",
        "order_id": _get("order_id") or "?",
        "trade_id": _get("trade_id") or "",
        "order_status": _get("order_status") or "?",
        "reason": _get("reason") or "",
    }
    text = template.format_map(display)
    # An "update" carrying a reject reason appends it (kept out of the template
    # so the common no-reason path stays clean).
    if str(event) == "order_update" and display["reason"]:
        suffix = "原因" if lang == "zh" else "reason"
        text = f"{text} {suffix}: {display['reason']}"
    # Collapse the double spaces left by empty optional fields.
    return " ".join(text.split())


_install_log_record_factory()


def has_configured_handler(
    name: Optional[str] = None, *, namespace_only: bool = False
) -> bool:
    """Return True when the logger hierarchy has a visible handler configured."""
    current: Optional[logging.Logger] = logging.getLogger(
        ROOT_LOGGER_NAME if name is None else name
    )
    while current is not None:
        if any(
            not isinstance(handler, logging.NullHandler) for handler in current.handlers
        ):
            return True
        if namespace_only and current.name == ROOT_LOGGER_NAME:
            return False
        if not current.propagate:
            return False
        parent = current.parent
        current = parent if isinstance(parent, logging.Logger) else None
    return False


@dataclass
class LogConfig:
    """Advanced logging configuration for AKQuant."""

    level: Union[str, int] = "INFO"
    console: bool = True
    console_level: Optional[Union[str, int]] = None
    console_format: Optional[str] = None
    console_show_context: Optional[bool] = None
    console_json: Optional[bool] = None
    filename: Optional[str] = None
    file_level: Optional[Union[str, int]] = None
    file_format: Optional[str] = None
    file_show_context: Optional[bool] = None
    file_json: Optional[bool] = None
    file_mode: str = "a"
    file_max_bytes: Optional[int] = None
    file_backup_count: int = 3
    profile: Optional[str] = None
    reset_handlers: bool = True
    propagate: bool = True
    mask_sensitive: bool = True
    # Console message language (RFC G6). English is always the canonical stored
    # message (files/JSON); this only re-renders the *console* audit line. Files
    # and JSON stay english regardless, so grep/alerting never fork by language.
    language: str = "en"
    # Dedicated order-lifecycle audit stream (RFC G1). When set, order audit
    # records are additionally written as JSON lines to this rotating file.
    order_audit_file: Optional[str] = None
    order_audit_level: Union[str, int] = "INFO"
    order_audit_max_bytes: Optional[int] = None
    order_audit_backup_count: int = 5


class Logger:
    r"""
    akquant 日志封装.

    :description: 提供控制台与文件日志的快捷配置
    """

    _instance = None

    def __init__(self) -> None:
        """Initialize the Logger."""
        self._logger = logging.getLogger(ROOT_LOGGER_NAME)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = True
        self._handlers: dict[str, logging.Handler] = {}  # key -> handler
        self._sensitive_filter = SensitiveFilter()
        self._order_audit_handler: Optional[logging.Handler] = None

        self._sync_handlers()
        if not self._logger.handlers:
            self._ensure_null_handler()

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance._logger

    def set_level(self, level: Union[str, int]) -> None:
        r"""
        设置日志等级.

        :param level: 日志等级字符串或整数 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        :type level: str | int
        """
        self._logger.setLevel(level)

    def _sync_handlers(self) -> None:
        """同步内部 handler 索引，移除已脱离 logger 的引用."""
        active_handlers = set(self._logger.handlers)
        stale_keys = [
            key
            for key, handler in self._handlers.items()
            if handler not in active_handlers
        ]
        for key in stale_keys:
            del self._handlers[key]

    def _ensure_null_handler(self) -> None:
        """Attach a NullHandler when no other handler is configured."""
        self._sync_handlers()
        if self._logger.handlers:
            return
        handler = logging.NullHandler()
        self._logger.addHandler(handler)
        self._handlers["null"] = handler

    def _remove_handler(self, key: str) -> None:
        """Remove a managed handler by key if present."""
        self._sync_handlers()
        handler = self._handlers.pop(key, None)
        if handler is None:
            return
        self._logger.removeHandler(handler)
        handler.close()

    def _apply_sensitive_filter(self, handler: logging.Handler, enabled: bool) -> None:
        """Attach or detach the shared sensitive-data filter on a handler."""
        if enabled:
            if self._sensitive_filter not in handler.filters:
                handler.addFilter(self._sensitive_filter)
        elif self._sensitive_filter in handler.filters:
            handler.removeFilter(self._sensitive_filter)

    def _remove_null_handler(self) -> None:
        """Remove the fallback NullHandler before enabling visible handlers."""
        self._remove_handler("null")

    def reset_managed_handlers(self) -> None:
        """Remove all AKQuant-managed handlers while preserving external ones."""
        self._sync_handlers()
        for key in list(self._handlers):
            self._remove_handler(key)
        self.disable_order_audit_file()

    def _order_audit_logger(self) -> logging.Logger:
        """Return the dedicated order-audit child logger."""
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{ORDER_AUDIT_LOGGER_NAME}")

    def enable_order_audit_file(
        self,
        filename: str,
        *,
        level: Union[str, int] = "INFO",
        max_bytes: Optional[int] = None,
        backup_count: int = 5,
        mask_sensitive: bool = True,
    ) -> None:
        """Attach a dedicated JSON audit file to the order-audit logger (RFC G1)."""
        self.disable_order_audit_file()
        max_bytes_value = int(max_bytes) if max_bytes is not None else 0
        if max_bytes_value > 0:
            handler: logging.Handler = RotatingFileHandler(
                filename=filename,
                mode="a",
                maxBytes=max_bytes_value,
                backupCount=max(int(backup_count), 1),
                encoding="utf-8",
            )
        else:
            handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
        handler.setFormatter(AKQuantJsonFormatter(datefmt=DATE_FORMAT))
        handler.setLevel(_normalize_level(level))
        if mask_sensitive:
            handler.addFilter(self._sensitive_filter)
        audit_logger = self._order_audit_logger()
        audit_logger.addHandler(handler)
        audit_logger.setLevel(_normalize_level(level))
        self._order_audit_handler = handler

    def disable_order_audit_file(self) -> None:
        """Detach and close the dedicated order-audit file handler if present."""
        if self._order_audit_handler is None:
            return
        self._order_audit_logger().removeHandler(self._order_audit_handler)
        self._order_audit_handler.close()
        self._order_audit_handler = None

    def enable_console(
        self,
        format_str: str = DEFAULT_FORMAT,
        level: Optional[Union[str, int]] = None,
        show_context: bool = False,
        json_output: bool = False,
        mask_sensitive: bool = True,
        language: str = "en",
    ) -> None:
        r"""
        启用控制台日志.

        :param format_str: 日志格式字符串
        :type format_str: str
        :param level: 控制台日志等级，为 None 时不修改
        :type level: str | int, optional
        :param language: 控制台审计消息语言（"en"/"zh"）；文件/JSON 恒英文
        :type language: str
        """
        self._sync_handlers()
        self._remove_null_handler()
        handler = self._handlers.get("console")
        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
            self._logger.addHandler(handler)
            self._handlers["console"] = handler
        self._apply_sensitive_filter(handler, mask_sensitive)
        handler.setFormatter(
            AKQuantJsonFormatter(datefmt=DATE_FORMAT)
            if json_output
            else AKQuantFormatter(
                format_str,
                datefmt=DATE_FORMAT,
                include_context=show_context,
                language=language,
            )
        )
        if level is not None:
            handler.setLevel(_normalize_level(level))

    def disable_console(self) -> None:
        r"""禁用控制台日志."""
        self._remove_handler("console")
        self._ensure_null_handler()

    def enable_file(
        self,
        filename: str,
        format_str: str = DEFAULT_FORMAT,
        mode: str = "a",
        level: Optional[Union[str, int]] = None,
        show_context: bool = False,
        max_bytes: Optional[int] = None,
        backup_count: int = 3,
        json_output: bool = False,
        mask_sensitive: bool = True,
    ) -> None:
        r"""
        启用文件日志.

        :param filename: 日志文件路径
        :type filename: str
        :param format_str: 日志格式字符串
        :type format_str: str
        :param mode: 文件打开模式 ('a' 追加 或 'w' 覆写)
        :type mode: str
        :param level: 文件日志等级，为 None 时不修改
        :type level: str | int, optional
        """
        self._sync_handlers()
        self._remove_null_handler()
        key = f"file_{filename}"
        handler = self._handlers.get(key)
        max_bytes_value = int(max_bytes) if max_bytes is not None else 0
        rotation_enabled = max_bytes_value > 0
        needs_rotating_handler = rotation_enabled
        if handler is not None and needs_rotating_handler != isinstance(
            handler, RotatingFileHandler
        ):
            self._remove_handler(key)
            handler = None
        if handler is None:
            if rotation_enabled:
                handler = RotatingFileHandler(
                    filename=filename,
                    mode=mode,
                    maxBytes=max_bytes_value,
                    backupCount=max(int(backup_count), 1),
                    encoding="utf-8",
                )
            else:
                handler = logging.FileHandler(filename, mode=mode, encoding="utf-8")
            self._logger.addHandler(handler)
            self._handlers[key] = handler
        self._apply_sensitive_filter(handler, mask_sensitive)
        handler.setFormatter(
            AKQuantJsonFormatter(datefmt=DATE_FORMAT)
            if json_output
            else AKQuantFormatter(
                format_str,
                datefmt=DATE_FORMAT,
                include_context=show_context,
            )
        )
        if level is not None:
            handler.setLevel(_normalize_level(level))

    def disable_file(self, filename: Optional[str] = None) -> None:
        """禁用一个或全部文件日志."""
        self._sync_handlers()
        file_keys = [
            key
            for key in self._handlers
            if key.startswith("file_")
            and (filename is None or key == f"file_{filename}")
        ]
        for key in file_keys:
            self._remove_handler(key)
        self._ensure_null_handler()

    def apply_config(self, config: LogConfig) -> logging.Logger:
        """Apply a structured logging configuration to the root logger."""
        if config.reset_handlers:
            self.reset_managed_handlers()

        self._logger.propagate = bool(config.propagate)
        profile_defaults = PROFILE_DEFAULTS.get(str(config.profile or "").strip())
        console_format = (
            config.console_format
            if config.console_format is not None
            else profile_defaults.get("console_format", DEFAULT_FORMAT)
            if profile_defaults
            else DEFAULT_FORMAT
        )
        file_format = (
            config.file_format
            if config.file_format is not None
            else profile_defaults.get("file_format", DEFAULT_FORMAT)
            if profile_defaults
            else DEFAULT_FORMAT
        )
        console_show_context = (
            config.console_show_context
            if config.console_show_context is not None
            else profile_defaults.get("console_show_context", False)
            if profile_defaults
            else False
        )
        file_show_context = (
            config.file_show_context
            if config.file_show_context is not None
            else profile_defaults.get("file_show_context", True)
            if profile_defaults
            else True
        )
        console_json = (
            bool(config.console_json) if config.console_json is not None else False
        )
        file_json = bool(config.file_json) if config.file_json is not None else False

        effective_levels: list[int] = []
        if config.console:
            console_level = config.console_level or config.level
            self.enable_console(
                format_str=console_format,
                level=console_level,
                show_context=console_show_context,
                json_output=console_json,
                mask_sensitive=config.mask_sensitive,
                language=config.language,
            )
            effective_levels.append(_normalize_level(console_level))
        else:
            self.disable_console()

        if config.filename:
            file_level = config.file_level or config.level
            self.disable_file()
            self.enable_file(
                filename=config.filename,
                format_str=file_format,
                mode=config.file_mode,
                level=file_level,
                show_context=file_show_context,
                max_bytes=config.file_max_bytes,
                backup_count=config.file_backup_count,
                json_output=file_json,
                mask_sensitive=config.mask_sensitive,
            )
            effective_levels.append(_normalize_level(file_level))
        else:
            self.disable_file()

        if config.order_audit_file:
            self.enable_order_audit_file(
                filename=config.order_audit_file,
                level=config.order_audit_level,
                max_bytes=config.order_audit_max_bytes,
                backup_count=config.order_audit_backup_count,
                mask_sensitive=config.mask_sensitive,
            )
        else:
            self.disable_order_audit_file()

        if effective_levels:
            self._logger.setLevel(min(effective_levels))
            self._remove_null_handler()
        else:
            self._logger.setLevel(_normalize_level(config.level))
            self._ensure_null_handler()

        return self._logger


# Global helper functions
def get_logger(name: Optional[str] = None) -> logging.Logger:
    r"""
    获取 AKQuant logger 实例.

    :param name: 子 logger 名称；不传时返回根 logger
    :type name: str, optional
    :return: 已初始化的 logger
    :rtype: logging.Logger
    """
    logger = Logger.get_logger()
    if not name or name == ROOT_LOGGER_NAME:
        return logger
    full_name = (
        name
        if name.startswith(f"{ROOT_LOGGER_NAME}.")
        else f"{ROOT_LOGGER_NAME}.{name}"
    )
    return logging.getLogger(full_name)


def set_log_level(level: Union[str, int]) -> None:
    r"""
    设置全局日志等级.

    :param level: 日志等级字符串或整数
    :type level: str | int
    """
    Logger.get_logger().setLevel(level)


def configure_logging(config: LogConfig) -> logging.Logger:
    """Configure the AKQuant root logger via a structured config."""
    logger_manager = Logger._instance or Logger()
    Logger._instance = logger_manager
    return logger_manager.apply_config(config)


def register_logger(
    filename: Optional[str] = None, console: bool = True, level: str = "INFO"
) -> None:
    r"""
    日志一体化配置.

    :param filename: 日志文件路径，提供则写入文件
    :type filename: str, optional
    :param console: 是否输出到控制台
    :type console: bool
    :param level: 日志等级 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    :type level: str
    """
    configure_logging(
        LogConfig(
            level=level.upper(),
            console=console,
            filename=filename,
            file_max_bytes=None,
            reset_handlers=True,
        )
    )
