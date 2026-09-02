from collections import deque
from typing import Any, Callable, Iterable

from ..log import build_log_extra, get_logger
from .order_audit import record_broker_event
from .symbol_match import normalize_symbol_for_match

logger = get_logger("gateway.live")

#: 会话级委托状态指纹表上限(有界 FIFO)。去重键拆成 ``order:``/
#: ``execution_report:`` 两个命名空间后, 同一批委托约占两倍条目, 等效容量
#: 减半; 上调到 10 万以恢复设计文档写的"5 万单"口径。
_ORDER_STATE_DEDUPE_LIMIT = 100000

#: 外来任务告警去重前缀表上限(有界 FIFO), 同 _ORDER_STATE_DEDUPE_LIMIT 的
#: 防无界增长设计, 只是外来任务前缀数量级远小于委托数, 给一个更小的上限。
_FOREIGN_TASK_PREFIX_DEDUPE_LIMIT = 10000

#: 外来 client_order_id 不符合 ``{broker}-{tag}-{seq}`` 形态(至少两个 '-')
#: 时归入的固定桶——同账户下可能还有非本框架系统(人工下单终端、别家网关)
#: 下的单, 形如 ``A0000001`` 无法按前缀分组, 若仍逐单生成告警键会让
#: _warned_foreign_task_prefixes 无界增长、且每笔单都是一条新 WARNING,
#: 重现刚修掉的稳态刷屏问题。
_UNSTRUCTURED_FOREIGN_TASK_BUCKET = "<unstructured>"


class BrokerEventBridge:
    """Own broker event deduplication, state updates and callback fanout."""

    def __init__(
        self,
        *,
        event_lock: Any,
        event_store: list[tuple[str, Any]],
        event_keys: set[str],
        get_on_broker_event: Callable[[], Callable[[dict[str, Any]], None] | None],
        make_event_key: Callable[[str, Any], str],
        update_broker_state: Callable[[str, Any], None],
        resolve_owner_strategy_id: Callable[[Any], str],
        payload_to_dict: Callable[[Any], dict[str, Any]],
        safe_strategy_callback: Callable[[Any, str, Any], None],
        adapt_strategy_payload: Callable[[str, Any], Any],
        payload_field: Callable[[Any, str], Any],
        resolve_trace_id: Callable[[Any], str] | None = None,
        get_subscribed_symbols: Callable[[], set[str]] | None = None,
        is_known_order: Callable[[str, str], bool] | None = None,
        own_session_prefix: str = "",
        strict_task_isolation: bool = False,
    ) -> None:
        """Bind the queue, state callbacks and observer fanout dependencies."""
        self._event_lock = event_lock
        self._event_store = event_store
        self._event_keys = event_keys
        self._get_on_broker_event = get_on_broker_event
        self._make_event_key = make_event_key
        self._update_broker_state = update_broker_state
        self._resolve_owner_strategy_id = resolve_owner_strategy_id
        self._resolve_trace_id = resolve_trace_id
        self._payload_to_dict = payload_to_dict
        self._safe_strategy_callback = safe_strategy_callback
        self._adapt_strategy_payload = adapt_strategy_payload
        # 会话级已入队 trade_id(不随 drain 清空): 防恢复循环重放同一成交
        # 导致 on_trade/_process_order_groups 重复触发。
        self._seen_trade_ids: set[str] = set()
        self._payload_field = payload_field
        # 会话级委托状态指纹(**不随 drain 清空**): 防 recovery 每轮重放同一状态的
        # 挂单导致 on_order 每轮重推。trade 侧早有 _seen_trade_ids, order 侧此前
        # 没有等价物——这就是"挂单频繁触发 order"那条反馈的活跃根因。
        self._seen_order_states: dict[str, str] = {}
        self._order_state_fifo: deque[str] = deque()
        self._dropped_duplicate_orders = 0
        # 由 Task 3(标的过滤)递增, 本任务先初始化以让 dropped_event_counts() 可读。
        self._dropped_foreign_symbols = 0
        self._get_subscribed_symbols = get_subscribed_symbols
        self._is_known_order = is_known_order
        # 已告警过的外来标的(防刷屏): 首次 WARNING 点名, 之后降 DEBUG。
        self._warned_foreign_symbols: set[str] = set()
        # 已告警过的访问器异常类型(防刷屏, 同上)。
        self._warned_accessor_errors: set[str] = set()
        # 本会话 client_order_id 前缀(形如 ``{broker}-{tag}-``), 空串表示
        # 未启用会话标记判据(向后兼容, 见 _session_layer_verdict)。
        self._own_session_prefix = own_session_prefix
        # 严格模式: 仅在调用方显式传了 session_tag 时才启用(见设计文档
        # "严格模式必须显式启用"一节)。未启用时前缀不匹配也一律放行,
        # 交由第 3 层(订单归属 + 标的)兜底, 与 v0.3.51 行为完全一致。
        self._strict_task_isolation = strict_task_isolation
        # 由第 2 层(会话标记严格拒绝)递增, 与 _dropped_foreign_symbols 分开计。
        self._dropped_foreign_tasks = 0
        # 已告警过的外来任务前缀(防刷屏, 同上); 有界 FIFO 防无界增长, 见
        # _FOREIGN_TASK_PREFIX_DEDUPE_LIMIT。
        self._warned_foreign_task_prefixes: set[str] = set()
        self._foreign_task_prefix_fifo: deque[str] = deque()

    def mark_trades_seen(self, trade_ids: Iterable[str]) -> None:
        """把 trade_id 灌入会话级 dedup 基线.

        已烘进持仓快照的成交, 后续重放经 queue_event 丢弃。
        """
        with self._event_lock:
            for tid in trade_ids:
                if tid:
                    self._seen_trade_ids.add(str(tid))

    def discard_pending_trades(self) -> None:
        """丢弃队列中待派发的成交事件并标记其 trade_id 已见.

        激活时: 这些成交已烘进持仓快照, 不应再 apply_fill/重放 on_trade。
        仅动 trade 事件, order/account 保留。
        """
        with self._event_lock:
            kept: list = []
            for event_name, payload in self._event_store:
                if event_name == "trade":
                    raw = getattr(payload, "trade_id", None)
                    if raw is None and isinstance(payload, dict):
                        raw = payload.get("trade_id")
                    if raw:
                        self._seen_trade_ids.add(str(raw))
                    continue  # 丢弃该 trade 事件
                kept.append((event_name, payload))
            self._event_store[:] = kept

    def _order_state_fingerprint(self, payload: Any) -> str:
        """委托状态指纹: 状态+已成交量+均价+拒单原因, **刻意不含时间戳**.

        口径对齐回测侧 ``strategy_order_events.order_event_key``: 含时间戳的键
        每次重推都会变、去重完全失效(这类缺陷最常见的写法); 只按订单号去重又会
        把 ``New -> PartiallyFilled -> Filled`` 这些真实的状态推进整批吞掉。

        :param payload: 委托快照(``UnifiedOrderSnapshot`` 或等价 dict)。
        :return: 可比较的指纹字符串。
        """
        fields = ("status", "filled_quantity", "avg_fill_price", "reject_reason")
        return "|".join(str(self._payload_field(payload, field)) for field in fields)

    def dropped_event_counts(self) -> dict[str, int]:
        """会话级丢弃计数(去重/过滤/外来任务分开计), 供收尾摘要与诊断读取.

        ``foreign_task`` 与 ``foreign_symbol`` 刻意分开: 前者只在显式传了
        ``session_tag``(严格任务隔离)时才计, 后者是标的判据的兜底计数。

        **三个数在稳态下都会持续增长, 都不是故障信号**: 计数点都在
        ``queue_event`` 入口、早于一切去重, 而柜台的
        ``sync_open_orders``/``sync_today_trades`` 返回的是**全账户**数据 ⇒
        每轮全量 sync 都会对同一笔委托重复 +1。尤其 ``foreign_symbol``:
        未启用严格隔离时, 别的任务/人工下单终端的委托不走 ``foreign_task``
        而是落进这里, 是预期的过滤工作量。判断"是否真有标的匹配问题"要看
        :meth:`dropped_foreign_symbol_names` 点名的标的是不是本任务挂载的,
        不能看这里的数值大小(2026-08-26 反馈就是被旧文案的"应恒为 0"误导)。

        :return: ``{"duplicate_order": N, "foreign_symbol": M, "foreign_task": K}``。
        """
        with self._event_lock:
            return {
                "duplicate_order": self._dropped_duplicate_orders,
                "foreign_symbol": self._dropped_foreign_symbols,
                "foreign_task": self._dropped_foreign_tasks,
            }

    def dropped_foreign_symbol_names(self) -> set[str]:
        """被标的判据挡掉的外来标的名(去重, 会话级累计)的**快照**.

        盘中汇总的触发判据(见
        ``LiveRunner._report_dropped_event_counts_if_changed``): 计数值在稳态
        下必然线性增长, 唯一有诊断价值的是**被挡的标的是谁** —— 若点名的标的
        是本任务挂载的, 才说明标的匹配或配置有问题。

        返回副本而非内部集合: 调用方会对它做集合运算, 交出内部对象会让外部
        改动污染防刷屏的去重状态。

        :return: 去重后的外来标的代码集合。
        """
        with self._event_lock:
            return set(self._warned_foreign_symbols)

    def _session_layer_verdict(self, event_name: str, payload: Any) -> bool | None:
        """会话标记判据(设计文档的第 1/2 层): 按 ``client_order_id`` 前缀强判.

        顺序不可调换, 且**优先于一切标的判据**(包括 ``_accepts_symbol`` 内的
        订单归属映射), 因为 ``BrokerOrderSink`` 经外部信号源直调
        ``submitter.submit_order`` 天然能报出挂载集合之外的标的
        (v0.3.50 I-1 修复过, 不能回归)——本会话报出的单只要前缀命中,
        不论标的是否挂载都必须放行。

        ``account`` 事件没有 ``client_order_id`` 概念(账户资金快照, 不挂靠
        任何一笔委托), 直接短路 ``None`` 交给 ``_accepts_symbol``(它对
        ``account`` 恒放行)——语义上会话判据本就不该管账户事件, 不能靠
        "payload 取不到 client_order_id 而落 None"这种巧合行为。

        - 前缀命中 -> ``True``(放行, 我的单)。
        - 前缀不匹配、``client_order_id`` 非空、且严格模式已启用
          -> ``False``(拒绝, 别的任务的单, 计入 ``foreign_task``)。
        - 其余情况(``account`` 事件/未配置前缀/严格模式未启用/
          ``client_order_id`` 为空/访问器抛异常) -> ``None``, 交由
          ``_accepts_symbol`` 的既有两级判据(订单归属 -> 标的)兜底,
          行为与 v0.3.51 完全一致。

        严格模式为何要显式启用见 ``BrokerRuntime``/``LiveRunner`` 的
        ``session_tag`` 参数: 柜台若截断或改写 ``client_order_id``,
        本任务自己的单前缀也会不匹配, 无条件拒绝会把自己的回报全部吞掉,
        表现为"下单成功却收不到回调"。

        :param event_name: 事件名, ``account`` 一律短路。
        :param payload: 事件载荷。
        :return: 见上, ``True``/``False``/``None`` 三态。
        """
        if event_name == "account":
            return None
        if not self._own_session_prefix:
            return None
        try:
            client_order_id = str(self._payload_field(payload, "client_order_id") or "")
        except Exception as exc:
            self._log_accessor_error(exc)
            return None
        if not client_order_id:
            return None
        if client_order_id.startswith(self._own_session_prefix):
            return True
        if self._strict_task_isolation:
            self._log_foreign_task(client_order_id)
            return False
        return None

    def _foreign_task_dedupe_key(self, client_order_id: str) -> str:
        """把外来 ``client_order_id`` 归到告警去重用的桶.

        本会话生成的 id 形如 ``{broker}-{tag}-{salt}{seq}``, 至少含两个
        ``-``; 同账户下别的 akquant 任务的单也是这个形态, 去掉最后一段
        (序号)按 ``{broker}-{tag}`` 分桶, 同一外来任务只告警一次。

        但同账户下未必只有 akquant 生成的单——人工下单终端、别家网关的
        ``client_order_id`` 可能是 ``A0000001`` 这类不含 ``-`` 的形态,
        `rsplit` 拿不到公共前缀, 每一笔都会生成不同的键: 若仍按原样返回,
        会同时导致(a) 每笔外来单都是一条新 WARNING, (b) 告警去重集合无界
        增长——都是刚修掉的"稳态刷屏"失败模式的同形复现。不符合
        ``{broker}-{tag}-{seq}``(至少两个 ``-``)形态时一律退化到固定桶
        ``_UNSTRUCTURED_FOREIGN_TASK_BUCKET``。

        :param client_order_id: 外来事件的 ``client_order_id``, 非空。
        :return: 用作告警去重键的字符串。
        """
        if client_order_id.count("-") >= 2:
            return client_order_id.rsplit("-", 1)[0]
        return _UNSTRUCTURED_FOREIGN_TASK_BUCKET

    def _log_foreign_task(self, client_order_id: str) -> None:
        """外来任务事件的丢弃留痕: 首次按去重键 WARNING, 之后降 DEBUG.

        去重键按 ``_foreign_task_dedupe_key`` 分桶(而非每个 client_order_id
        一条), 否则同一外来任务的每一笔单都会打一条 WARNING——多任务稳态下
        这必然持续增长, 会重现"稳态刷屏"的失败模式。去重键集合本身也是
        有界 FIFO(``_FOREIGN_TASK_PREFIX_DEDUPE_LIMIT``), 防止大量不同来源
        的外来单把该集合撑到无界。
        """
        foreign_key = self._foreign_task_dedupe_key(client_order_id)
        first_time = foreign_key not in self._warned_foreign_task_prefixes
        if first_time:
            self._warned_foreign_task_prefixes.add(foreign_key)
            self._foreign_task_prefix_fifo.append(foreign_key)
            while (
                len(self._foreign_task_prefix_fifo) > _FOREIGN_TASK_PREFIX_DEDUPE_LIMIT
            ):
                stale = self._foreign_task_prefix_fifo.popleft()
                self._warned_foreign_task_prefixes.discard(stale)
        log = logger.warning if first_time else logger.debug
        log(
            "Dropped event for foreign task/session %s (client_order_id=%s)",
            foreign_key,
            client_order_id,
            extra=build_log_extra(phase="gateway"),
        )

    def _accepts_symbol(self, event_name: str, payload: Any) -> bool:
        """判断事件是否属于本会话: 先认订单归属, 未知单再按标的判.

        判据分两级:

        1. **订单归属优先**: ``order``/``trade``/``execution_report`` 若其
           ``broker_order_id``/``client_order_id`` 命中本会话已知映射(见
           ``is_known_order``), 一律放行, 即便标的不在挂载集合内。这条覆盖
           ``BrokerOrderSink`` 经外部信号源直调 ``submitter.submit_order``、
           不经引擎合约登记表就能合法报出挂载集合之外标的的场景; 否则会把
           自己报出的单错判成"外来"并吞掉, 与本任务要防的事故是同一类。
        2. **标的兜底**: 订单未知(例如跨会话恢复的老挂单)时按标的比对。柜台的
           ``sync_open_orders`` / ``sync_today_trades`` 返回全账户委托与成交,
           不限于本会话订阅的标的, 不过滤会让策略收到不属于自己的委托回报。

        两级判据用到的访问器均为外部注入的 callable, 一旦抛异常整段用
        try/except 兜底放行——不兜底会让异常顺着 ``queue_event`` 一路炸到
        ``broker_recovery`` 的 ``sync_open_orders``/``sync_today_trades``
        循环, 被外层宽 ``except Exception`` 吞掉整批剩余委托, 且默认
        ``recovery_mode="compatible"`` 下连日志都没有。

        所有边界情况一律放行(无访问器 / 订阅集为空 / payload 无 symbol /
        访问器抛异常): 吞掉真实回报的代价远大于多派发一条。``account`` 事件
        无标的概念, 直接放行。

        :param event_name: 事件名。
        :param payload: 事件载荷。
        :return: 是否应当入队派发。
        """
        if event_name == "account":
            return True
        normalized = ""
        try:
            if self._is_own_order(payload):
                return True
            if self._get_subscribed_symbols is None:
                return True
            allowed = self._get_subscribed_symbols()
            if not allowed:
                return True
            normalized = normalize_symbol_for_match(
                self._payload_field(payload, "symbol")
            )
            if not normalized:
                return True
            if normalized in allowed:
                return True
        except Exception as exc:
            self._log_accessor_error(exc)
            return True
        self._log_foreign_symbol(event_name, normalized)
        return False

    def _is_own_order(self, payload: Any) -> bool:
        """查询 payload 对应的委托是否为本会话已知(已建立 id 映射)的订单.

        :param payload: 事件载荷。
        :return: 命中已知映射时 True; 无访问器、无订单号或未命中时 False
            (未命中不代表"不是我的", 由标的判据兜底决定)。
        """
        if self._is_known_order is None:
            return False
        broker_order_id = str(self._payload_field(payload, "broker_order_id") or "")
        client_order_id = str(self._payload_field(payload, "client_order_id") or "")
        if not broker_order_id and not client_order_id:
            return False
        return self._is_known_order(broker_order_id, client_order_id)

    def _log_foreign_symbol(self, event_name: str, symbol: str) -> None:
        """外来标的事件的丢弃留痕: 首次 WARNING 点名, 之后同标的降 DEBUG.

        去重集合的读写持 ``_event_lock``(日志调用本身放在锁外, 不把 IO 拖进
        临界区): 该集合同时是 :meth:`dropped_foreign_symbol_names` 的数据源,
        而写入方是行情/推送线程、读取方是 recovery 线程 —— 无锁复制会在
        "复制期间另一线程 add"时抛 ``Set changed size during iteration``。
        顺带让并发下同一标的只打一条 WARNING(此前两个线程可能各打一条)。
        """
        with self._event_lock:
            first_time = symbol not in self._warned_foreign_symbols
            self._warned_foreign_symbols.add(symbol)
        log = logger.warning if first_time else logger.debug
        log(
            "Dropped %s event for unsubscribed symbol %s",
            event_name,
            symbol,
            extra=build_log_extra(phase="gateway", symbol=symbol),
        )

    def _log_accessor_error(self, exc: Exception) -> None:
        """标的过滤访问器抛异常时的降噪留痕: 按异常类型只警告一次, 之后降 DEBUG."""
        kind = type(exc).__name__
        first_time = kind not in self._warned_accessor_errors
        self._warned_accessor_errors.add(kind)
        log = logger.warning if first_time else logger.debug
        log(
            "Symbol filter accessor raised %s; allowing event through",
            kind,
            exc_info=exc if first_time else None,
            extra=build_log_extra(phase="gateway"),
        )

    def queue_event(self, event_name: str, payload: Any) -> None:
        """Add a broker event to the dispatch queue with semantic deduplication."""
        # 会话标记判据(第 1/2 层)优先于标的判据(第 3 层, _accepts_symbol
        # 内的既有两级): True 时直接放行(跳过标的判据), False 时直接拒绝
        # (计入 foreign_task), None 时才落到 _accepts_symbol。
        session_verdict = self._session_layer_verdict(event_name, payload)
        if session_verdict is False:
            with self._event_lock:
                self._dropped_foreign_tasks += 1
            return
        if session_verdict is None and not self._accepts_symbol(event_name, payload):
            with self._event_lock:
                self._dropped_foreign_symbols += 1
            return
        event_key = self._make_event_key(event_name, payload)
        trade_id = ""
        if event_name == "trade":
            # getattr 优先、dict 兜底: 与 _make_event_key 的 _payload_field 语义一致,
            # 且对 slotted payload 稳健(_payload_to_dict 依赖 __dict__ 会漏)。
            raw = getattr(payload, "trade_id", None)
            if raw is None and isinstance(payload, dict):
                raw = payload.get("trade_id")
            trade_id = str(raw) if raw else ""
        state_key = ""
        state_fingerprint = ""
        if event_name in ("order", "execution_report"):
            order_id = str(self._payload_field(payload, "broker_order_id") or "")
            if order_id:
                # 键带事件类型: order 与 execution_report 是两类独立回调, 内置
                # broker(ctp/miniqmt/ptrade)对同一次状态变化会用同一 payload 成对
                # 派发 order + execution_report, 四个指纹字段逐字相同; 共用命名
                # 空间会让第二个事件被误判"已派发过"而永久吞掉。
                state_key = f"{event_name}:{order_id}"
                state_fingerprint = self._order_state_fingerprint(payload)
        with self._event_lock:
            if trade_id:
                if trade_id in self._seen_trade_ids:
                    return  # 会话级: 该成交已入队(实盘推送/恢复重放), 丢弃
                self._seen_trade_ids.add(trade_id)
            if state_key:
                if self._seen_order_states.get(state_key) == state_fingerprint:
                    self._dropped_duplicate_orders += 1
                    return  # 会话级: 该委托的这个状态已派发过
            if event_key in self._event_keys:
                return
            # 指纹提交刻意放在 event_key 批内键否决之后: 两层去重字段集不一致
            # (指纹多含 avg_fill_price/reject_reason), 若在此之前提交, 会出现
            # "指纹已写成新值但事件被批内键丢弃"的洞——柜台下一次再推同一状态
            # 时指纹命中, 永久吞掉这条修正。只为真正入队的事件记指纹。
            if state_key:
                if state_key not in self._seen_order_states:
                    self._order_state_fifo.append(state_key)
                    while len(self._order_state_fifo) > _ORDER_STATE_DEDUPE_LIMIT:
                        stale = self._order_state_fifo.popleft()
                        self._seen_order_states.pop(stale, None)
                self._seen_order_states[state_key] = state_fingerprint
            self._event_keys.add(event_key)
            self._event_store.append((event_name, payload))

    def drain_events(self, strategy: Any) -> None:
        """Drain queued broker events, update state and dispatch callbacks."""
        with self._event_lock:
            events = list(self._event_store)
            self._event_store.clear()
            self._event_keys.clear()
        for event_name, payload in events:
            adapted = self._adapt_strategy_payload(event_name, payload)
            self._update_broker_state(event_name, payload)
            self._emit_observer_event(event_name, payload)
            record_broker_event(
                event_name,
                payload,
                owner_strategy_id=self._resolve_owner_strategy_id(payload),
                trace_id=(
                    self._resolve_trace_id(payload)
                    if self._resolve_trace_id is not None
                    else None
                ),
            )
            self._dispatch_strategy_event(strategy, event_name, adapted)

    def emit_observer_event(self, event_name: str, payload: Any) -> None:
        """Emit a normalized event snapshot to the optional observer hook."""
        self._emit_observer_event(event_name, payload)

    def _emit_observer_event(self, event_name: str, payload: Any) -> None:
        on_broker_event = self._get_on_broker_event()
        if on_broker_event is None:
            return
        owner_strategy_id = self._resolve_owner_strategy_id(payload)
        payload_dict = self._payload_to_dict(payload)
        try:
            on_broker_event(
                {
                    "event_type": event_name,
                    "owner_strategy_id": owner_strategy_id,
                    "payload": payload_dict,
                }
            )
        except Exception as exc:
            logger.warning(
                "Broker event observer failed",
                exc_info=exc,
                extra=build_log_extra(
                    phase="gateway",
                    strategy_id=owner_strategy_id,
                    slot=owner_strategy_id if owner_strategy_id != "_default" else None,
                    symbol=str(payload_dict.get("symbol", "") or "").strip() or None,
                    order_id=payload_dict.get("broker_order_id")
                    or payload_dict.get("order_id"),
                    client_order_id=payload_dict.get("client_order_id"),
                ),
            )

    def _dispatch_strategy_event(
        self,
        strategy: Any,
        event_name: str,
        payload: Any,
    ) -> None:
        # `payload` is already adapted by `drain_events` (via `_adapt_strategy_payload`)
        # while the request cache was still populated, so it's dispatched as-is here.
        if event_name == "order":
            self._safe_strategy_callback(strategy, "on_order", payload)
        elif event_name == "trade":
            self._safe_strategy_callback(strategy, "on_trade", payload)
            # broker_live 下由真实成交驱动 OCO/Bracket 协调(回测由引擎驱动);
            # 经 _safe_strategy_callback 异常隔离, 无组时 no-op。
            self._safe_strategy_callback(strategy, "_process_order_groups", payload)
        elif event_name == "execution_report":
            self._safe_strategy_callback(strategy, "on_execution_report", payload)
        elif event_name == "account":
            self._safe_strategy_callback(strategy, "on_portfolio_update", payload)
