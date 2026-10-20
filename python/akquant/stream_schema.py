"""Single source of truth for the frontend stream message schema version.

Both stream bridges (:mod:`akquant.indicator_stream` and
:mod:`akquant.trade_stream`) stamp every message envelope with
``schema_version`` so a frontend can negotiate or gracefully degrade when the
contract evolves. The version follows ``MAJOR.MINOR``:

- **MAJOR** bumps on breaking envelope/payload changes (fields removed or
  re-typed); a consumer built for an older major should refuse or downgrade.
- **MINOR** bumps on backward-compatible additions (new optional fields); an
  older consumer can safely ignore what it does not recognize.

History:
- ``1.0`` — initial versioned envelope. Shared keys: ``channel``, ``type``,
  ``run_id``, ``seq``, ``ts``, ``symbol``, ``level``, ``schema_version``.
- ``1.1`` — added ``fill.position_effect`` to the trade channel (backward
  compatible; older consumers can ignore the new field).
- ``1.2`` — added ``timestamp_ms`` to the indicator channel's ``indicator``
  (point) and ``snapshot`` objects, so the stream exit agrees with
  ``indicator_df()`` / ``export_indicators()`` on the field (backward
  compatible; older consumers can keep reading the nanosecond ``timestamp``).
- ``1.3`` — added ``confirmed`` to the indicator channel's ``indicator``
  (point) and ``snapshot`` objects. It is currently always ``True``: every
  indicator is computed after its bar closes. It exists now so that live
  intrabar indicators can later emit provisional points with ``False``
  without a MAJOR bump — a frontend that hardcodes "field absent means
  always confirmed" would make that addition breaking. Consumers should
  treat a later point with the same ``(indicator_key, symbol, timestamp)``
  as superseding an earlier one.
- ``1.4`` — added ``phase`` to the indicator channel's ``indicator`` (point)
  and ``snapshot`` objects (and to each snapshot item). ``"preload"`` marks
  points replayed from ``run_live(preload_history=)`` before the first live
  bar; ``"run"`` marks everything else (all backtest points, all live points
  after the preload). A snapshot is ``"preload"`` only when every item is.
  Absent means ``"run"``. The preload and live segments are produced by the
  same indicator instances, so a frontend can draw them as one continuous
  series and just style or filter on ``phase``.
- ``1.5`` — added ``series_key`` to the indicator channel's ``indicator``
  (point) object, and ``owner_strategy_id`` + ``series_key`` to each snapshot
  item; the snapshot gained ``series_keys`` (parallel to ``indicator_keys``),
  and its ``items_by_key`` / ``value_by_key`` are now keyed by ``series_key``.
  ``series_key`` follows the same rule as the historical
  ``akquant.indicator_identity.series_keys``: the plain ``indicator_key`` when
  only one owner uses it in the session, ``"{owner}:{key}"`` otherwise — so two
  owners (e.g. the main strategy and an ``aq.Indicator`` script) sharing a key
  no longer collapse into one line or overwrite each other. Absent (1.4
  events) means ``indicator_key``. When a second owner starts using a key
  mid-run, the switch to prefixed names happens from that message on; earlier
  points are not rewritten. Single-owner sessions see no value changes. The
  supersede rule from ``1.3`` is scoped by owner:
  ``(owner_strategy_id, indicator_key, symbol, timestamp)``.
- ``1.6`` — added ``group`` and ``role`` to the indicator channel's
  ``indicator`` (point) object and to each snapshot item, and the new
  ``render_type`` value ``"candle"``. A candle indicator
  (``Strategy.I(outputs=("open","high","low","close"), render_type="candle")``)
  still emits one single-value point per component (``indicator_key`` like
  ``ha.open``); ``group`` is the declaration name (``ha``) and ``role`` is the
  component (``open``/``high``/``low``/``close``), so a frontend can merge the
  four points of one ``(owner_strategy_id, group, symbol, timestamp)`` into a
  bar. Both fields are ``""`` for every other render type, and absent
  (1.5 events) means ``""``.
"""

STREAM_SCHEMA_VERSION = "1.6"
