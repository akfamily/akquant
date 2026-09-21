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
"""

STREAM_SCHEMA_VERSION = "1.3"
