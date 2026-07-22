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
"""

STREAM_SCHEMA_VERSION = "1.0"
