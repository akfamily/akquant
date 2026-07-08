"""
FXMacroData economic-calendar integration example for AKQuant.

The example fetches the FXMacroData macro calendar and shows how a strategy can
skip entries near top-tier macro releases. USD calendar access works without an
API key; set FXMD_API_KEY for protected currencies or endpoints.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FXMACRODATA_API_ROOT = "https://fxmacrodata.com/api/v1"


def _parse_event_time(row: dict[str, Any]) -> datetime | None:
    value = row.get("announcement_datetime_utc") or row.get("announcement_datetime_local")
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def fetch_fxmacrodata_calendar(
    currency: str = "usd",
    start_date: str | None = None,
    end_date: str | None = None,
    top_tier_only: bool = True,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {}
    api_key = os.getenv("FXMD_API_KEY")
    if api_key:
        params["api_key"] = api_key
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    query = f"?{urlencode(params)}" if params else ""
    url = f"{FXMACRODATA_API_ROOT}/calendar/{currency.lower()}{query}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "akquant-fxmacrodata-example"})
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = list(payload.get("data") or [])
    if top_tier_only:
        rows = [row for row in rows if row.get("top_tier_for_currency") or row.get("market_tier") == 1]
    return rows


def has_macro_event_near(
    events: list[dict[str, Any]],
    timestamp: datetime,
    before: timedelta = timedelta(hours=4),
    after: timedelta = timedelta(hours=2),
) -> bool:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    for row in events:
        event_time = _parse_event_time(row)
        if event_time is None:
            continue
        if event_time - before <= timestamp <= event_time + after:
            return True
    return False


if __name__ == "__main__":
    calendar = fetch_fxmacrodata_calendar("usd", top_tier_only=True)
    print(f"Loaded {len(calendar)} top-tier USD macro events from FXMacroData")
    for event in calendar[:5]:
        print(event.get("announcement_datetime_utc"), event.get("name"))

