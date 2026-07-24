"""Named fill modes — the sole public way to specify order execution timing.

Replaces the flat ``fill_policy`` dict (price_basis × bar_offset × temporal),
whose Cartesian product allowed illegal/inert combinations. Each named mode
carries only the parameters that apply to it; ``timer_fill_timing`` exists only
on :class:`CurrentClose`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class FillMode:
    """Base class. Do not instantiate directly."""

    def _to_core(self) -> Tuple[str, int, str]:
        """Translate to the engine core triple ``(price_basis, bar_offset, temporal)``.

        This is the single source of truth for mode → core mapping.
        """
        raise NotImplementedError("FillMode is abstract; use a concrete subclass")


@dataclass(frozen=True)
class NextOpen(FillMode):
    """Fill at the open of the next bar (default; no look-ahead)."""

    def _to_core(self) -> Tuple[str, int, str]:
        return ("open", 1, "same_cycle")


@dataclass(frozen=True)
class NextClose(FillMode):
    """Fill at the close of the next bar."""

    def _to_core(self) -> Tuple[str, int, str]:
        return ("close", 1, "same_cycle")


@dataclass(frozen=True)
class NextAverage(FillMode):
    """Fill at next bar's OHLC4 average."""

    def _to_core(self) -> Tuple[str, int, str]:
        return ("ohlc4", 1, "same_cycle")


@dataclass(frozen=True)
class NextHighLowMid(FillMode):
    """Fill at next bar's (high+low)/2."""

    def _to_core(self) -> Tuple[str, int, str]:
        return ("hl2", 1, "same_cycle")


@dataclass(frozen=True)
class CurrentClose(FillMode):
    """Fill at the current bar's close.

    ``timer_fill_timing`` affects ONLY how on_timer orders resolve under this
    mode (akquant's timer is a first-class fillable event):
      - "immediate": timer fires -> fill at current close in the same cycle.
      - "deferred": timer does not constitute a fill point; defer to next bar.
    It has no effect on plain bar orders.
    """

    timer_fill_timing: Literal["immediate", "deferred"] = "immediate"

    def _to_core(self) -> Tuple[str, int, str]:
        temporal = (
            "same_cycle" if self.timer_fill_timing == "immediate" else "next_event"
        )
        return ("close", 0, temporal)


def fill_mode_from_core(price_basis: str, bar_offset: int, temporal: str) -> FillMode:
    """Rebuild a FillMode from an engine core triple for checkpoints."""
    key = (price_basis, int(bar_offset))
    if key == ("open", 1):
        return NextOpen()
    if key == ("close", 1):
        return NextClose()
    if key == ("ohlc4", 1):
        return NextAverage()
    if key == ("hl2", 1):
        return NextHighLowMid()
    if key == ("close", 0):
        return CurrentClose(
            timer_fill_timing="deferred" if temporal == "next_event" else "immediate"
        )
    raise ValueError(
        f"Unrepresentable core triple: {(price_basis, bar_offset, temporal)}"
    )
