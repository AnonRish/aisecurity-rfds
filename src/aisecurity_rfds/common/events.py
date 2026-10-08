"""Tiny shared event types so independent RFD packages can interoperate without importing each other."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TamperEvent:
    """A tamper detection emitted by a sensor / enclosure model (see RFDs 020, 028, 036)."""

    source: str
    severity: float  # 0.0 (noise) .. 1.0 (certain breach)
    timestamp_ns: int
    detail: str = ""
