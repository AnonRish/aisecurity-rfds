"""Meter kinds from the RFD's "Meter Options" table."""

from __future__ import annotations

import enum

U64_MAX = 2**64 - 1


class Meter(str, enum.Enum):
    FLOPS = "flops"
    INT_OPS = "int_ops"
    MEM_BYTES = "mem_bytes"
    INTERCONNECT_BYTES = "interconnect_bytes"
    ENERGY_MWH = "energy_mwh"
    CYCLES = "cycles"
