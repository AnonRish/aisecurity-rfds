"""Fault-injectable non-volatile memory bundle for a simulated chip.

Assumptions (a model, not a measurement of any real part): every flash write and every fuse burn is one
numbered *persistent operation*. A ``FaultPlan`` can cut power before, during (flash only: a torn write)
or after any numbered operation. An attacker can additionally starve the high-voltage supply
(``set_hv_disrupted``), after which flash writes and fuse burns silently do nothing, and can image and
restore the flash (``flash.snapshot`` / ``flash.physical_restore``). Fuses cannot be restored.
"""

from __future__ import annotations

from aisecurity_rfds.common.nvm import FlashStore, OtpFuseBank, UnaryCounter


class PowerLoss(Exception):
    """Power was cut at an injected fault point; volatile state is gone, NVM keeps what was written."""


class FaultPlan:
    """Logs persistent operations and fires one injected power loss at a chosen operation."""

    def __init__(self) -> None:
        self.log: list[tuple[str, int]] = []  # (kind, length) for every persistent operation so far
        self._target: int | None = None
        self._mode = "after"
        self._torn_at: int | None = None

    @property
    def op_count(self) -> int:
        return len(self.log)

    def arm(self, ops_from_now: int, mode: str = "after", torn_at: int | None = None) -> None:
        """Fire on the persistent operation ``ops_from_now`` operations ahead (0 = the next one)."""
        if mode not in {"before", "after", "torn"}:
            raise ValueError(f"unknown fault mode {mode!r}")
        if mode == "torn" and torn_at is None:
            raise ValueError("torn faults need torn_at")
        if ops_from_now < 0:
            raise ValueError("ops_from_now must be >= 0")
        self._target = self.op_count + ops_from_now
        self._mode = mode
        self._torn_at = torn_at

    def disarm(self) -> None:
        self._target = None

    def step(self, kind: str, length: int) -> tuple[bool, str, int | None]:
        """Register one persistent operation. Returns ``(fire, mode, torn_at)``."""
        index = self.op_count
        self.log.append((kind, length))
        if self._target is not None and index == self._target:
            self._target = None
            return True, self._mode, self._torn_at
        return False, "after", None


class FaultyFlash(FlashStore):
    """Flash with injectable power loss, torn writes and a silently failing write path (HV disruption)."""

    def __init__(self, plan: FaultPlan) -> None:
        super().__init__()
        self.plan = plan
        self.hv_disrupted = False

    def write(self, key: str, value: bytes, *, torn_at: int | None = None) -> None:
        fire, mode, fault_tear = self.plan.step("flash", len(value))
        if fire and mode == "before":
            raise PowerLoss("power lost before flash write")
        if not self.hv_disrupted:
            tear = fault_tear if (fire and mode == "torn") else torn_at
            super().write(key, value, torn_at=tear)
        if fire:
            raise PowerLoss(f"power lost at flash write ({mode})")


class FaultyFuses(OtpFuseBank):
    """OTP fuse bank with injectable power loss around a burn (a single burn is atomic here)."""

    def __init__(self, n_bits: int, plan: FaultPlan) -> None:
        super().__init__(n_bits)
        self.plan = plan

    def burn(self, i: int, *, verify: bool = False) -> None:
        fire, mode, _ = self.plan.step("fuse", 1)
        if fire and mode in {"before", "torn"}:
            raise PowerLoss("power lost before fuse burn")
        super().burn(i, verify=verify)
        if fire:
            raise PowerLoss("power lost after fuse burn")


class ChipNvm:
    """Everything a chip keeps across power cycles: flash, OTP fuses and one unary counter on the fuses."""

    def __init__(self, n_fuses: int = 512) -> None:
        self.plan = FaultPlan()
        self.flash = FaultyFlash(self.plan)
        self.fuses = FaultyFuses(n_fuses, self.plan)
        self.otp = UnaryCounter(self.fuses)

    def set_hv_disrupted(self, disrupted: bool) -> None:
        self.flash.hv_disrupted = disrupted
        self.fuses.hv_disrupted = disrupted
