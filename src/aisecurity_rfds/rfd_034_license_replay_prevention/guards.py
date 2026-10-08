"""Replay guards: the three approaches of RFD 034 plus its hybrid, on the simulated non-volatile memories.

Common contract (``ReplayGuard.process``): device binding, then signature, then a read-only replay
check, then a durable commit with optional read-back, and only then is the budget reported as credited.

* Rejections never modify state, so unsigned licenses cannot poison the counters.
* Commit-before-credit means a power loss can *lose* a license (recorded as used, never credited) but can
  never credit it twice.
* Guards keep no state in Python attributes: everything is re-read from the NVM, so
  rebuilding a guard on the same ``ChipNvm`` is a faithful power cycle.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.hashing import ct_equal, domain_hash, u64
from aisecurity_rfds.common.nvm import JournaledRecord, UnaryCounter, WriteVerifyError
from aisecurity_rfds.common.signing import verify as ed25519_verify

from .hardware import ChipNvm
from .licensing import LICENSE_ID_BYTES, License

Verifier = Callable[[bytes, bytes, bytes], bool]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    budget_credited: int = 0


class StrideFloor:
    """Rollback floor in OTP fuses: counter value ``b`` means "a sequence number >= b * stride was accepted".

    Cost: one fuse per ``stride`` sequence numbers. Residual exposure: a flash rollback can replay at most
    the licenses above the last stride boundary (see ``scenarios.worst_rollback_replay``).
    """

    def __init__(self, counter: UnaryCounter, stride: int) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._counter = counter
        self.stride = stride

    def last_sequence_floor(self) -> int:
        return self._counter.value() * self.stride

    def can_reach(self, sequence: int) -> bool:
        return sequence // self.stride <= self._counter.capacity

    def advance_to(self, sequence: int, *, verify: bool) -> None:
        # A fixed number of burns (never "until it reads back"), so a disrupted supply cannot loop forever.
        for _ in range(sequence // self.stride - self._counter.value()):
            self._counter.increment(verify=verify)


class ReplayGuard(abc.ABC):
    name = "abstract"

    def __init__(
        self,
        nvm: ChipNvm,
        device_id: bytes,
        authority_public: bytes,
        *,
        verify_after_write: bool = True,
        verifier: Verifier = ed25519_verify,
    ) -> None:
        self.nvm = nvm
        self.device_id = device_id
        self.authority_public = authority_public
        self.verify_after_write = verify_after_write
        self._verifier = verifier

    def process(self, lic: License) -> Decision:
        if lic.device_id != self.device_id:
            return Decision(False, "wrong-device")
        if not self._verifier(self.authority_public, lic.content_digest(), lic.signature):
            return Decision(False, "bad-signature")
        reason = self._replay_reason(lic)
        if reason is not None:
            return Decision(False, reason)
        try:
            self._commit(lic)
        except WriteVerifyError:
            return Decision(False, "commit-failed")
        return Decision(True, "accepted", lic.budget_flops)

    @abc.abstractmethod
    def _replay_reason(self, lic: License) -> str | None:
        """Read-only: why this license must be rejected, or None."""

    @abc.abstractmethod
    def _commit(self, lic: License) -> None:
        """Durably record the license as used (raises WriteVerifyError if read-back fails)."""

    @abc.abstractmethod
    def payload_bytes(self) -> int:
        """Logical state size (what the RFD's "storage requirements" counts), excluding journaling."""

    def boot(self) -> None:
        """Reconcile redundant state after power-up. Default: nothing to do."""

    def _put(self, record: JournaledRecord, payload: bytes) -> None:
        record.write(payload)
        if self.verify_after_write and record.read() != payload:
            raise WriteVerifyError(f"{self.name}: record did not persist (read-back mismatch)")


class HashLogGuard(ReplayGuard):
    """Approach 1: remember a digest of every applied license (optionally in a bounded ring)."""

    def __init__(
        self, *args: object, capacity: int | None = None, hash_scope: str = "content", **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be >= 1")
        if hash_scope not in {"content", "full"}:
            raise ValueError("hash_scope must be 'content' or 'full'")
        self.capacity = capacity
        self.hash_scope = hash_scope
        self.name = "hash-log" + (f"/ring{capacity}" if capacity else "") + ("/full-digest" if hash_scope == "full" else "")
        self._count = JournaledRecord(self.nvm.flash, "hashlog.count")

    def _digest(self, lic: License) -> bytes:
        return lic.content_digest() if self.hash_scope == "content" else lic.full_digest()

    def _total(self) -> int:
        raw = self._count.read()
        return int.from_bytes(raw, "big") if raw else 0

    def _stored(self) -> int:
        total = self._total()
        return total if self.capacity is None else min(total, self.capacity)

    def _replay_reason(self, lic: License) -> str | None:
        digest = self._digest(lic)
        for slot in range(self._stored()):
            if self.nvm.flash.read(f"hashlog.entry.{slot}") == digest:
                return "replay"
        return None

    def _commit(self, lic: License) -> None:
        digest, total = self._digest(lic), self._total()
        slot = total if self.capacity is None else total % self.capacity
        key = f"hashlog.entry.{slot}"
        self.nvm.flash.write(key, digest)  # entry first, count second: a torn commit leaves an invisible entry
        if self.verify_after_write and self.nvm.flash.read(key) != digest:
            raise WriteVerifyError("hash-log entry did not persist")
        self._put(self._count, u64(total + 1))

    def payload_bytes(self) -> int:
        return 8 + 32 * self._stored()


class SequenceGuard(ReplayGuard):
    """Approach 2: accept only strictly increasing sequence numbers, with an optional OTP rollback floor.

    ``atomic=False`` is a deliberately naive negative control: a raw little-endian counter in flash, which
    a torn write can move *backwards*.
    """

    def __init__(
        self, *args: object, floor_stride: int | None = None, atomic: bool = True, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.floor = StrideFloor(self.nvm.otp, floor_stride) if floor_stride else None
        self.atomic = atomic
        self._rec = JournaledRecord(self.nvm.flash, "sequence")
        self.name = "sequence" + (f"+otp(S={floor_stride})" if floor_stride else "") + ("" if atomic else "/non-atomic")

    def flash_last(self) -> int:
        if self.atomic:
            raw = self._rec.read()
            return int.from_bytes(raw, "big") if raw else 0
        raw = self.nvm.flash.read("sequence.raw")
        return int.from_bytes(raw, "little") if raw else 0

    def last_sequence(self) -> int:
        """Effective high-water mark: the larger of the flash value and the OTP floor."""
        return max(self.flash_last(), self.floor.last_sequence_floor() if self.floor else 0)

    def _replay_reason(self, lic: License) -> str | None:
        if self.floor is not None and not self.floor.can_reach(lic.sequence):
            return "otp-exhausted"
        return "replay-or-stale" if lic.sequence <= self.last_sequence() else None

    def _commit(self, lic: License) -> None:
        if self.atomic:
            self._put(self._rec, u64(lic.sequence))
        else:
            self.nvm.flash.write("sequence.raw", lic.sequence.to_bytes(8, "little"))
        if self.floor is not None:
            self.floor.advance_to(lic.sequence, verify=self.verify_after_write)

    def boot(self) -> None:
        # Flash is written before the fuses, so a crash between the two leaves the floor behind: repair it.
        if self.floor is not None:
            self.floor.advance_to(self.last_sequence(), verify=True)

    def payload_bytes(self) -> int:
        return 8


class HybridGuard(ReplayGuard):
    """Hybrid of the RFD: sequence number for coarse ordering plus a bounded log of recent license ids.

    The log is pruned by *sequence number* (not arrival order), so each entry stores ``(sequence, id)``:
    24 bytes, not the 16 the RFD's pseudo-code suggests.
    """

    def __init__(self, *args: object, window: int, floor_stride: int | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if window < 0:
            raise ValueError("window must be >= 0")
        self.window = window
        self.floor = StrideFloor(self.nvm.otp, floor_stride) if floor_stride else None
        self._rec = JournaledRecord(self.nvm.flash, "hybrid")
        self.name = f"hybrid(W={window})" + (f"+otp(S={floor_stride})" if floor_stride else "")

    def _load(self) -> tuple[int, list[tuple[int, bytes]]]:
        raw = self._rec.read()
        if not raw:
            return 0, []
        step = 8 + LICENSE_ID_BYTES
        body = raw[8:]
        entries = [(int.from_bytes(body[i : i + 8], "big"), body[i + 8 : i + step]) for i in range(0, len(body), step)]
        return int.from_bytes(raw[:8], "big"), entries

    def log_entries(self) -> list[tuple[int, bytes]]:
        return self._load()[1]

    def _replay_reason(self, lic: License) -> str | None:
        last, entries = self._load()
        if self.floor is not None:
            if not self.floor.can_reach(lic.sequence):
                return "otp-exhausted"
            last = max(last, self.floor.last_sequence_floor())
        if lic.sequence + self.window < last:
            return "too-old"
        if any(license_id == lic.license_id for _, license_id in entries):
            return "replay"
        return None

    def _commit(self, lic: License) -> None:
        last, entries = self._load()
        new_last = max(last, lic.sequence)
        kept = sorted(e for e in [*entries, (lic.sequence, lic.license_id)] if e[0] + self.window >= new_last)
        self._put(self._rec, u64(new_last) + b"".join(u64(s) + i for s, i in kept))
        if self.floor is not None:
            self.floor.advance_to(new_last, verify=self.verify_after_write)

    def boot(self) -> None:
        if self.floor is not None:
            self.floor.advance_to(self._load()[0], verify=True)

    def payload_bytes(self) -> int:
        return len(self._rec.read() or b"")


class ChallengeGuard(ReplayGuard):
    """Approach 3: the chip issues a random challenge; a license must sign it; the challenge is single-use.

    ``use_otp_epoch`` binds each challenge to the OTP counter, which advances on every accepted license.
    That defeats flash rollback and a random-number generator that repeats after reset, at a cost of one
    fuse per license for the chip's lifetime.
    """

    def __init__(self, *args: object, rng: np.random.Generator, use_otp_epoch: bool = True, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._rng = rng
        self.use_otp_epoch = use_otp_epoch
        self._pending = JournaledRecord(self.nvm.flash, "challenge.pending")
        self.name = "challenge" + ("+otp-epoch" if use_otp_epoch else "")

    def _epoch(self) -> int:
        return self.nvm.otp.value() if self.use_otp_epoch else 0

    def _challenge_for(self, epoch: int, nonce: bytes) -> bytes:
        return domain_hash("rfd034/challenge/v1", self.device_id, u64(epoch), nonce)

    def request_challenge(self) -> bytes:
        """Generate, persist and return a challenge. Only one challenge is outstanding at a time."""
        epoch = self._epoch()
        nonce = bytes(self._rng.integers(0, 256, 16, dtype=np.uint8))
        self._put(self._pending, u64(epoch) + nonce)
        return self._challenge_for(epoch, nonce)

    def _replay_reason(self, lic: License) -> str | None:
        raw = self._pending.read()
        if not raw:
            return "no-pending-challenge"
        epoch, nonce = int.from_bytes(raw[:8], "big"), raw[8:]
        if self.use_otp_epoch:
            if epoch != self.nvm.otp.value():
                return "stale-epoch"
            if self.nvm.otp.value() >= self.nvm.otp.capacity:
                return "otp-exhausted"
        if not ct_equal(lic.challenge, self._challenge_for(epoch, nonce)):
            return "challenge-mismatch"
        return None

    def _commit(self, lic: License) -> None:
        self._put(self._pending, b"")
        if self.use_otp_epoch:
            self.nvm.otp.increment(verify=self.verify_after_write)

    def payload_bytes(self) -> int:
        return len(self._pending.read() or b"")
