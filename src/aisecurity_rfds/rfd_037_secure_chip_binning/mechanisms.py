"""The RFD's lockout mechanisms as small executable chip models, and the attackers that try to undo them.

Each class models one way of keeping a feature off on a chip sold at a lower tier. An ``Attacker`` is a budget,
not a technique: everyone has privileged software on the host, and may additionally have fault injections, circuit
edits, a leaked token or secret, or a way to read the chip's PUF. ``attack`` applies the best use of that budget
to a fresh chip, and the chip then says whether the feature works and what its own report claims.

Assumptions that decide the results (they are modelling choices, not measurements of any process):

* A *fault injection* (``glitches``) corrupts one read of one decision in one boot, transiently. A *circuit edit*
  (``edits``) is a permanent focused-ion-beam change: it can reconnect a blown fuse, reconnect a cut power rail,
  force a decision line, or force the chip's report. Counts are budgets, not costs in money or time.
* A chip's report is derived from the same state that gates the feature, so a bypass that does not also edit the
  report path shows up in it ("visible"). Hiding it takes one more edit ("silent"). A report forged with an
  extracted device key is out of scope (see RFDs 026, 028 and 036).
* Tokens are Ed25519 signatures checked against a public key held in fuses. Software alone cannot forge one.
* The PUF-bound chip gates the feature on an on-chip comparator against a key reproduced from its PUF with public
  helper data. A design that used the key to decrypt the feature's configuration would not fall to a forced
  comparator; this model does not represent that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from aisecurity_rfds.common.hashing import domain_hash, u64
from aisecurity_rfds.common.nvm import OtpFuseBank, UnaryCounter
from aisecurity_rfds.common.puf import FuzzyExtractor, SimulatedPuf
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.signing import KeyPair, verify

FEATURE = b"restricted-compute-blocks"
BASE_FEATURE = b"base-compute-blocks"


@dataclass(frozen=True)
class Attacker:
    """An attack budget. Privileged software on the host is always assumed."""

    glitches: int = 0
    edits: int = 0
    other_device_token: bool = False
    old_token: bool = False
    manufacturer_secrets: bool = False
    puf_readout: bool = False

    def __post_init__(self) -> None:
        if self.glitches < 0 or self.edits < 0:
            raise ValueError("budgets cannot be negative")


class Outcome(Enum):
    """What the attacker achieved on one chip."""

    BLOCKED = "blocked"
    VISIBLE = "visible"
    SILENT = "silent"


class Lockout(Protocol):
    name: str

    def attack(self, attacker: Attacker) -> None: ...

    def feature_works(self) -> bool: ...

    def reports_disabled(self) -> bool: ...


def force_decision(attacker: Attacker, reads: int) -> tuple[bool, int]:
    """Can the attacker flip a majority-voted decision that is read ``reads`` times in one boot?

    Fault injections flip reads for free (in edits); each remaining read must be forced by a circuit edit. Returns
    whether the vote flips and how many edits are left over.
    """
    need = reads // 2 + 1
    via_edit = need - min(attacker.glitches, need)
    left = attacker.edits - via_edit
    return left >= 0, left


class FirmwareFlag:
    """A feature disabled by a flag that the driver or firmware owns (the RFD's tensor-core and clock rows)."""

    name = "firmware_flag"

    def __init__(self) -> None:
        self._flag = False
        self._spoof = False

    def attack(self, attacker: Attacker) -> None:
        self._flag = True
        self._spoof = True  # the report is produced by the same software, so it is forged too

    def feature_works(self) -> bool:
        return self._flag

    def reports_disabled(self) -> bool:
        return self._spoof or not self._flag


class FuseGate:
    """A feature disabled by ``copies`` burned fuses read at boot and majority-voted (the RFD's SM and memory rows).

    ``copies=1`` is the plain fuse blow; ``copies=3`` stands for "bypass-resistant fusing" (recommendation 3).
    """

    def __init__(self, copies: int) -> None:
        if copies < 1 or copies % 2 == 0:
            raise ValueError("copies must be a positive odd number")
        self.copies = copies
        self.name = f"fuse_x{copies}"
        self._bits = np.ones(copies, dtype=np.uint8)  # 1 = burned
        self._spoof = False
        self._disabled = True
        self._boot(0)

    def _boot(self, corrupted: int) -> None:
        reads = self._bits.copy()
        for i in np.flatnonzero(reads == 1)[:corrupted]:
            reads[i] = 0
        self._disabled = int(reads.sum()) * 2 > self.copies

    def attack(self, attacker: Attacker) -> None:
        ok, left = force_decision(attacker, self.copies)
        if not ok:
            return
        need = self.copies // 2 + 1
        glitched = min(attacker.glitches, need)
        for i in range(need - glitched):  # the rest of the vote is forced by reconnecting fuse links
            self._bits[i] = 0
        self._spoof = left >= 1
        self._boot(glitched)

    def reboot(self) -> None:
        """A clean boot: faults are gone, reconnected fuse links are not."""
        self._boot(0)

    def boot_with_random_faults(self, rng: np.random.Generator, p: float) -> bool:
        """One boot in which every read is independently corrupted with probability ``p``; True if the vote flips."""
        self._boot(int((rng.random(self.copies) < p).sum()))
        return not self._disabled

    def feature_works(self) -> bool:
        return not self._disabled

    def reports_disabled(self) -> bool:
        return self._disabled or self._spoof


class AntifuseRail:
    """Antifuses cut the feature's power rail; a self-test checks it stays dead (RFD mechanism 2).

    The self-test reads the feature ``selftest_reads`` times at boot. If any read sees it respond, the tamper
    response bricks the chip. Bridging the rail takes one edit; the self-test then has to be fooled, either by
    a second edit on its sense line or by corrupting every one of its reads.
    """

    def __init__(self, selftest_reads: int = 1) -> None:
        if selftest_reads < 1:
            raise ValueError("the self-test needs at least one read")
        self.selftest_reads = selftest_reads
        self.name = f"antifuse_rail_st{selftest_reads}"
        self._rail_cut = True
        self._sense_forced = False
        self._corrupted = 0
        self._bricked = False
        self._boot()

    def _boot(self) -> None:
        powered = not self._rail_cut
        sees_response = [powered and not self._sense_forced and i >= self._corrupted for i in range(self.selftest_reads)]
        self._bricked = any(sees_response)

    def attack(self, attacker: Attacker) -> None:
        if attacker.edits < 1:
            return
        self._rail_cut = False
        if attacker.edits >= 2:
            self._sense_forced = True
        else:
            self._corrupted = attacker.glitches
        self._boot()

    @property
    def bricked(self) -> bool:
        """True once the tamper response has fired."""
        return self._bricked

    def feature_works(self) -> bool:
        return not self._rail_cut and not self._bricked

    def reports_disabled(self) -> bool:
        return not self._bricked


@dataclass(frozen=True)
class Token:
    epoch: int
    signature: bytes


class Authority:
    """The manufacturer's token-signing service."""

    def __init__(self, label: str = "rfd037-authority") -> None:
        self._key = KeyPair.from_seed(domain_hash("rfd037.authority", label.encode()))
        self.public_key = self._key.public_bytes

    @staticmethod
    def message(variant: str, feature: bytes, device_id: bytes, epoch: int) -> bytes:
        if variant == "unbound":
            return domain_hash("rfd037.token.unbound", feature)
        if variant == "bound":
            return domain_hash("rfd037.token.bound", feature, device_id)
        if variant == "epoch":
            return domain_hash("rfd037.token.epoch", feature, device_id, u64(epoch))
        raise ValueError(f"unknown token variant {variant!r}")

    def mint(self, variant: str, feature: bytes, device_id: bytes, epoch: int) -> Token:
        return Token(epoch, self._key.sign(self.message(variant, feature, device_id, epoch)))


class TokenGate:
    """A feature enabled only by a signed activation token (RFD mechanism 3), in three strengths.

    ``unbound``: the token names only the feature, so one token enables every chip. ``bound``: it names the
    feature and the chip. ``epoch``: it also names an epoch, and the chip refuses tokens older than the epoch
    counter in its fuses, which a downgrade advances. The chip starts disabled after an earlier downgrade, so an
    attacker may hold the token that was valid before it.
    """

    NAMES = {"unbound": "token_unbound", "bound": "token_bound", "epoch": "token_bound_epoch"}

    def __init__(
        self, variant: str, authority: Authority | None = None, device_id: bytes = b"chip-A", decision_reads: int = 1
    ) -> None:
        self.variant = variant
        self.name = self.NAMES[variant]
        self.device_id = device_id
        self.decision_reads = decision_reads
        self._authority = authority or Authority()
        self._public_key = self._authority.public_key  # fused
        self._bank = OtpFuseBank(8)
        self._epoch = UnaryCounter(self._bank)
        self._old_token = self._authority.mint(variant, FEATURE, device_id, 0)
        if variant == "epoch":
            self._epoch.increment()  # the earlier downgrade
        self._enabled = False
        self._spoof = False

    def accepts(self, token: Token) -> bool:
        message = Authority.message(self.variant, FEATURE, self.device_id, token.epoch)
        if not verify(self._public_key, message, token.signature):
            return False
        return not (self.variant == "epoch" and token.epoch < self._epoch.value())

    def attack(self, attacker: Attacker) -> None:
        offered = [Token(0, bytes(64))]  # software alone can only offer a forgery
        if attacker.other_device_token:
            offered.append(self._authority.mint(self.variant, FEATURE, b"chip-B", self._epoch.value()))
        if attacker.old_token:
            offered.append(self._old_token)
        if attacker.manufacturer_secrets:
            offered.append(self._authority.mint(self.variant, FEATURE, self.device_id, self._epoch.value()))
        if any(self.accepts(t) for t in offered):
            self._enabled = True
            self._spoof = attacker.edits >= 1
            return
        ok, left = force_decision(attacker, self.decision_reads)
        if ok:
            self._enabled = True
            self._spoof = left >= 1

    def feature_works(self) -> bool:
        return self._enabled

    def reports_disabled(self) -> bool:
        return self._spoof or not self._enabled


class PufBound:
    """A feature enabled only by a key derived from the chip's PUF and the feature id (RFD mechanism 4).

    Public helper data for every feature sits on the chip; the key is reproduced inside the chip and compared
    with what is presented. The manufacturer's database holds keys only for purchased features, so a leaked
    database or signing key opens nothing else. Reading the PUF response from the silicon is the way in.
    """

    name = "puf_bound"

    def __init__(self, seed: int = 7, repetition: int = 15, decision_reads: int = 1) -> None:
        self._fe = FuzzyExtractor(128, repetition)
        n = self._fe.n_cells_required
        self._n = n
        self.decision_reads = decision_reads
        self._puf = SimulatedPuf(derive_seed("rfd037.puf-bound", seed), n_cells=2 * n)
        rng = make_rng(derive_seed("rfd037.puf-enrol", seed))
        raw = self._puf.reference_response()
        self._helpers = {}
        self.database: dict[bytes, bytes] = {}
        for slot, feature in enumerate((BASE_FEATURE, FEATURE)):
            key, helper = self._fe.enroll(raw[slot * n : (slot + 1) * n], rng=rng, context=feature)
            self._helpers[feature] = helper
            if feature == BASE_FEATURE:
                self.database[feature] = key  # purchased
        self._enabled = False
        self._spoof = False

    def _chip_key(self, feature: bytes) -> bytes:
        slot = 0 if feature == BASE_FEATURE else 1
        raw = self._puf.measure(rng=make_rng(derive_seed("rfd037.puf-measure", slot)))
        key, _ = self._fe.reproduce(raw[slot * self._n : (slot + 1) * self._n], self._helpers[feature], context=feature)
        return key

    def accepts(self, feature: bytes, presented: bytes | None) -> bool:
        return presented is not None and presented == self._chip_key(feature)

    def invasive_key(self, feature: bytes) -> bytes:
        """What an attacker who can read this chip's PUF response computes with the public helper data."""
        slot = 0 if feature == BASE_FEATURE else 1
        raw = self._puf.reference_response()
        key, _ = self._fe.reproduce(raw[slot * self._n : (slot + 1) * self._n], self._helpers[feature], context=feature)
        return key

    def attack(self, attacker: Attacker) -> None:
        offered: list[bytes | None] = [bytes(16)]  # software alone can only guess
        if attacker.manufacturer_secrets:
            offered.append(self.database.get(FEATURE))
        if attacker.puf_readout:
            offered.append(self.invasive_key(FEATURE))
        if any(self.accepts(FEATURE, key) for key in offered):
            self._enabled = True
            self._spoof = attacker.edits >= 1
            return
        ok, left = force_decision(attacker, self.decision_reads)
        if ok:
            self._enabled = True
            self._spoof = left >= 1

    def feature_works(self) -> bool:
        return self._enabled

    def reports_disabled(self) -> bool:
        return self._spoof or not self._enabled


class SeparateDie:
    """The feature's circuits are not on this die (RFD mechanism 1). No attack budget adds circuits."""

    name = "separate_die"

    def __init__(self, blocks_present: int = 114, blocks_needed: int = 132) -> None:
        self.blocks_present = blocks_present
        self.blocks_needed = blocks_needed

    def attack(self, attacker: Attacker) -> None:
        return None

    def feature_works(self) -> bool:
        return self.blocks_present >= self.blocks_needed

    def reports_disabled(self) -> bool:
        return True  # nothing on the die can say otherwise


def evaluate(factory, attacker: Attacker) -> Outcome:
    """Run ``attacker`` against a fresh chip from ``factory`` and classify the result."""
    chip: Lockout = factory()
    chip.attack(attacker)
    if not chip.feature_works():
        return Outcome.BLOCKED
    return Outcome.SILENT if chip.reports_disabled() else Outcome.VISIBLE
