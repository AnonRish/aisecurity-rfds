"""Attack, failure and trade-off scenarios for the replay guards (all on simulated hardware).

Every scenario is deterministic: licenses carry ids from a seeded generator, time is a counter, and
"power loss" is injected at numbered persistent operations rather than at wall-clock instants.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.signing import KeyPair
from aisecurity_rfds.common.signing import verify as ed25519_verify

from .guards import ChallengeGuard, Decision, HashLogGuard, HybridGuard, ReplayGuard, SequenceGuard, Verifier
from .hardware import ChipNvm, PowerLoss
from .ledger import CommitmentEmitter, ExternalLedger
from .licensing import License, LicenseAuthority

BUDGET = 10**15  # FLOPs granted per license: bookkeeping only, an assumption of the model
N_FUSES = 512
DEVICE_ID = b"CHIP-0001"
SEQ_A, SEQ_B = 511, 512  # crash scenario sequences chosen so a torn little-endian counter moves backwards


@dataclass(frozen=True)
class GuardConfig:
    label: str
    build: Callable[..., ReplayGuard]
    needs_challenge: bool = False


def hybrid_config(window: int, stride: int | None = None) -> GuardConfig:
    label = f"hybrid(W={window})" + (f"+otp(S={stride})" if stride else "")
    return GuardConfig(label, lambda n, d, p, r, **o: HybridGuard(n, d, p, window=window, floor_stride=stride, **o))


def sequence_config(stride: int | None = None) -> GuardConfig:
    label = "sequence" + (f"+otp(S={stride})" if stride else "")
    return GuardConfig(label, lambda n, d, p, r, **o: SequenceGuard(n, d, p, floor_stride=stride, **o))


CONFIGS: tuple[GuardConfig, ...] = (
    GuardConfig("hash-log", lambda n, d, p, r, **o: HashLogGuard(n, d, p, **o)),
    GuardConfig("hash-log/ring4", lambda n, d, p, r, **o: HashLogGuard(n, d, p, capacity=4, **o)),
    sequence_config(),
    sequence_config(4),
    hybrid_config(4),
    hybrid_config(4, 4),
    GuardConfig(
        "challenge", lambda n, d, p, r, **o: ChallengeGuard(n, d, p, rng=r, use_otp_epoch=False, **o), True
    ),
    GuardConfig(
        "challenge+otp-epoch", lambda n, d, p, r, **o: ChallengeGuard(n, d, p, rng=r, use_otp_epoch=True, **o), True
    ),
)
NAIVE_SEQUENCE = GuardConfig(
    "sequence/non-atomic", lambda n, d, p, r, **o: SequenceGuard(n, d, p, atomic=False, **o)
)


def default_authority(seed: int = 0) -> LicenseAuthority:
    key = KeyPair.from_seed(sha256(b"rfd034 license authority"))
    return LicenseAuthority(key, make_rng(derive_seed("rfd034", "authority", seed)))


class Rig:
    """One simulated chip, its license authority, and a running total of budget actually credited."""

    def __init__(
        self,
        cfg: GuardConfig,
        *,
        verify_after_write: bool = True,
        verifier: Verifier | None = None,
        authority: LicenseAuthority | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.nvm = ChipNvm(N_FUSES)
        self.authority = authority if authority is not None else default_authority(seed)
        self._options: dict[str, object] = {"verify_after_write": verify_after_write}
        if verifier is not None:
            self._options["verifier"] = verifier
        self._boot_count = 0
        self._rng_seed = derive_seed("rfd034", "chip-rng", seed)
        self.guard = self._build(self._rng_seed)
        self.next_sequence = 1
        self.total_credited = 0

    def _build(self, rng_seed: int) -> ReplayGuard:
        return self.cfg.build(self.nvm, DEVICE_ID, self.authority.public_key, make_rng(rng_seed), **self._options)

    def reboot(self, *, rng_restarts: bool = False) -> None:
        """Power cycle: volatile state is lost and the guard is rebuilt from non-volatile memory only."""
        self._boot_count += 1
        seed = self._rng_seed if rng_restarts else derive_seed("rfd034", "chip-rng-boot", self._boot_count)
        self.guard = self._build(seed)
        self.guard.boot()

    def issue(self, *, budget: int = BUDGET) -> License:
        """Authority issues the next license (challenge guards first ask the chip for a fresh challenge)."""
        if self.cfg.needs_challenge:
            if not isinstance(self.guard, ChallengeGuard):
                raise TypeError("challenge configs need a ChallengeGuard")
            challenge = self.guard.request_challenge()
            return self.authority.issue(DEVICE_ID, sequence=0, budget_flops=budget, challenge=challenge)
        lic = self.authority.issue(DEVICE_ID, sequence=self.next_sequence, budget_flops=budget)
        self.next_sequence += 1
        return lic

    def apply(self, lic: License) -> Decision:
        decision = self.guard.process(lic)
        self.total_credited += decision.budget_credited
        return decision


@dataclass(frozen=True)
class Outcome:
    succeeded: bool  # did the attacker obtain budget (or a capability) the authority did not grant?
    detail: str = ""

    def cell(self) -> str:
        if not self.succeeded:
            return "blocked"
        return f"BYPASSED {self.detail}".strip()


def _apply_all(rig: Rig, licenses: Sequence[License]) -> int:
    return sum(rig.apply(lic).accepted for lic in licenses)


def attack_plain_replay(cfg: GuardConfig) -> Outcome:
    rig = Rig(cfg)
    lic = rig.issue()
    rig.apply(lic)
    again = rig.apply(lic)
    return Outcome(again.accepted, "second use accepted" if again.accepted else again.reason)


def attack_replay_after_reboot(cfg: GuardConfig) -> Outcome:
    rig = Rig(cfg)
    lic = rig.issue()
    rig.apply(lic)
    rig.reboot()
    again = rig.apply(lic)
    return Outcome(again.accepted, "accepted after power cycle" if again.accepted else again.reason)


def attack_flash_rollback(cfg: GuardConfig, n: int = 10) -> Outcome:
    """Image the flash, use ``n`` licenses legitimately, restore the image, replay them all."""
    rig = Rig(cfg)
    first = rig.issue()
    snapshot = rig.nvm.flash.snapshot()
    licenses = [first]
    rig.apply(first)
    for _ in range(n - 1):
        lic = rig.issue()
        licenses.append(lic)
        rig.apply(lic)
    if rig.total_credited != n * BUDGET:
        raise RuntimeError("scenario bug: legitimate licenses were not all accepted")
    rig.nvm.flash.physical_restore(snapshot)
    rig.reboot()
    replayed = _apply_all(rig, licenses)
    return Outcome(replayed > 0, f"{replayed}/{n} replayed")


def attack_hv_disruption(cfg: GuardConfig, verify_after_write: bool) -> Outcome:
    """Starve the programming supply while a license is applied, restore it, apply the license again."""
    rig = Rig(cfg, verify_after_write=verify_after_write)
    lic = rig.issue()
    rig.nvm.set_hv_disrupted(True)
    rig.apply(lic)
    rig.nvm.set_hv_disrupted(False)
    rig.apply(lic)
    times = rig.total_credited // BUDGET
    return Outcome(times > 1, f"credited {times}x")


def attack_forged_state_poisoning(cfg: GuardConfig) -> Outcome:
    """An unsigned license with a huge sequence number tries to brick the counter for honest licenses."""
    rig = Rig(cfg)
    honest = rig.issue()
    forged = License(DEVICE_ID, 2**40, bytes(16), BUDGET, b"", bytes(64))
    rig.apply(forged)
    honest_ok = rig.apply(honest).accepted
    return Outcome(not honest_ok, "honest license rejected after forged one")


def attack_ring_eviction(cfg: GuardConfig, n: int = 6) -> Outcome:
    """Use ``n`` licenses, then replay the oldest one (relevant to bounded hash logs)."""
    rig = Rig(cfg)
    licenses = []
    for _ in range(n):
        lic = rig.issue()
        licenses.append(lic)
        rig.apply(lic)
    again = rig.apply(licenses[0])
    return Outcome(again.accepted, "oldest license accepted again" if again.accepted else again.reason)


def attack_stockpile(cfg: GuardConfig, n: int = 5) -> Outcome:
    """Collect ``n`` licenses first, use them all later (e.g. after a policy change)."""
    rig = Rig(cfg)
    if cfg.needs_challenge:
        held = []
        for _ in range(n):
            lic = rig.issue()
            held.append((lic, rig.nvm.flash.snapshot()))  # attacker images the flash holding this challenge
        accepted = 0
        for lic, image in held:
            rig.nvm.flash.physical_restore(image)
            accepted += rig.apply(lic).accepted
    else:
        accepted = _apply_all(rig, [rig.issue() for _ in range(n)])
    return Outcome(accepted > 1, f"{accepted}/{n} stockpiled licenses usable")


ATTACKS: tuple[tuple[str, Callable[[GuardConfig], Outcome]], ...] = (
    ("plain replay", attack_plain_replay),
    ("replay after power cycle", attack_replay_after_reboot),
    ("flash rollback, replay 10", attack_flash_rollback),
    ("HV disruption, no read-back", lambda c: attack_hv_disruption(c, False)),
    ("HV disruption, read-back", lambda c: attack_hv_disruption(c, True)),
    ("forged huge sequence", attack_forged_state_poisoning),
    ("oldest of 6 after eviction", attack_ring_eviction),
    ("stockpile 5 licenses", attack_stockpile),
)


def attack_matrix(configs: Sequence[GuardConfig] = CONFIGS) -> list[tuple[str, dict[str, Outcome]]]:
    return [(name, {c.label: fn(c) for c in configs}) for name, fn in ATTACKS]


def liveness_out_of_order(cfg: GuardConfig) -> int | None:
    """How many of three in-flight licenses are usable when they arrive as 3, 1, 2 (None: not applicable)."""
    if cfg.needs_challenge:
        return None
    rig = Rig(cfg)
    l1, l2, l3 = (rig.issue() for _ in range(3))
    return _apply_all(rig, [l3, l1, l2])


# ------------------------------------------------------------------ exhaustive power-loss enumeration


def _crash_rig(cfg: GuardConfig) -> tuple[Rig, License, License]:
    rig = Rig(cfg)
    if cfg.needs_challenge:
        la = rig.issue()
        rig.apply(la)
        lb = rig.issue()
    else:
        la = rig.authority.issue(DEVICE_ID, sequence=SEQ_A, budget_flops=BUDGET)
        rig.apply(la)
        lb = rig.authority.issue(DEVICE_ID, sequence=SEQ_B, budget_flops=BUDGET)
    return rig, la, lb


def crash_points(cfg: GuardConfig) -> list[tuple[int, str, int | None]]:
    """Every (operation index, mode, torn byte count) at which applying the second license can be interrupted."""
    rig, _, lb = _crash_rig(cfg)
    start = rig.nvm.plan.op_count
    rig.apply(lb)
    points: list[tuple[int, str, int | None]] = []
    for k, (kind, length) in enumerate(rig.nvm.plan.log[start:]):
        points += [(k, "before", None), (k, "after", None)]
        if kind == "flash":
            points += [(k, "torn", t) for t in range(1, length)]
    return points


def run_crash_point(cfg: GuardConfig, point: tuple[int, str, int | None]) -> tuple[bool, bool]:
    """Cut power at ``point`` while applying the second license, reboot, retry. Returns (violation, lost)."""
    rig, la, lb = _crash_rig(cfg)
    k, mode, torn_at = point
    rig.nvm.plan.arm(k, mode, torn_at)
    try:
        rig.apply(lb)
    except PowerLoss:
        pass
    else:
        raise RuntimeError("scenario bug: the injected power loss never fired")
    rig.nvm.plan.disarm()
    rig.reboot()
    rig.apply(la)  # an already-credited license must not be accepted again
    rig.apply(lb)  # retry the interrupted license ...
    rig.apply(lb)  # ... and once more: it must never be credited twice
    violation = rig.total_credited > 2 * BUDGET
    lost = rig.total_credited < 2 * BUDGET
    return violation, lost


@dataclass(frozen=True)
class CrashReport:
    label: str
    points: int
    violations: int
    lost: int

    @property
    def recoverable(self) -> int:
        return self.points - self.violations - self.lost


def crash_consistency(cfg: GuardConfig) -> CrashReport:
    points = crash_points(cfg)
    results = [run_crash_point(cfg, p) for p in points]
    return CrashReport(cfg.label, len(points), sum(v for v, _ in results), sum(lost for _, lost in results))


# ------------------------------------------------------------------ quantitative trade-offs


def storage_profile(cfg: GuardConfig, n: int) -> tuple[int, int, int]:
    """(logical payload bytes, physical flash bytes incl. journaling, fuses burned) after ``n`` licenses."""
    rig = Rig(cfg)
    for _ in range(n):
        rig.apply(rig.issue())
    flash_bytes = sum(len(v) for v in rig.nvm.flash.snapshot().values())
    return rig.guard.payload_bytes(), flash_bytes, rig.nvm.fuses.burned()


def worst_rollback_replay(cfg: GuardConfig, n: int = 32) -> tuple[int, int]:
    """Worst case, over every prefix length, of credited licenses replayable after a flash rollback.

    Returns ``(worst_replay_count, fuses_burned_after_n_licenses)``.
    """
    authority = default_authority()
    licenses = [authority.issue(DEVICE_ID, sequence=s, budget_flops=BUDGET) for s in range(1, n + 1)]
    worst, fuses_after_n = 0, 0
    for s in range(1, n + 1):
        rig = Rig(cfg, authority=authority)
        snapshot = rig.nvm.flash.snapshot()
        _apply_all(rig, licenses[:s])
        if s == n:
            fuses_after_n = rig.nvm.fuses.burned()
        rig.nvm.flash.physical_restore(snapshot)
        rig.reboot()
        worst = max(worst, _apply_all(rig, licenses[:s]))
    return worst, fuses_after_n


def mean_stranded_fraction(
    window: int | None, displacement: int, *, n: int = 60, trials: int = 10, seed: int = 5
) -> float:
    """Mean fraction of legitimate licenses rejected when delivery is shuffled by up to ``displacement`` places.

    ``window=None`` is the plain sequence guard; otherwise the hybrid guard with that window.
    """
    cfg = sequence_config() if window is None else hybrid_config(window)
    authority = default_authority()
    licenses = [authority.issue(DEVICE_ID, sequence=s, budget_flops=BUDGET) for s in range(1, n + 1)]
    rng = make_rng(derive_seed("rfd034", "reorder", seed, displacement))
    total = 0.0
    for _ in range(trials):
        order = np.argsort(np.arange(n) + rng.uniform(0, displacement, n)) if displacement else np.arange(n)
        rig = Rig(cfg, authority=authority)
        accepted = sum(rig.apply(licenses[int(i)]).accepted for i in order)
        total += 1 - accepted / n
    return total / trials


CONFIG_BY_LABEL: dict[str, GuardConfig] = {c.label: c for c in (*CONFIGS, NAIVE_SEQUENCE)}


def weak_rng_replay(use_otp_epoch: bool) -> bool:
    """A chip whose random-number generator restarts from the same state at every power-up.

    Apply a license, power-cycle (the RNG repeats), let the attacker trigger a new challenge request, then
    replay the first license. True if the replay is accepted.
    """
    rig = Rig(CONFIG_BY_LABEL["challenge+otp-epoch" if use_otp_epoch else "challenge"])
    lic = rig.issue()
    rig.apply(lic)
    rig.reboot(rng_restarts=True)
    rig.guard.request_challenge()  # type: ignore[attr-defined]
    return rig.apply(lic).accepted


def lenient_verifier(public: bytes, message: bytes, signature: bytes) -> bool:
    """A sloppy verifier that ignores bytes after the 64-byte signature (models non-canonical-encoding bugs)."""
    return ed25519_verify(public, message, signature[:64])


def padded_signature_replay(hash_scope: str, verifier: Verifier) -> bool:
    """Apply a license, then replay it with a padded (differently encoded) signature. True if the replay works."""
    cfg = GuardConfig("hash-log", lambda n, d, p, r, **o: HashLogGuard(n, d, p, hash_scope=hash_scope, **o))
    rig = Rig(cfg, verifier=verifier)
    lic = rig.issue()
    rig.apply(lic)
    return rig.apply(lic.with_signature(lic.signature + b"\x00")).accepted


def ledger_detection(careful: bool, *, every: int = 5) -> tuple[list[str], int]:
    """Rollback attack against the external-commitment option.

    The attacker images the flash, uses 10 licenses (commitments delivered), rolls back and replays them,
    then uses 5 more. A careless attacker lets the chip report the replay's commitments; a careful one
    withholds them. Returns ``(ledger anomalies, licenses credited in total)``.
    """
    rig = Rig(sequence_config())
    device_key = KeyPair.from_seed(sha256(b"rfd034 device key"))
    emitter = CommitmentEmitter(device_key, DEVICE_ID, every)
    ledger = ExternalLedger(DEVICE_ID, device_key.public_bytes)
    anomalies: list[str] = []

    def apply(lic: License, *, deliver: bool) -> None:
        if rig.apply(lic).accepted:
            commitment = emitter.on_accepted(rig.guard.last_sequence())  # type: ignore[attr-defined]
            if commitment is not None and deliver:
                anomalies.extend(ledger.submit(commitment))

    licenses = [rig.issue() for _ in range(15)]
    snapshot = rig.nvm.flash.snapshot()
    for lic in licenses[:10]:
        apply(lic, deliver=True)
    rig.nvm.flash.physical_restore(snapshot)
    rig.reboot()
    for lic in licenses[:10]:
        apply(lic, deliver=not careful)
    for lic in licenses[10:]:
        apply(lic, deliver=True)
    return anomalies, rig.total_credited // BUDGET
