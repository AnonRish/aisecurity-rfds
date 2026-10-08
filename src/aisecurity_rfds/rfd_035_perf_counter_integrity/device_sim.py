"""End-to-end runs of the signed-readout protocol against firmware-level and physical attackers.

A device runs a dense-matmul workload for ``n_intervals`` intervals. Each interval the physical process increments
the counter registers by the true FLOPs and memory accesses, the trusted block signs a readout, and "firmware"
carries it to the verifier. Chip power is measured by a separate physical sensor the attacker cannot edit.

Outcomes distinguish who caught the attack: the protocol (signature, sequence, chain, plausibility), the block
itself (it refused a ticket), the power cross-check (counters accepted but inconsistent with watts), or nobody.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.signing import KeyPair

from .attacks import analytic_power_detection_probability
from .checks import power_z_from_deltas
from .model import CLASSES, FLOPS_PER_FLOP_INSTR, N_SM, P_IDLE, PEAK_ISSUE, WorkloadClass, generate_reading
from .readout import ReadoutBlock, ReadoutVerifier, ResetAuthority, ResetTicket, TicketRefused

DEVICE_ID = b"GPU-0001"
# Assumption-derived physical bounds on counter growth per interval (all SMs issuing at peak).
MAX_RATE = (int(N_SM * PEAK_ISSUE * FLOPS_PER_FLOP_INSTR), int(N_SM * PEAK_ISSUE))

SCENARIOS = (
    "honest",
    "legitimate-reset",
    "ticket-hoarded-then-used",
    "firmware-edits-counters",
    "firmware-replays-old-readout",
    "firmware-drops-a-readout",
    "firmware-forges-reset-ticket",
    "firmware-replays-used-ticket",
    "firmware-uses-expired-ticket",
    "glitch-clears-registers",
    "clock-gating-stall",
    "undercount-1pct",
    "undercount-5pct",
    "undercount-30pct",
)
_HONEST_LIKE = {"honest", "legitimate-reset", "ticket-hoarded-then-used"}


@dataclass(frozen=True)
class DeviceOutcome:
    scenario: str
    intervals_accepted: int
    first_rejection: str | None
    block_refusal: str | None
    power_z: float
    power_flagged: bool
    credited_flops: int
    true_flops: int  # over the intervals whose readout the verifier accepted
    verdict: str

    @property
    def hidden_fraction(self) -> float:
        return max(0.0, 1.0 - self.credited_flops / self.true_flops) if self.true_flops else 0.0


def simulate_device(
    scenario: str,
    *,
    n_intervals: int = 40,
    onset: int = 20,
    seed: int = 0,
    cls: WorkloadClass = CLASSES["dense-matmul"],
    sigma_rel: float = 0.02,
    z_crit: float = 3.72,
    undercount: float | None = None,
) -> DeviceOutcome:
    """Run one scenario. ``undercount`` overrides the fraction hidden by the ``undercount-*`` scenarios."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = make_rng(derive_seed("rfd035-device-workload", seed))
    authority = ResetAuthority(KeyPair.from_seed(sha256(b"rfd035 authority")))
    device_key = KeyPair.from_seed(sha256(b"rfd035 device key"))
    attacker_key = KeyPair.from_seed(sha256(b"rfd035 attacker"))
    block = ReadoutBlock(DEVICE_ID, device_key, authority.public_bytes)
    verifier = ReadoutVerifier(DEVICE_ID, device_key.public_bytes, authority.public_bytes, max_rate_per_tick=MAX_RATE)

    if scenario.startswith("undercount"):
        fraction = undercount if undercount is not None else {"undercount-1pct": 0.01, "undercount-5pct": 0.05, "undercount-30pct": 0.30}[scenario]
        block.set_scale(1.0 - fraction)

    ticket: ResetTicket | None = None
    used_ticket: ResetTicket | None = None
    if scenario == "ticket-hoarded-then-used":
        ticket = authority.issue(DEVICE_ID, 0, not_after_tick=10 * n_intervals)  # issued at the start, long expiry
    if scenario == "firmware-uses-expired-ticket":
        ticket = authority.issue(DEVICE_ID, 0, not_after_tick=5)

    delivered = []
    dF: list[float] = []
    dM: list[float] = []
    power: list[float] = []
    true_f_accepted = 0
    first_rejection: str | None = None
    block_refusal: str | None = None
    pending_drop = False

    for k in range(n_intervals):
        reading = generate_reading(cls, rng)
        f, m = int(reading.flops.sum()), int(reading.mem.sum())
        if scenario == "clock-gating-stall" and k == onset:
            block.set_stalled(True)
        block.advance(1, [f, m])
        if scenario == "glitch-clears-registers" and k == onset:
            block.glitch_clear_registers()

        to_deliver = None
        if scenario in ("legitimate-reset", "firmware-replays-used-ticket") and k == onset:
            used_ticket = authority.issue(DEVICE_ID, block.epoch, not_after_tick=block.ticks + 5)
            to_deliver = block.reset(used_ticket)
        elif scenario == "ticket-hoarded-then-used" and k == onset:
            used_ticket = ticket
            to_deliver = block.reset(ticket)  # type: ignore[arg-type]
        else:
            ro = block.read()
            if scenario == "firmware-edits-counters" and k >= onset:
                ro = replace(ro, counters=(int(ro.counters[0] * 0.7), ro.counters[1]))
            if scenario == "firmware-replays-old-readout" and k == onset:
                ro = delivered[-5]
            if scenario == "firmware-drops-a-readout" and k == onset:
                pending_drop = True
            to_deliver = None if pending_drop and k == onset else ro

        if scenario == "firmware-forges-reset-ticket" and k == onset:
            forged = ResetTicket(DEVICE_ID, block.epoch, 999, 10**9, b"")
            forged = replace(forged, signature=attacker_key.sign(forged.digest()))
            try:
                block.reset(forged)
            except TicketRefused as exc:
                block_refusal = exc.reason
        if scenario == "firmware-uses-expired-ticket" and k == onset:
            try:
                block.reset(ticket)  # type: ignore[arg-type]
            except TicketRefused as exc:
                block_refusal = exc.reason
        if scenario == "firmware-replays-used-ticket" and k == onset + 10:
            try:
                block.reset(used_ticket)  # type: ignore[arg-type]  # a legitimate reset used it at ``onset``; replay it
            except TicketRefused as exc:
                block_refusal = exc.reason

        if to_deliver is None:
            continue  # dropped
        before = list(verifier.credited)
        verdict = verifier.accept(to_deliver)
        if not verdict.ok:
            first_rejection = verdict.reason
            break
        delivered.append(to_deliver)
        dF.append(verifier.credited[0] - before[0])
        dM.append(verifier.credited[1] - before[1])
        power.append(reading.power_w)
        true_f_accepted += f

    z = power_z_from_deltas(dF, dM, power, sigma_rel) if power else 0.0
    flagged = abs(z) > z_crit
    if first_rejection:
        verdict_text = f"caught by protocol ({first_rejection})"
    elif block_refusal:
        verdict_text = f"refused by block ({block_refusal})"
    elif flagged:
        verdict_text = "caught by power cross-check"
    elif scenario in _HONEST_LIKE:
        verdict_text = "no attack: accepted, credit matches true work"
    else:
        verdict_text = "MISSED: accepted and power check silent"
    return DeviceOutcome(scenario, len(delivered), first_rejection, block_refusal, z, flagged, verifier.credited[0], true_f_accepted, verdict_text)


def run_all(seed: int = 0, **kwargs: object) -> list[DeviceOutcome]:
    return [simulate_device(s, seed=seed, **kwargs) for s in SCENARIOS]  # type: ignore[arg-type]


def power_miss_probability(h: float, n_intervals: int, cls: WorkloadClass = CLASSES["dense-matmul"], sigma_rel: float = 0.02, z_crit: float = 3.72) -> float:
    """Analytic probability that the whole-run power test stays silent against a constant under-count ``h``."""
    r = generate_reading(cls, make_rng(0), power_noise=0.0)
    return 1.0 - analytic_power_detection_probability(h, r.power_w - P_IDLE, r.power_w, sigma_rel, z_crit, n=n_intervals)
