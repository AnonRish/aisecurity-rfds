"""Print the headline findings quoted in README.md. Deterministic; a few seconds of CPU."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from aisecurity_rfds.common.hashing import sha256
from aisecurity_rfds.common.rng import make_rng
from aisecurity_rfds.common.signing import KeyPair

from . import (
    ComputeLicense,
    HardwareProfile,
    LicenseAuthority,
    Meter,
    OfflineLicenseChip,
    SimParams,
    ThrottleAction,
    drawer_stockpile,
    revocation_outcome,
    simulate,
    throttle_rate,
)
from .policy_sim import POLICIES, checkpoint_interval_for_endurance, power_cycle_experiment
from .throttle import preset_jobs


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, rule, *body])


def _protocol_battery() -> list[tuple[str, str]]:
    authority = LicenseAuthority(KeyPair.from_seed(sha256(b"rfd032 authority")))
    serial = b"CHIP-A"

    def new_chip(serial_: bytes = serial, **kwargs: object) -> OfflineLicenseChip:
        return OfflineLicenseChip(serial_, authority.public_key, enforced=[Meter.FLOPS, Meter.MEM_BYTES], **kwargs)  # type: ignore[arg-type]

    both = {Meter.FLOPS: 100, Meter.MEM_BYTES: 50}
    good = authority.issue(serial, sequence=1, budgets=both)
    rows: list[tuple[str, str]] = []
    rows.append(("valid license on the intended chip", new_chip().apply_license(good).reason))
    rows.append(("same license on another chip", new_chip(b"CHIP-B").apply_license(good).reason))
    tampered = dataclasses.replace(good, budgets=(("flops", 10**9), ("mem_bytes", 50)))
    rows.append(("budget field edited after signing", new_chip().apply_license(tampered).reason))
    chip = new_chip()
    chip.apply_license(good)
    rows.append(("replay of the same license", chip.apply_license(good).reason))
    chip.apply_license(authority.issue(serial, sequence=3, budgets=both))
    rows.append(("older sequence number (2 after 3)", chip.apply_license(authority.issue(serial, sequence=2, budgets=both)).reason))
    rows.append(("meter the chip does not enforce", new_chip().apply_license(authority.issue(serial, sequence=1, budgets={Meter.CYCLES: 1})).reason))
    dup = authority.sign(ComputeLicense(serial, 1, "add", (("flops", 5), ("flops", 5))))
    rows.append(("duplicate meter entries (double count)", new_chip().apply_license(dup).reason))
    big = authority.issue(serial, sequence=1, budgets={Meter.FLOPS: 2**64 - 1})
    chip = new_chip()
    chip.apply_license(authority.issue(serial, sequence=1, budgets={Meter.FLOPS: 5}))
    rows.append(("top-up that overflows 2^64", chip.apply_license(authority.issue(serial, sequence=2, budgets={Meter.FLOPS: 2**64 - 1})).reason))
    rows.append(("max-size license on a fresh chip", new_chip().apply_license(big).reason))
    capped = new_chip(max_balance={Meter.FLOPS: 150})
    capped.apply_license(authority.issue(serial, sequence=1, budgets={Meter.FLOPS: 100}))
    rows.append(("top-up above the on-chip balance cap", capped.apply_license(authority.issue(serial, sequence=2, budgets={Meter.FLOPS: 100})).reason))
    return rows


def main() -> None:
    print("RFD 032 - Offline licensing. Protocol results use real Ed25519; throttle and policy numbers use ASSUMED parameters.\n")

    print("1. On-chip protocol: outcome of each license presented to a chip enforcing FLOPS and MEM_BYTES")
    print(_table(["presented license", "decision"], _protocol_battery()))

    profile = HardwareProfile()
    jobs = preset_jobs()
    print(f"\n2. Speed after exhaustion (fraction of full speed) for ASSUMED profile {profile}")
    rows = [[a.value, *(f"{throttle_rate(a, j, profile):.4f}" for j in jobs.values())] for a in ThrottleAction]
    print(_table(["throttle action", *jobs], rows))

    rng = make_rng(7)
    print("\n3. Power cycling: restored balance vs true balance (budget 10^6, 400 random steps, ~5% cycle chance each)")
    rows = []
    for interval in (10, 100, 1000):
        r = power_cycle_experiment(10**6, interval, rng)
        rows.append([interval, r.cycles, r.max_gain, r.max_loss, r.checkpoint_writes])
    print(_table(["checkpoint interval", "power cycles", "max gain", "max loss", "flash writes"], rows))
    interval = checkpoint_interval_for_endurance(10**18, 10**5)
    print(f"   Endurance arithmetic (per the RFD's 10^18-FLOP example; 10^5 writes is an ASSUMPTION): interval >= {interval:.0e} FLOPs,"
          f" i.e. at most {interval / 10**18:.0e} of the budget lost per power cycle.")

    print("\n4. Mid-job exhaustion policies (lognormal job sizes, sigma=0.5 ASSUMED; delay=10 job-times ASSUMED; 5000 jobs)")
    rows = []
    for budget in (1, 5, 20, 100):
        for policy in POLICIES:
            r = simulate(policy, SimParams(budget=budget), make_rng(1))
            rows.append([budget, policy, f"{100 * r.interrupted_fraction:.1f}", f"{100 * r.never_completed_fraction:.1f}",
                         f"{r.mean_slowdown:.3f}", f"{100 * r.wasted_fraction:.1f}", f"{1 + 10 / budget:.3f}"])
    print(_table(["budget", "policy", "interrupted %", "never done %", "slowdown", "wasted %", "1+D/B"], rows))

    print("\n5. Stockpiling: 5 pre-issued top-ups of 100 against a chip with a balance cap")
    rows = []
    for cap in (100, 300, 500):
        total, peak = drawer_stockpile(5, 100, cap)
        rows.append([cap, total, peak])
    print(_table(["balance cap", "total usable", "peak balance"], rows))
    print(f"   Revocation (set-to-0, sequence 6) applied first: {revocation_outcome(5, 100, True)} usable;"
          f" withheld by the attacker: {revocation_outcome(5, 100, False)} usable")

    print("\n6. License size: overhead (D/B) against drawer exposure if 3 licenses are pre-issued (D=10 job-times)")
    rows = [[b, f"{100 * 10 / b:.1f}", 3 * b] for b in (10, 50, 100, 500)]
    print(_table(["budget B (job-times)", "reissue overhead %", "drawer exposure 3B"], rows))


if __name__ == "__main__":
    main()
