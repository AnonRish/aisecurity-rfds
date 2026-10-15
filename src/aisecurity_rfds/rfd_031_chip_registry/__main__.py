"""Prints every number the README quotes: ``python -m aisecurity_rfds.rfd_031_chip_registry``."""

from __future__ import annotations

import math
from collections.abc import Sequence

from aisecurity_rfds.common.hashing import domain_hash
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.stats import clopper_pearson

from . import attacks, identity, location, witness
from .registry import ENROLL, make_request, sign_request, verify_history, verify_query
from .scenario import World

CONDITIONS = {
    "25C new": (25.0, 0.0),
    "85C 5y": (85.0, 5.0),
    "125C 20y": (125.0, 20.0),
    "150C 40y": (150.0, 40.0),
}


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(line(row))
    print()


def sci(x: float) -> str:
    return f"{x:.1e}"


def section_identifiers() -> None:
    print("== 1. Chip identifiers (technical requirement 1, open question 1) ==")
    agree = all(
        abs(identity.clone_detection_by_enumeration(m, q, careful) - identity.clone_detection_probability(m, q, careful)) < 1e-12
        for m in range(1, 7)
        for q in (0.1, 0.5, 0.9)
        for careful in (False, True)
    )
    print(
        "Copy detection by the attestation counter: each attestation the registry sees is the copy's with probability q."
        " A copy that starts level with the original is caught at the first switch between the two;"
        " a copy that starts ahead is caught only when the original attests after it."
    )
    print(f"The closed forms agree to 1e-12 with every sequence of up to 6 attestations run through real chips: {agree}.")
    rows = []
    for events in (2, 5, 10, 20):
        for q in (0.05, 0.2, 0.5):
            rows.append(
                [
                    str(events),
                    f"{q:.2f}",
                    f"{identity.clone_detection_probability(events, q):.4f}",
                    f"{identity.clone_detection_probability(events, q, careful=True):.4f}",
                ]
            )
    table(["attestations seen", "share by the copy", "copy starts level", "copy starts ahead"], rows)

    devices, cells, accept = 2000, 1024, 0.25
    separation = identity.puf_separation(devices, cells, accept, CONDITIONS, seed=3)
    print(f"Raw PUF responses, {cells} cells, {devices} devices; a response within {accept:.0%} of the enrolled one is accepted.")
    table(
        ["condition", "mean intra-chip BER", "worst intra-chip BER", "false rejects", "mean inter-chip BER"],
        [
            [
                s.condition,
                f"{s.intra_mean:.4f}",
                f"{s.intra_worst:.4f}",
                f"{round(s.false_reject_rate * devices)} of {devices}",
                f"{s.inter_mean:.4f}",
            ]
            for s in separation
        ],
    )
    upper = clopper_pearson(0, devices)[1]
    print(f"With no false rejects in {devices}, the 95% upper bound on the false-reject rate is {upper:.4f}.")
    print(f"An unrelated chip is accepted with probability {sci(identity.puf_false_accept_probability(cells, accept))} ({cells} cells).")
    print(f"With 256 cells it is {sci(identity.puf_false_accept_probability(256, accept))}; with 128 cells {sci(identity.puf_false_accept_probability(128, accept))}.")
    print()

    print("A key derived from the PUF at every boot (code-offset extractor, 128-bit key): share of 200 attempts that attest.")
    rng = make_rng(derive_seed("rfd031.main.pufchip"))
    rows = []
    for repetition in (3, 5, 9, 15):
        chip = identity.PufChip(2, repetition=repetition)
        cells_used = 128 * repetition
        shares = []
        for temperature, age in CONDITIONS.values():
            ok = sum(chip.attest(b"challenge", rng, temperature, age) is not None for _ in range(200))
            shares.append(f"{ok / 200:.3f}")
        rows.append([str(repetition), str(cells_used), *shares])
    table(["repetition", "cells", *CONDITIONS], rows)

    n, radius, accept = 30, 0.012, 0.8
    rng = make_rng(derive_seed("rfd031.main.glitter"))
    print(f"Glitter tag: {n} flakes on the unit torus, match radius {radius}, accept when {accept:.0%} of flakes match.")
    print("Read sigma as the camera's noise for false rejects. Read it as a forger's placement error, with the registry")
    print("photograph in hand, and the forger passes with probability 1 minus the false-reject rate.")
    rows = []
    for sigma in (0.004, 0.005, 0.006, 0.008):
        frr, far = identity.glitter_error_rates(n, sigma, radius, accept)
        sim_frr, _ = identity.simulate_glitter(n, sigma, radius, accept, 3000, rng)
        rows.append(
            [f"{sigma:.3f}", sci(frr), f"{sim_frr.p:.4f}", f"[{sim_frr.lo:.4f}, {sim_frr.hi:.4f}]", f"{1.0 - frr:.4f}", sci(far)]
        )
    table(
        ["sigma", "false reject (model)", "false reject (3000 photos)", "95% interval", "forger passes (model)", "false accept (model)"],
        rows,
    )
    _, far = identity.glitter_error_rates(n, 0.005, radius, accept)
    tags = 10_000_000
    print(f"Expected confusable pairs among {tags:,} tags: {sci(tags * (tags - 1) / 2 * far)}.")
    print("Where false accepts are measurable the independence approximation is only approximate:")
    rows = []
    for flakes, sigma, rad, thr in ((8, 0.03, 0.12, 0.5), (10, 0.05, 0.15, 0.6), (12, 0.04, 0.14, 0.5)):
        _, far = identity.glitter_error_rates(flakes, sigma, rad, thr)
        _, sim_far = identity.simulate_glitter(flakes, sigma, rad, thr, 4000, rng)
        rows.append([str(flakes), f"{sigma:.2f}", f"{rad:.2f}", f"{thr:.1f}", f"{far:.4f}", f"{sim_far.p:.4f}"])
    table(["flakes", "sigma", "radius", "accept at", "false accept (model)", "false accept (4000 pairs)"], rows)


def section_protocol() -> None:
    print("== 2. The update protocol under attack (technical requirement 3, open questions 5 and 6) ==")
    results = attacks.protocol_attacks()
    table(["Attack", "Defence under test", "Outcome"], [[r.attack, r.defence, r.outcome] for r in results])
    rejected = sum(r.outcome.startswith("rejected: ") for r in results)
    alerted = sum(r.outcome.startswith("alert: ") for r in results)
    accepted = sum(r.outcome.startswith("accepted") for r in results)
    print(f"{len(results)} attacks: {rejected} rejected by a rule, {alerted} raised as an alert, {accepted} accepted.")
    print()

    w = World.new()
    chip = w.enroll("lifecycle", jurisdiction="CA")
    w.sell(chip, w.fab, w.alice, jurisdiction="US")
    w.relocate(chip, w.alice, "US", location=domain_hash("example", b"site"))
    w.sell(chip, w.alice, w.bob, jurisdiction="GB")
    w.retire(chip, w.bob, "destroyed", witnessed=True)
    names = {w.fab.public_bytes: "fab", w.alice.public_bytes: "alice", w.bob.public_bytes: "bob"}
    history = w.registry.history(chip.chip_id, w.clock + 1)
    current = w.registry.query(chip.chip_id, w.clock + 1)
    print("One chip's life, from its history (every row replayed by the client):")
    table(
        ["version", "request", "status", "owner", "jurisdiction"],
        [
            [
                str(item.entry.record_after.version),
                item.entry.signed.request.kind,
                item.entry.record_after.status,
                names[item.entry.record_after.owner],
                item.entry.record_after.jurisdiction,
            ]
            for item in history.items
        ],
    )
    verdict = verify_history(w.operator.public_bytes, w.policy, history, current)
    print(f"The client replays the {len(history.items)} entries and finds them consistent with the map: {verdict.ok}.")
    size = sum(32 * len(item.proof) for item in history.items) + current.proof.size_bytes()
    print(f"Proof material for that answer: {size} bytes ({len(history.items)} log proofs and one map proof).")
    print()


def section_governance() -> None:
    print("== 3. A dishonest operator, and who notices (open question 4) ==")
    results = attacks.operator_attacks()
    table(["Attack", "Defence under test", "Outcome"], [[r.attack, r.defence, r.outcome] for r in results])
    detected = sum(r.outcome.startswith("detected: ") for r in results)
    refused = sum(r.outcome.startswith("refused: ") for r in results)
    rejected = sum(r.outcome.startswith("rejected: ") for r in results)
    quorum = sum(r.defence == "witness quorum" for r in results)
    unnoticed = sum(r.outcome.startswith("accepted") for r in results)
    print(
        f"{len(results)} attacks: {detected} detected by an auditor, {refused} refused by a witness, {rejected} rejected by a client,"
        f" {quorum} fork attempts against a quorum, {unnoticed} that passed the check named."
    )
    print()

    operator_key, head_a, head_b, now = attacks.forked_heads()
    print("Witness quorums: n witnesses, t of which must cosign a head. Each row was checked by trying every way of")
    print("showing two conflicting heads to real witness objects, with every possible number of colluding witnesses.")
    rows = []
    for n, t in ((3, 1), (3, 2), (3, 3), (4, 2), (5, 3), (5, 4), (5, 5), (7, 4), (7, 5), (7, 6)):
        needed = witness.min_colluders_to_fork(n, t)
        agrees = all(
            witness.fork_by_enumeration(n, t, c, operator_key, head_a, head_b, now) == (c >= needed) for c in range(n + 1)
        )
        rows.append([str(n), str(t), str(needed), str(witness.liveness_margin(n, t)), "yes" if agrees else "NO"])
    table(["witnesses", "threshold", "colluders needed to fork", "witnesses that may be offline", "matches enumeration"], rows)

    print("Proof sizes, measured on registries of different sizes (mean over 32 chips):")
    rows = []
    w = World.new()
    enrolled: list[bytes] = []
    for target in (128, 1024, 4096):
        while len(enrolled) < target:
            chip = identity.DeviceKeyChip(b"bulk-%d" % len(enrolled))
            request = make_request(ENROLL, chip.chip_id, 0, public_key=chip.public_key, jurisdiction="CA")
            w.registry.submit(sign_request(request, w.fab), w.tick(1))
            enrolled.append(chip.chip_id)
        picks = enrolled[:: target // 32]
        map_siblings, map_bytes, log_hashes = [], [], []
        for chip_id in picks:
            answer = w.registry.query(chip_id, w.clock)
            assert verify_query(w.operator.public_bytes, answer, now=w.clock, max_age=60).ok
            map_siblings.append(len(answer.proof.siblings))
            map_bytes.append(answer.proof.size_bytes())
            log_hashes.append(len(w.registry.history(chip_id, w.clock).items[0].proof))
        rows.append(
            [
                str(target),
                f"{sum(map_siblings) / 32:.1f}",
                f"{sum(map_bytes) / 32:.0f}",
                f"{sum(log_hashes) / 32:.1f}",
                f"{sum(log_hashes):.0f}",
            ]
        )
    table(["chips", "map siblings", "map proof bytes", "log hashes", "log proof bytes"], rows)
    print("Extrapolated, not measured: a proof carries about log2(n) hashes in each structure.")
    rows = []
    for chips in (10**6, 10**7, 10**9):
        hashes = math.ceil(math.log2(chips))
        rows.append([f"{chips:,}", str(hashes), str(32 * hashes)])
    table(["chips", "hashes per proof", "bytes per proof"], rows)


def section_location() -> None:
    print("== 4. Location and audits (open questions 2 and 3) ==")
    salt = bytes(range(16))
    commitment = location.commit_location("Facility-A", salt)
    ok = location.verify_opening(commitment, location.Opening("Facility-A", salt))
    other = location.verify_opening(commitment, location.Opening("Facility-B", salt))
    print(f"Opening a commitment with the facility and salt it was made with: {ok}; with another facility: {other}.")
    candidates = [f"Facility-{i:03d}" for i in range(100)]
    rows = []
    for salt_bytes in (0, 1, 2, 16):
        work = location.dictionary_work(len(candidates), salt_bytes)
        if salt_bytes <= 1:
            target = location.commit_location(candidates[70], bytes([0x5A]) * salt_bytes)
            found = location.guess_facility(target, candidates, salt_bytes)
            recovered = "yes" if found is not None and found.facility == candidates[70] else "no"
        else:
            recovered = "not tried"
        rows.append([str(salt_bytes), f"{work:,}" if work < 10**9 else sci(float(work)), recovered])
    print("Dictionary attack on a commitment, with 100 candidate facility names:")
    table(["salt bytes", "hash evaluations to exhaust", "facility recovered"], rows)

    facilities = [f"Facility-{i}" for i in range(10) for _ in range(100)]
    shared = location.distinct_commitments(facilities, [domain_hash("salt", f.encode()) for f in facilities])
    per_chip = location.distinct_commitments(facilities, [domain_hash("salt", f.encode(), bytes([i % 256, i // 256])) for i, f in enumerate(facilities)])
    print(f"What a registry reader sees for 1000 chips in 10 facilities: {shared} distinct commitments with a salt per facility, {per_chip} with a salt per chip.")

    total = 10_000
    print(f"Ghost-inventory audit: {total:,} declared chips, some not where declared; chips an inspector must challenge:")
    rows = []
    for ghosts in (1000, 500, 100, 10):
        needed = location.chips_to_inspect(total, ghosts, 0.95)
        rows.append(
            [
                str(ghosts),
                f"{100 * ghosts / total:.1f}%",
                str(needed),
                f"{location.detection_probability(total, ghosts, 58):.4f}",
            ]
        )
    table(["ghosts", "share", "chips to inspect for 95%", "P(detect) inspecting 58"], rows)
    rng = make_rng(derive_seed("rfd031.main.audit"))
    exact = location.detection_probability(500, 30, 20)
    sim = location.simulate_detection(500, 30, 20, 20_000, rng)
    print(f"Check by simulation: 500 declared, 30 ghosts, 20 inspected: model {exact:.4f}, simulated {sim.p:.4f} [{sim.lo:.4f}, {sim.hi:.4f}].")

    chips = [identity.DeviceKeyChip(b"audit-%d" % i) for i in range(200)]
    site = location.Facility(chips[12:])
    declared = [(c.chip_id, c.public_key) for c in chips]
    sample = sorted(int(i) for i in rng.choice(200, size=60, replace=False))
    failures = location.audit_declared(declared, site, sample, lambda chip_id: domain_hash("audit", chip_id))
    expected = [i for i in sample if i < 12]
    print(f"One audit with real attestations: 200 declared, the first 12 absent; 60 challenged; {len(failures)} failed.")
    print(f"The failures are exactly the absent chips in the sample: {failures == expected}.")


def main() -> None:
    section_identifiers()
    section_protocol()
    section_governance()
    section_location()


if __name__ == "__main__":
    main()
