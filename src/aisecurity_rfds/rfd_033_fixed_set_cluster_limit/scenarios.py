"""The RFD's attack table, run against the protocol code and reduced to verdicts.

Each row builds a small fleet, mounts the attack with exactly the capability named, and reports what happened.
Outcomes: ``blocked`` (the attacker gains nothing), ``succeeds`` (the attacker gets what it wanted: a documented
limitation, not a bug) or ``partial``. These are the rows printed by ``python -m ...`` and pinned by a test; the
unit tests in ``tests/`` probe the same mechanisms one at a time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

from aisecurity_rfds.common.rng import make_rng

from .cluster import build_fleet, establish
from .pod import HandshakeError, Hello1, Manufacturer, ProvisionError, x25519_generate


@dataclass(frozen=True)
class Verdict:
    attack: str
    defence: str
    outcome: str


def _refused(fn: Callable[[], object], exc: type[Exception] = ProvisionError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def outsider_claims_membership() -> Verdict:
    fleet = build_fleet(2, 3)
    victim, outsider = fleet.chips[0], fleet.chips[3]
    _, pub = x25519_generate()
    claim = Hello1(victim.pod_id, outsider.device_id, pub, b"n" * 16)  # type: ignore[arg-type]
    blocked = _refused(lambda: victim.respond(claim), HandshakeError) and not establish(outsider, victim)
    return Verdict("chip of another pod claims membership", "peer must be in the manifest", "blocked" if blocked else "succeeds")


def self_signed_manifest() -> Verdict:
    fleet = build_fleet(2, 3)
    forged = Manufacturer(b"attacker").sign_manifest(fleet.mfr.pod_id_for(9), 9, [c.device_id for c in fleet.chips])
    blocked = all(_refused(lambda c=c: c.provision(forged)) for c in fleet.chips)
    return Verdict("manifest signed by a key that is not the manufacturer's", "manufacturer key in fuses", "blocked" if blocked else "succeeds")


def spliced_manifest() -> Verdict:
    fleet = build_fleet(2, 3)
    blocked = all(_refused(lambda c=c: c.provision(fleet.manifests[0])) for c in fleet.chips[3:])
    return Verdict("pod A's manifest loaded into pod B's chips", "chip must be listed in the manifest", "blocked" if blocked else "succeeds")


def oversized_pod_on_consumer_chip() -> Verdict:
    fleet = build_fleet(2, 1)
    both = fleet.mfr.sign_manifest(fleet.mfr.pod_id_for(9), 9, [c.device_id for c in fleet.chips])
    blocked = all(_refused(lambda c=c: c.provision(both)) for c in fleet.chips)
    return Verdict("validly signed 2-chip pod offered to cap-1 chips", "size cap in fuses", "blocked" if blocked else "succeeds")


def replayed_manifest_after_update() -> Verdict:
    fleet = build_fleet(1, 3)
    chip, m1 = fleet.chips[0], fleet.manifests[0]
    chip.provision(fleet.mfr.sign_manifest(m1.pod_id, 2, list(m1.members)))
    blocked = _refused(lambda: chip.provision(m1))
    return Verdict("old manifest replayed after an update", "generation floor in fuses", "blocked" if blocked else "succeeds")


def flash_rollback_after_update() -> Verdict:
    fleet = build_fleet(1, 3)
    chip, m1 = fleet.chips[0], fleet.manifests[0]
    image = chip.flash.snapshot()
    chip.provision(fleet.mfr.sign_manifest(m1.pod_id, 2, list(m1.members)))
    chip.flash.physical_restore(image)
    chip.boot()
    return Verdict("flash image from before an update restored", "generation floor in fuses", "blocked" if not chip.is_member else "succeeds")


def flash_restore_after_tamper() -> Verdict:
    fleet = build_fleet(1, 3)
    chip = fleet.chips[0]
    image = chip.flash.snapshot()
    chip.tamper("mesh broken")
    chip.flash.physical_restore(image)
    chip.boot()
    return Verdict("flash image restored after a tamper response", "tamper burns a generation fuse", "blocked" if not chip.is_member else "succeeds")


def flash_restore_after_starved_tamper() -> Verdict:
    fleet = build_fleet(1, 3)
    chip = fleet.chips[0]
    image = chip.flash.snapshot()
    chip.fuses.hv_disrupted = True
    chip.tamper("mesh broken")
    chip.flash.physical_restore(image)
    chip.boot()
    return Verdict("burn supply starved, tamper response, flash restored", "none (the burn did not take)", "blocked" if not chip.is_member else "succeeds")


def replayed_handshake() -> Verdict:
    fleet = build_fleet(1, 3)
    a, b = fleet.chips[0], fleet.chips[1]
    h1 = a.begin()
    h2 = b.respond(h1)
    fin = a.complete(h2)
    b.finalize(fin)
    b.respond(h1)  # the attacker replays the first message
    blocked = _refused(lambda: b.finalize(fin), HandshakeError)
    return Verdict("recorded handshake replayed", "fresh nonces and ephemeral keys in a signed transcript", "blocked" if blocked else "succeeds")


def reflected_hello() -> Verdict:
    fleet = build_fleet(1, 3)
    a = fleet.chips[0]
    return Verdict("chip's own hello reflected back at it", "own identity refused, roles signed separately", "blocked" if _refused(lambda: a.respond(a.begin()), HandshakeError) else "succeeds")


def relabelled_initiator() -> Verdict:
    fleet = build_fleet(1, 3)
    a, b, c = fleet.chips
    h1 = a.begin()
    h2 = b.respond(dataclasses.replace(h1, device=c.device_id))
    return Verdict("initiator identity swapped in flight", "both identities in the signed transcript", "blocked" if _refused(lambda: a.complete(h2), HandshakeError) else "succeeds")


def replayed_release() -> Verdict:
    fleet = build_fleet(1, 3, bind_in_fuses=False)
    chip = fleet.chips[0]
    release = fleet.mfr.release_chip(chip)
    chip.release(release)
    return Verdict("release authorisation replayed", "generation floor in fuses", "blocked" if _refused(lambda: chip.release(release)) else "succeeds")


def removed_chip_and_stale_peer() -> Verdict:
    fleet = build_fleet(1, 4)
    a, b, c, x = fleet.chips
    m1 = fleet.manifests[0]
    a.provision(fleet.mfr.sign_manifest(m1.pod_id, 2, [a.device_id, b.device_id, c.device_id]))
    stale_accepts = establish(b, x)
    establish(a, b)  # gossip: b adopts the new manifest
    after = establish(b, x) or establish(x, b)
    outcome = "partial" if stale_accepts and not after else ("succeeds" if after else "blocked")
    return Verdict("removed chip talks to a member that has not updated", "gossip of the newer manifest", outcome)


def _blank_flash_then_move(bind_in_fuses: bool) -> bool:
    """Wipe a chip's flash (the attacker can image and restore it), then offer a signed manifest for another pod."""
    fleet = build_fleet(2, 2, bind_in_fuses=bind_in_fuses, seed=b"mv")
    chip = fleet.chips[0]
    chip.flash.physical_restore({})
    chip.boot()
    careless = fleet.mfr.sign_manifest(fleet.mfr.pod_id_for(7), 9, [chip.device_id, fleet.chips[2].device_id])
    return not _refused(lambda: chip.provision(careless))


def flash_wiped_then_moved_flash_bound() -> Verdict:
    moved = _blank_flash_then_move(False)
    return Verdict("flash wiped, then a manifest for another pod (flash-bound chip)", "none: the binding lived in flash", "succeeds" if moved else "blocked")


def flash_wiped_then_moved_fuse_bound() -> Verdict:
    moved = _blank_flash_then_move(True)
    return Verdict("flash wiped, then a manifest for another pod (fuse-bound chip)", "pod binding in fuses", "succeeds" if moved else "blocked")


def all_verdicts() -> list[Verdict]:
    return [
        outsider_claims_membership(),
        self_signed_manifest(),
        spliced_manifest(),
        oversized_pod_on_consumer_chip(),
        replayed_manifest_after_update(),
        flash_rollback_after_update(),
        flash_restore_after_tamper(),
        flash_restore_after_starved_tamper(),
        replayed_handshake(),
        reflected_hello(),
        relabelled_initiator(),
        replayed_release(),
        removed_chip_and_stale_peer(),
        flash_wiped_then_moved_flash_bound(),
        flash_wiped_then_moved_fuse_bound(),
    ]


@dataclass(frozen=True)
class StaleWindow:
    """How long a removed chip keeps links with members that have not heard of the update."""

    pod_size: int
    seed: int
    rounds_to_full_adoption: int
    exposed_per_round: tuple[int, ...]  # members whose manifest still lists the removed chip, before each round
    accepted_before: int  # members that complete a handshake with the removed chip right after the facility update
    accepted_after: int  # the same, once every member has adopted the new manifest


def stale_window(pod_size: int, seed: int, *, max_rounds: int = 200) -> StaleWindow:
    """Remove one chip from a pod by updating a single member, then let ordinary peer re-verification spread it.

    One round = every pod member (and the removed chip, which keeps trying) starts one handshake with a peer drawn
    uniformly from the peers in its own manifest, processed one after another. The only dissemination channel is
    the manifest carried in the handshake messages; the facility touches exactly one chip.
    """
    if pod_size < 3:
        raise ValueError("need at least two remaining members and one chip to remove")
    rng = make_rng(seed)
    fleet = build_fleet(1, pod_size, seed=b"gossip-%d-%d" % (pod_size, seed))
    chips = fleet.chips
    removed, kept = chips[-1], chips[:-1]
    m1 = fleet.manifests[0]
    kept[0].provision(fleet.mfr.sign_manifest(m1.pod_id, 2, [c.device_id for c in kept]))
    accepted_before = sum(establish(removed, c) for c in kept)
    exposed: list[int] = []
    rounds = 0
    while True:
        exposed_now = sum(removed.device_id in c.manifest.members for c in kept if c.manifest is not None)
        exposed.append(exposed_now)
        if exposed_now == 0 or rounds >= max_rounds:
            break
        for c in [*kept, removed]:
            known = [p for p in chips if p is not c and c.manifest is not None and p.device_id in c.manifest.members]
            if known:
                establish(c, known[int(rng.integers(len(known)))])
        rounds += 1
    accepted_after = sum(establish(removed, c) for c in kept)
    return StaleWindow(pod_size, seed, rounds, tuple(exposed), accepted_before, accepted_after)
