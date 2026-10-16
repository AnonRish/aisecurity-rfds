"""Fleet-level experiments: which chips end up in one high-bandwidth component, and what an attacker can add?

Every edge below is produced by running the real handshake in ``pod.py`` between two chips; nothing is assumed
about who can talk to whom. The *cluster size* the RFD cares about is the number of genuine chips that are
connected by HIGH links, directly or through a device the attacker controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from aisecurity_rfds.common.signing import KeyPair

from .pod import Chip, HandshakeError, LinkMode, Manufacturer, PodManifest


def establish(a: Chip, b: Chip) -> bool:
    """Run the three-message handshake, ``a`` initiating. True iff both ends now see a HIGH link."""
    try:
        h1 = a.begin()
        h2 = b.respond(h1)
        fin = a.complete(h2)
        b.finalize(fin)
    except HandshakeError:
        return False
    return a.link_mode(b.device_id) is LinkMode.HIGH and b.link_mode(a.device_id) is LinkMode.HIGH


@dataclass
class Fleet:
    mfr: Manufacturer
    chips: list[Chip]
    pods: list[list[int]]
    manifests: list[PodManifest]


def build_fleet(
    n_pods: int, pod_size: int, *, pod_cap: int | None = None, bind_in_fuses: bool = True, seed: bytes = b"fleet"
) -> Fleet:
    """``n_pods`` pods of ``pod_size`` chips each, formed by the manufacturer's disciplined path."""
    mfr = Manufacturer(seed)
    cap = pod_size if pod_cap is None else pod_cap
    chips: list[Chip] = []
    pods: list[list[int]] = []
    manifests: list[PodManifest] = []
    for p in range(n_pods):
        group = [mfr.make_chip(pod_cap=cap, bind_in_fuses=bind_in_fuses) for _ in range(pod_size)]
        manifests.append(mfr.form_pod(group, label=p))
        pods.append(list(range(len(chips), len(chips) + pod_size)))
        chips.extend(group)
    return Fleet(mfr, chips, pods, manifests)


def hbw_edges(nodes: Sequence[Chip]) -> list[tuple[int, int]]:
    """All pairs (i < j) of ``nodes`` that can establish a HIGH link."""
    return [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if establish(nodes[i], nodes[j])]


def components(n: int, edges: Iterable[tuple[int, int]], same_entity: Iterable[Sequence[int]] = ()) -> list[set[int]]:
    """Connected components of an undirected graph; ``same_entity`` groups are fused (one physical device)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for a, b in edges:
        union(a, b)
    for group in same_entity:
        for other in group[1:]:
            union(group[0], other)
    out: dict[int, set[int]] = {}
    for i in range(n):
        out.setdefault(find(i), set()).add(i)
    return sorted(out.values(), key=lambda s: (-len(s), min(s)))


def largest_hbw_cluster(chips: Sequence[Chip]) -> int:
    """Size of the largest set of genuine chips joined by HIGH links (no attacker devices)."""
    comps = components(len(chips), hbw_edges(chips))
    return len(comps[0]) if comps else 0


def impersonate(mfr_public: bytes, stolen: KeyPair, victim_manifest: PodManifest, *, serial: int, pod_cap: int) -> Chip:
    """The attacker's emulator: software that holds a stolen device key and a copy of the victim pod's manifest.

    It needs nothing the manufacturer did not already hand to the victim chip: the manifest is signed, lists the
    stolen key, and the emulator's own fuses are whatever the attacker says they are.
    """
    emulator = Chip(stolen, mfr_public, serial=serial, pod_cap=pod_cap, bind_in_fuses=False)
    emulator.provision(victim_manifest)
    return emulator


def bridge_cluster_size(fleet: Fleet, extracted: Sequence[int], *, pod_cap: int | None = None) -> int:
    """Largest set of genuine chips joined once an attacker extracts the device key of each chip index in
    ``extracted`` and runs ONE bridge device that presents all of those identities.

    A HIGH link to any identity is a HIGH link to the bridge, so every pod that lost a key is joined to every other.
    """
    cap = pod_cap if pod_cap is not None else max(len(p) for p in fleet.pods)
    pod_of = {i: p for p, members in enumerate(fleet.pods) for i in members}
    emulators = [
        impersonate(
            fleet.mfr.public,
            fleet.chips[i].device,
            fleet.manifests[pod_of[i]],
            serial=10_000 + k,
            pod_cap=cap,
        )
        for k, i in enumerate(extracted)
    ]
    nodes = [*fleet.chips, *emulators]
    n_real = len(fleet.chips)
    edges = hbw_edges(nodes)
    comps = components(len(nodes), edges, same_entity=[list(range(n_real, len(nodes)))] if emulators else [])
    return max(len([i for i in c if i < n_real]) for c in comps)
