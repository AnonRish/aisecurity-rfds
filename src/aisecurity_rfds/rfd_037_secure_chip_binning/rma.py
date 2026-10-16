"""A bounded model checker for tier changes after sale (RFD 037, open questions 3 and 5).

The world has two chips. Each has a physical cap (the highest tier its silicon can reach), an entitlement (what the
customer paid for), a registry record (what governance believes), the tier it currently runs at, an epoch counter
in fuses, and the set of activation tokens ever minted for it. The attacker is worst case: it holds every token
ever minted for any chip and may present any of them to any chip at any time.

Actions are legitimate sales and returns (``upgrade``, ``downgrade``), a replacement of the whole die (``swap``) and
the attacker's ``replay``. Breadth-first search over everything reachable within a depth bound reports the shortest
trace that breaks each invariant:

* I1, governance sees what runs: no chip runs above the tier the registry records for it.
* I2, the record is accurate: the registry equals the entitlement.

A design is four switches: whether dies are made per tier (``cap_follows_tier``), whether a token works only on the
chip it was minted for (``device_bound``), whether a downgrade advances the epoch so older tokens die
(``revocation``), and whether every sale and return updates the registry (``registry_sync``). A search that finds no
violation is a statement about this model within the depth bound, not a proof about hardware or about unbounded
histories. In the per-tier-die design a tier change is a swap, and the model assumes the replaced die is destroyed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

TIERS = (0, 1, 2)  # 0 low, 1 mid, 2 top (the restricted tier)
MAX_EPOCH = 2  # the epoch counter has room for two revocations


@dataclass(frozen=True)
class Chip:
    cap: int
    entitled: int
    registry: int
    active: int
    epoch: int
    issued: frozenset[tuple[int, int]]  # (tier, epoch) of every token minted for this chip


World = tuple[Chip, ...]


@dataclass(frozen=True)
class Design:
    name: str
    cap_follows_tier: bool
    device_bound: bool
    revocation: bool
    registry_sync: bool


DESIGNS = (
    Design("binned dies, bare tokens", False, False, False, True),
    Design("binned dies, device-bound tokens", False, True, False, True),
    Design("binned dies, bound tokens with revocation", False, True, True, True),
    Design("same, registry not updated on changes", False, True, True, False),
    Design("per-tier dies, bare tokens", True, False, False, True),
)


def fresh_chip(design: Design, tier: int) -> Chip:
    """A newly made chip sold at ``tier``: registered, running at that tier, with its token if it needs one."""
    cap = tier if design.cap_follows_tier else max(TIERS)
    issued = frozenset({(tier, 0)}) if tier > 0 else frozenset()
    return Chip(cap, tier, tier, tier, 0, issued)


def initial_world(design: Design) -> World:
    return (fresh_chip(design, 0), fresh_chip(design, 2))


def accepts(design: Design, target: Chip, target_index: int, owner_index: int, tier: int, epoch: int) -> bool:
    """Does chip ``target`` accept the token for ``tier`` at ``epoch`` that was minted for chip ``owner_index``?"""
    if tier > target.cap:
        return False
    if design.device_bound and owner_index != target_index:
        return False
    return not (design.revocation and epoch < target.epoch)


def successors(design: Design, world: World) -> Iterator[tuple[str, World]]:
    def put(index: int, chip: Chip) -> World:
        return world[:index] + (chip,) + world[index + 1 :]

    for i, chip in enumerate(world):
        for t in TIERS:
            if chip.entitled < t <= chip.cap:
                registry = t if design.registry_sync else chip.registry
                issued = chip.issued | {(t, chip.epoch)}
                yield f"upgrade({i},{t})", put(i, replace(chip, entitled=t, registry=registry, active=t, issued=issued))
        for t in TIERS:
            if t < chip.entitled and not design.cap_follows_tier:
                if design.revocation and chip.epoch >= MAX_EPOCH:
                    continue  # the counter is full: only a swap can lower this chip now
                registry = t if design.registry_sync else chip.registry
                epoch = chip.epoch + 1 if design.revocation else chip.epoch
                yield f"downgrade({i},{t})", put(i, replace(chip, entitled=t, registry=registry, active=t, epoch=epoch))
        for t in TIERS:
            if t != chip.entitled:
                yield f"swap({i},{t})", put(i, fresh_chip(design, t))
        for j, owner in enumerate(world):
            for tier, epoch in sorted(owner.issued):
                if accepts(design, chip, i, j, tier, epoch) and tier > chip.active:
                    yield f"replay({i},token {j}:{tier}@{epoch})", put(i, replace(chip, active=tier))


def violations(world: World) -> list[tuple[str, str]]:
    found = []
    for i, chip in enumerate(world):
        if chip.active > chip.registry:
            found.append(("I1", f"chip {i} runs at tier {chip.active} but the registry records {chip.registry}"))
        if chip.registry != chip.entitled:
            found.append(("I2", f"chip {i} is recorded at tier {chip.registry} but entitled to {chip.entitled}"))
    return found


@dataclass(frozen=True)
class Violation:
    invariant: str
    trace: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class Exploration:
    """``complete`` is True when the search ran out of new states: every reachable state was visited."""

    design: Design
    depth: int
    states: int
    complete: bool
    violations: dict[str, Violation]

    def holds(self, invariant: str) -> bool:
        return invariant not in self.violations


def explore(design: Design, depth: int | None = None) -> Exploration:
    """Breadth-first search; the first violation of each invariant found is a shortest one.

    With ``depth=None`` the search runs to a fixed point (the state space of this model is finite), so a design
    with no violation has none anywhere in the model. With a depth bound it explores only that many actions.
    """
    start = initial_world(design)
    if violations(start):
        raise ValueError("the initial world already violates an invariant")
    seen = {start: ()}
    frontier = [start]
    found: dict[str, Violation] = {}
    level = 0
    while frontier and (depth is None or level < depth):
        following = []
        for world in frontier:
            for label, nxt in successors(design, world):
                if nxt in seen:
                    continue
                seen[nxt] = seen[world] + (label,)
                for invariant, detail in violations(nxt):
                    found.setdefault(invariant, Violation(invariant, seen[nxt], detail))
                following.append(nxt)
        frontier = following
        level += 1
    return Exploration(design, level, len(seen), not frontier, found)


def brute_force_depths(design: Design, depth: int) -> dict[str, int]:
    """Shortest violating action count per invariant by enumerating every action sequence (no state merging)."""
    best: dict[str, int] = {}

    def walk(world: World, taken: int) -> None:
        if taken == depth:
            return
        for _, nxt in successors(design, world):
            for invariant, _ in violations(nxt):
                best[invariant] = min(best.get(invariant, taken + 1), taken + 1)
            walk(nxt, taken + 1)

    walk(initial_world(design), 0)
    return best


def in_place_upgrade_possible(design: Design) -> bool:
    """Can a customer raise a low-tier chip's tier without replacing the die?"""
    return any(label.startswith("upgrade") for label, _ in successors(design, initial_world(design)))
