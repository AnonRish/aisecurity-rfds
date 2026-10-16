"""The chip registry: records, signed update requests, a verifiable log plus map, and what a client can check (RFD 031).

The RFD's technical requirements map onto this module as follows.

* Registry database: ``Record`` values held in a sparse Merkle map keyed by chip id (``smt.py``).
* Update protocol: signed ``Request`` objects applied by the pure function ``apply_request``. Every request names the
  record version it applies to, so an old request cannot be replayed and two competing requests cannot both succeed.
* Audit trail: every accepted request is appended, with the record it produced, to a Merkle log, and the operator signs
  a tree head that commits to both the log and the map.
* Query interface: ``Registry.query`` returns a record with a proof against the signed head; ``verify_query`` checks
  the signature, the head's age, and the proof (including a proof that a chip is *not* registered).

``apply_request`` is deterministic and takes the time as an argument. The operator uses it to accept requests, a
client uses it to replay a chip's history, and an auditor uses it to replay the whole log, so the three can disagree
only if the operator's published data are inconsistent. One rule is only partly shared: a copied chip is told apart
by an attestation counter that fails to rise. A client or auditor can check the counters that the log carries (one per
completed transfer); the operator's tracker also sees the attestations that inspectors present, which the log does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from aisecurity_rfds.common.hashing import domain_hash, u32, u64
from aisecurity_rfds.common.merkle import MerkleLog, verify_inclusion
from aisecurity_rfds.common.signing import KeyPair, verify

from .identity import FRESH, Attestation, CounterTracker, chip_id_of, verify_attestation
from .smt import MapProof, SparseMerkleMap, verify_proof

ACTIVE = "active"
IN_TRANSIT = "in-transit"
RETIRED = "retired"

ENROLL = "enroll"
START_TRANSFER = "start-transfer"
COMPLETE_TRANSFER = "complete-transfer"
CANCEL_TRANSFER = "cancel-transfer"
RELOCATE = "relocate"
RETIRE = "retire"
KINDS = (ENROLL, START_TRANSFER, COMPLETE_TRANSFER, CANCEL_TRANSFER, RELOCATE, RETIRE)

RETIRE_REASONS = ("destroyed", "recycled", "repurposed", "lost")
KEY_BYTES = 32


class Rejected(ValueError):
    """A request that the rules refuse; ``code`` says which rule."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class Policy:
    """Public parameters of the registry: everyone who replays the log needs the same ones."""

    manufacturers: frozenset[bytes]
    max_transfer_window: int = 14 * 24 * 3600
    embargoed: frozenset[str] = frozenset()
    destruction_witnesses: frozenset[bytes] = frozenset()


def _pack(*parts: bytes) -> bytes:
    return b"".join(u32(len(p)) + p for p in parts)


@dataclass(frozen=True)
class Record:
    chip_id: bytes
    public_key: bytes
    manufacturer: bytes
    owner: bytes
    status: str
    jurisdiction: str
    location_commitment: bytes
    version: int
    pending_buyer: bytes = b""
    pending_until: int = 0
    retired_reason: str = ""

    def encode(self) -> bytes:
        return _pack(
            self.chip_id,
            self.public_key,
            self.manufacturer,
            self.owner,
            self.status.encode(),
            self.jurisdiction.encode(),
            self.location_commitment,
            u64(self.version),
            self.pending_buyer,
            u64(self.pending_until),
            self.retired_reason.encode(),
        )


@dataclass(frozen=True)
class Request:
    kind: str
    chip_id: bytes
    expected_version: int
    fields: tuple[tuple[str, bytes], ...] = ()

    def digest(self) -> bytes:
        parts = [self.kind.encode(), self.chip_id, u64(self.expected_version)]
        for name, value in self.fields:
            parts += [name.encode(), value]
        return domain_hash("rfd031.request", *parts)

    def get(self, name: str, default: bytes | None = None) -> bytes:
        for key, value in self.fields:
            if key == name:
                return value
        if default is None:
            raise Rejected("bad-field", f"missing field {name!r}")
        return default


def _as_bytes(value: bytes | str | int) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return u64(value)


def make_request(kind: str, chip_id: bytes, expected_version: int, **fields: bytes | str | int) -> Request:
    return Request(kind, chip_id, expected_version, tuple(sorted((k, _as_bytes(v)) for k, v in fields.items())))


@dataclass(frozen=True)
class SignedRequest:
    request: Request
    signatures: tuple[tuple[bytes, bytes], ...]  # (public key, signature over request.digest())

    def signed_by(self, public_key: bytes) -> bool:
        digest = self.request.digest()
        return any(pk == public_key and verify(public_key, digest, sig) for pk, sig in self.signatures)

    def encode(self) -> bytes:
        return _pack(self.request.digest(), *[_pack(pk, sig) for pk, sig in self.signatures])


def sign_request(request: Request, *signers: KeyPair) -> SignedRequest:
    digest = request.digest()
    return SignedRequest(request, tuple((k.public_bytes, k.sign(digest)) for k in signers))


def transfer_challenge(record: Record) -> bytes:
    """What a chip is asked to sign to complete a transfer: bound to the chip, the record version and both parties."""
    return domain_hash(
        "rfd031.transfer", record.chip_id, u64(record.version), record.owner, record.pending_buyer, u64(record.pending_until)
    )


def _int(req: Request, name: str) -> int:
    return int.from_bytes(req.get(name), "big")


def _text(req: Request, name: str) -> str:
    try:
        return req.get(name).decode()
    except UnicodeDecodeError:
        raise Rejected("bad-field", f"{name} is not text") from None


def _jurisdiction(req: Request, policy: Policy) -> str:
    value = _text(req, "jurisdiction")
    if not value or len(value) > 64:
        raise Rejected("bad-field", "jurisdiction must be 1 to 64 characters")
    if value in policy.embargoed:
        raise Rejected("embargoed", f"{value} may not hold registered chips")
    return value


def _location(req: Request) -> bytes:
    value = req.get("location", b"")
    if value and len(value) != 32:
        raise Rejected("bad-field", "a location commitment is 32 bytes or empty")
    return value


def _transfer_attestation(record: Record, req: Request) -> Attestation:
    return Attestation(record.chip_id, transfer_challenge(record), _int(req, "att_counter"), req.get("att_signature"))


def _enroll(prev: Record | None, signed: SignedRequest, policy: Policy) -> Record:
    req = signed.request
    if prev is not None:
        raise Rejected("duplicate", "this chip id is already registered")
    if req.expected_version != 0:
        raise Rejected("stale-version", "an enrolment starts from version 0")
    manufacturer = next((pk for pk in sorted(policy.manufacturers) if signed.signed_by(pk)), None)
    if manufacturer is None:
        raise Rejected("unauthorised", "enrolment needs an approved manufacturer's signature")
    public_key = req.get("public_key")
    if len(public_key) != KEY_BYTES or chip_id_of(public_key) != req.chip_id:
        raise Rejected("bad-chip-id", "the chip id must be the hash of the chip's public key")
    return Record(
        chip_id=req.chip_id,
        public_key=public_key,
        manufacturer=manufacturer,
        owner=manufacturer,
        status=ACTIVE,
        jurisdiction=_jurisdiction(req, policy),
        location_commitment=_location(req),
        version=1,
    )


def _start_transfer(prev: Record, signed: SignedRequest, now: int, policy: Policy) -> Record:
    req = signed.request
    if not signed.signed_by(prev.owner):
        raise Rejected("unauthorised", "only the registered owner can sell")
    if prev.status != ACTIVE:
        raise Rejected("bad-state", "only an active chip can be put up for transfer")
    buyer = req.get("buyer")
    if len(buyer) != KEY_BYTES or buyer == prev.owner:
        raise Rejected("bad-field", "the buyer must be another party's public key")
    expires = _int(req, "expires")
    if not now < expires <= now + policy.max_transfer_window:
        raise Rejected("bad-window", "the offer must expire in the future and within the policy window")
    return replace(prev, status=IN_TRANSIT, pending_buyer=buyer, pending_until=expires, version=prev.version + 1)


def _complete_transfer(prev: Record, signed: SignedRequest, now: int, policy: Policy) -> Record:
    req = signed.request
    if prev.status != IN_TRANSIT:
        raise Rejected("bad-state", "no transfer is pending")
    if not signed.signed_by(prev.pending_buyer):
        raise Rejected("unauthorised", "only the named buyer can complete the transfer")
    if now > prev.pending_until:
        raise Rejected("expired", "the offer has lapsed")
    jurisdiction = _jurisdiction(req, policy)
    if not verify_attestation(prev.public_key, transfer_challenge(prev), _transfer_attestation(prev, req)):
        raise Rejected("bad-attestation", "the chip did not sign this transfer's challenge")
    return replace(
        prev,
        owner=prev.pending_buyer,
        status=ACTIVE,
        jurisdiction=jurisdiction,
        location_commitment=b"",
        pending_buyer=b"",
        pending_until=0,
        version=prev.version + 1,
    )


def _cancel_transfer(prev: Record, signed: SignedRequest, now: int, policy: Policy) -> Record:
    if prev.status != IN_TRANSIT:
        raise Rejected("bad-state", "no transfer is pending")
    if not signed.signed_by(prev.pending_buyer):
        if not signed.signed_by(prev.owner):
            raise Rejected("unauthorised", "only the buyer or the owner can cancel")
        if now <= prev.pending_until:
            raise Rejected("not-expired", "the owner cannot withdraw before the offer lapses")
    return replace(prev, status=ACTIVE, pending_buyer=b"", pending_until=0, version=prev.version + 1)


def _relocate(prev: Record, signed: SignedRequest, now: int, policy: Policy) -> Record:
    req = signed.request
    if not signed.signed_by(prev.owner):
        raise Rejected("unauthorised", "only the registered owner can update the location")
    if prev.status != ACTIVE:
        raise Rejected("bad-state", "only an active chip can be relocated")
    return replace(
        prev, jurisdiction=_jurisdiction(req, policy), location_commitment=_location(req), version=prev.version + 1
    )


def _retire(prev: Record, signed: SignedRequest, now: int, policy: Policy) -> Record:
    req = signed.request
    if not signed.signed_by(prev.owner):
        raise Rejected("unauthorised", "only the registered owner can retire a chip")
    if prev.status != ACTIVE:
        raise Rejected("bad-state", "cancel the pending transfer before retiring")
    reason = _text(req, "reason")
    if reason not in RETIRE_REASONS:
        raise Rejected("bad-field", f"reason must be one of {RETIRE_REASONS}")
    if reason == "destroyed" and policy.destruction_witnesses:
        if not any(signed.signed_by(w) for w in sorted(policy.destruction_witnesses)):
            raise Rejected("unwitnessed", "destruction needs an accredited witness's signature")
    return replace(prev, status=RETIRED, location_commitment=b"", retired_reason=reason, version=prev.version + 1)


_HANDLERS = {
    START_TRANSFER: _start_transfer,
    COMPLETE_TRANSFER: _complete_transfer,
    CANCEL_TRANSFER: _cancel_transfer,
    RELOCATE: _relocate,
    RETIRE: _retire,
}


def apply_request(prev: Record | None, signed: SignedRequest, now: int, policy: Policy) -> Record:
    """The record that results from ``signed`` applied to ``prev`` at time ``now``, or ``Rejected``."""
    req = signed.request
    if req.kind not in KINDS:
        raise Rejected("unknown-kind", req.kind)
    if len({name for name, _ in req.fields}) != len(req.fields):
        raise Rejected("bad-field", "duplicate field")
    if req.kind == ENROLL:
        return _enroll(prev, signed, policy)
    if prev is None or prev.chip_id != req.chip_id:
        raise Rejected("unknown-chip", "no such chip in the registry")
    if req.expected_version != prev.version:
        raise Rejected("stale-version", f"request is for version {req.expected_version}, record is at {prev.version}")
    if prev.status == RETIRED:
        raise Rejected("retired", "retirement is final")
    return _HANDLERS[req.kind](prev, signed, now, policy)


# ---- the operator's side -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEntry:
    index: int
    timestamp: int
    signed: SignedRequest
    record_after: Record

    def encode(self) -> bytes:
        return _pack(u64(self.index), u64(self.timestamp), self.signed.encode(), self.record_after.encode())


@dataclass(frozen=True)
class SignedTreeHead:
    size: int
    log_root: bytes
    map_root: bytes
    timestamp: int
    signature: bytes

    def message(self) -> bytes:
        return sth_message(self.size, self.log_root, self.map_root, self.timestamp)

    def verify(self, public_key: bytes) -> bool:
        return verify(public_key, self.message(), self.signature)


def sth_message(size: int, log_root: bytes, map_root: bytes, timestamp: int) -> bytes:
    return domain_hash("rfd031.sth", u64(size), log_root, map_root, u64(timestamp))


@dataclass(frozen=True)
class Alert:
    kind: str  # clone-suspected, zombie, invalid-attestation or unknown-chip
    chip_id: bytes
    time: int
    detail: str = ""


class Registry:
    def __init__(self, operator_key: KeyPair, policy: Policy) -> None:
        self._key = operator_key
        self.policy = policy
        self._log = MerkleLog()
        self._map = SparseMerkleMap()
        self._records: dict[bytes, Record] = {}
        self._entries: list[LogEntry] = []
        self._by_chip: dict[bytes, list[int]] = {}
        self._last_time = 0
        self.tracker = CounterTracker()
        self.alerts: list[Alert] = []

    @classmethod
    def replicate(cls, operator_key: KeyPair, policy: Policy, entries: Sequence[LogEntry]) -> Registry:
        """A registry rebuilt by re-validating every request in ``entries``."""
        registry = cls(operator_key, policy)
        for entry in entries:
            registry.submit(entry.signed, entry.timestamp)
        return registry

    @property
    def operator_public_key(self) -> bytes:
        return self._key.public_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, chip_id: bytes) -> Record | None:
        return self._records.get(chip_id)

    def entries(self) -> tuple[LogEntry, ...]:
        return tuple(self._entries)

    def submit(self, signed: SignedRequest, now: int) -> LogEntry:
        if now < self._last_time:
            raise Rejected("clock", "time must not go backwards")
        req = signed.request
        prev = self._records.get(req.chip_id)
        record = apply_request(prev, signed, now, self.policy)
        if req.kind == COMPLETE_TRANSFER and prev is not None:
            attestation = _transfer_attestation(prev, req)
            last = self.tracker.last(prev.chip_id)
            if self.tracker.observe(attestation) != FRESH:
                self.alerts.append(
                    Alert("clone-suspected", prev.chip_id, now, f"transfer attestation counter {attestation.counter} <= {last}")
                )
                raise Rejected("counter-regression", "the chip's counter went backwards: copied or rolled back")
        entry = LogEntry(len(self._entries), now, signed, record)
        self._log.append(entry.encode())
        self._map.set(record.chip_id, record.encode())
        self._records[record.chip_id] = record
        self._by_chip.setdefault(record.chip_id, []).append(entry.index)
        self._entries.append(entry)
        self._last_time = now
        return entry

    def sth(self, now: int) -> SignedTreeHead:
        size, log_root, map_root = len(self._entries), self._log.root(), self._map.root()
        return SignedTreeHead(size, log_root, map_root, now, self._key.sign(sth_message(size, log_root, map_root, now)))

    def consistency_proof(self, first: int, second: int | None = None) -> list[bytes]:
        return self._log.consistency_proof(first, second)

    def query(self, chip_id: bytes, now: int) -> QueryResponse:
        return QueryResponse(chip_id, self._records.get(chip_id), self._map.prove(chip_id), self.sth(now))

    def history(self, chip_id: bytes, now: int) -> HistoryResponse:
        sth = self.sth(now)
        items = tuple(
            HistoryItem(self._entries[i], tuple(self._log.inclusion_proof(i, sth.size))) for i in self._by_chip.get(chip_id, ())
        )
        return HistoryResponse(chip_id, items, sth)

    def observe_attestation(self, attestation: Attestation, challenge: bytes, now: int) -> list[Alert]:
        """Check an attestation seen by an inspector and record what it reveals about the chip."""
        chip_id = attestation.chip_id
        record = self._records.get(chip_id)
        if record is None:
            alerts = [Alert("unknown-chip", chip_id, now)]
        elif not verify_attestation(record.public_key, challenge, attestation):
            alerts = [Alert("invalid-attestation", chip_id, now)]
        else:
            alerts = []
            if record.status == RETIRED:
                alerts.append(Alert("zombie", chip_id, now, f"retired as {record.retired_reason}"))
            last = self.tracker.last(chip_id)
            if self.tracker.observe(attestation) != FRESH:
                alerts.append(Alert("clone-suspected", chip_id, now, f"counter {attestation.counter} <= {last}"))
        self.alerts.extend(alerts)
        return alerts


# ---- what a client or auditor checks -------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryResponse:
    chip_id: bytes
    record: Record | None
    proof: MapProof
    sth: SignedTreeHead


@dataclass(frozen=True)
class HistoryItem:
    entry: LogEntry
    proof: tuple[bytes, ...]


@dataclass(frozen=True)
class HistoryResponse:
    chip_id: bytes
    items: tuple[HistoryItem, ...]
    sth: SignedTreeHead


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = "ok"
    record: Record | None = None


def verify_query(
    operator_public_key: bytes, response: QueryResponse, *, now: int, max_age: int, max_skew: int = 60
) -> Verdict:
    """Is ``response`` a fresh, signed, provable statement about the chip (present or absent)?"""
    sth = response.sth
    if not sth.verify(operator_public_key):
        return Verdict(False, "head-signature")
    if sth.timestamp - now > max_skew:
        return Verdict(False, "future-head")
    if now - sth.timestamp > max_age:
        return Verdict(False, "stale-head")
    record = response.record
    if record is not None and record.chip_id != response.chip_id:
        return Verdict(False, "wrong-record")
    value = None if record is None else record.encode()
    if not verify_proof(sth.map_root, response.chip_id, value, response.proof):
        return Verdict(False, "bad-proof")
    return Verdict(True, "ok", record)


def verify_history(
    operator_public_key: bytes, policy: Policy, history: HistoryResponse, current: QueryResponse
) -> Verdict:
    """Replay a chip's logged requests and check that they end in the record the map proves."""
    sth = history.sth
    if not sth.verify(operator_public_key):
        return Verdict(False, "head-signature")
    cur = current.sth
    if (sth.size, sth.log_root, sth.map_root) != (cur.size, cur.log_root, cur.map_root):
        return Verdict(False, "heads-differ")
    prev: Record | None = None
    tracker = CounterTracker()
    for item in history.items:
        entry = item.entry
        if entry.signed.request.chip_id != history.chip_id:
            return Verdict(False, "wrong-chip")
        if entry.index >= sth.size or not verify_inclusion(entry.encode(), entry.index, sth.size, list(item.proof), sth.log_root):
            return Verdict(False, "inclusion")
        try:
            rebuilt = apply_request(prev, entry.signed, entry.timestamp, policy)
        except Rejected as exc:
            return Verdict(False, f"invalid-entry:{exc.code}")
        if rebuilt != entry.record_after:
            return Verdict(False, "state-mismatch")
        if entry.signed.request.kind == COMPLETE_TRANSFER and prev is not None:
            if tracker.observe(_transfer_attestation(prev, entry.signed.request)) != FRESH:
                return Verdict(False, "counter-regression")
        prev = rebuilt
    if prev != current.record:
        return Verdict(False, "history-disagrees-with-map")
    return Verdict(True, "ok", prev)


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    entries: int
    problems: tuple[str, ...]


def audit_log(
    entries: Sequence[LogEntry],
    policy: Policy,
    sth: SignedTreeHead,
    operator_public_key: bytes,
    prior_heads: Sequence[SignedTreeHead] = (),
) -> AuditReport:
    """Replay the whole log and compare the roots it produces with the signed head and with any earlier heads kept.

    Earlier heads let an auditor see a history that was rewritten into another valid history (reordered, say), and bound
    how far an entry can be backdated: an entry must be dated no earlier than every head published without it and no
    later than the first head that covers it.
    """
    problems: list[str] = []
    if not sth.verify(operator_public_key):
        problems.append("head signature is invalid")
    for head in prior_heads:
        if not head.verify(operator_public_key):
            problems.append(f"an earlier head (size {head.size}) has an invalid signature")
        if head.size > len(entries):
            problems.append(f"an earlier head covers {head.size} entries but the log has {len(entries)}")
    if len(entries) < sth.size:
        problems.append(f"log has {len(entries)} entries but the head covers {sth.size}")
    log, smap = MerkleLog(), SparseMerkleMap()
    records: dict[bytes, Record] = {}
    tracker = CounterTracker()
    last_time = 0
    checked = min(len(entries), sth.size)

    def compare_earlier_heads(size: int) -> None:
        for head in prior_heads:
            if head.size == size and log.root() != head.log_root:
                problems.append(f"log root at size {size} differs from an earlier head")
            if head.size == size and smap.root() != head.map_root:
                problems.append(f"map root at size {size} differs from an earlier head")

    for i, entry in enumerate(entries[:checked]):
        compare_earlier_heads(i)
        if entry.index != i:
            problems.append(f"entry {i} claims index {entry.index}")
        if entry.timestamp < last_time:
            problems.append(f"entry {i} is dated before the entry preceding it")
        last_time = max(last_time, entry.timestamp)
        if any(h.size <= i and entry.timestamp < h.timestamp for h in prior_heads):
            problems.append(f"entry {i} is dated before a head that was published without it")
        if any(h.size > i and entry.timestamp > h.timestamp for h in prior_heads):
            problems.append(f"entry {i} is dated after a head that covers it")
        log.append(entry.encode())
        chip_id = entry.signed.request.chip_id
        before = records.get(chip_id)
        try:
            rebuilt = apply_request(before, entry.signed, entry.timestamp, policy)
        except Rejected as exc:
            problems.append(f"entry {i} is not a valid request ({exc.code})")
            continue
        if rebuilt != entry.record_after:
            problems.append(f"entry {i} records a state its request does not produce")
        if entry.signed.request.kind == COMPLETE_TRANSFER and before is not None:
            if tracker.observe(_transfer_attestation(before, entry.signed.request)) != FRESH:
                problems.append(f"entry {i} completes a transfer with a counter that did not rise")
        records[chip_id] = rebuilt
        smap.set(chip_id, rebuilt.encode())
    compare_earlier_heads(checked)
    if checked == sth.size:
        if log.root() != sth.log_root:
            problems.append("log root differs from the signed head")
        if smap.root() != sth.map_root:
            problems.append("map root differs from the signed head")
    return AuditReport(not problems, checked, tuple(problems))
