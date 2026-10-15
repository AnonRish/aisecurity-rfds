"""Pod membership for a fixed set of chips: signed manifests, one-pod binding, authenticated links.

Model (RFD 033, "Pod Formation Protocol"):

1. *Manufacture.* Every chip gets a device key pair; the public key goes into the manufacturer's registry. The
   manufacturer's public key and the chip's maximum pod size are burned into the chip.
2. *Pod configuration (trusted facility).* The manufacturer signs a manifest ``(pod id, generation, members)`` and
   the chip stores it. A chip accepts a manifest only if the signature verifies under the burned manufacturer key,
   it lists the chip itself, it is no larger than the chip's cap, it is bound to the same pod as the chip, and its
   generation is strictly above the chip's one-way fuse counter.
3. *Deployment.* Two chips run a three-message handshake (signed X25519). Only after both signatures verify and
   both identities appear in each side's own manifest does a link become ``HIGH``; its traffic is AES-GCM under
   keys derived from that handshake.
4. *Operation.* Links fall back to ``LOW`` after ``reverify_interval`` ticks unless re-established. A tamper event
   invalidates the membership irreversibly (it burns one fuse of the generation counter).

Assumptions and what is NOT modelled (see the README): the manufacturer key and each chip's secure element are
trusted; ``Chip.device`` is exactly what a successful physical extraction would hand an attacker; ephemeral keys
come from the operating-system RNG; the physical link layer, timing side channels and the chip's non-security
firmware are out of scope.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from aisecurity_rfds.common.channel import (
    AeadReceiver,
    AeadSender,
    derive_session,
    x25519_exchange,
    x25519_generate,
)
from aisecurity_rfds.common.hashing import domain_hash, u32, u64
from aisecurity_rfds.common.nvm import (
    CounterExhaustedError,
    FlashStore,
    JournaledRecord,
    NvmError,
    OtpFuseBank,
    UnaryCounter,
)
from aisecurity_rfds.common.rng import derive_seed, make_rng
from aisecurity_rfds.common.signing import KeyPair, verify

POD_ID_BYTES = 8
KEY_BYTES = 32
SIG_BYTES = 64
NONCE_BYTES = 16

GENERATION_FUSES = 120  # capacity of the one-way generation counter
_DEAD_FUSE = GENERATION_FUSES  # burned when a tamper response finds the counter exhausted
_BIND_OFFSET = GENERATION_FUSES + 1
_BIND_BITS = POD_ID_BYTES * 8
FUSE_BITS = _BIND_OFFSET + _BIND_BITS

MAX_PENDING = 32  # half-open handshakes kept per chip; the oldest is dropped first


class ProvisionError(Exception):
    """A manifest or release was refused; the chip's state is unchanged."""


class HandshakeError(Exception):
    """A handshake message was refused."""


class LinkDownError(Exception):
    """Data was offered to, or received from, a peer without a current HIGH link."""


class LinkMode(Enum):
    LOW = "low"
    HIGH = "high"


# --------------------------------------------------------------------------------------------------------------
# Signed objects
# --------------------------------------------------------------------------------------------------------------


def manifest_digest(pod_id: bytes, generation: int, members: Sequence[bytes]) -> bytes:
    return domain_hash("rfd033/manifest/v1", pod_id, u64(generation), u32(len(members)), *members)


@dataclass(frozen=True)
class PodManifest:
    """The membership list the manufacturer signs and each member stores."""

    pod_id: bytes
    generation: int
    members: tuple[bytes, ...]
    signature: bytes

    def digest(self) -> bytes:
        return manifest_digest(self.pod_id, self.generation, self.members)

    def to_bytes(self) -> bytes:
        return (
            self.pod_id
            + u64(self.generation)
            + u32(len(self.members))
            + b"".join(self.members)
            + self.signature
        )

    @staticmethod
    def from_bytes(blob: bytes) -> "PodManifest":
        header = POD_ID_BYTES + 8 + 4
        if len(blob) < header + SIG_BYTES:
            raise ValueError("manifest too short")
        n = int.from_bytes(blob[POD_ID_BYTES + 8 : header], "big")
        if len(blob) != header + n * KEY_BYTES + SIG_BYTES:
            raise ValueError("manifest length does not match its member count")
        members = tuple(blob[header + i * KEY_BYTES : header + (i + 1) * KEY_BYTES] for i in range(n))
        return PodManifest(
            pod_id=blob[:POD_ID_BYTES],
            generation=int.from_bytes(blob[POD_ID_BYTES : POD_ID_BYTES + 8], "big"),
            members=members,
            signature=blob[-SIG_BYTES:],
        )


def manifest_problem(mfr_pub: bytes, m: PodManifest, *, cap: int) -> str | None:
    """Why ``m`` is unacceptable to a chip with this manufacturer key and size cap, or ``None`` if it is fine."""
    if len(m.pod_id) != POD_ID_BYTES or m.pod_id == bytes(POD_ID_BYTES):
        return "malformed pod id"
    if m.generation < 1 or m.generation >= 2**63:
        return "generation out of range"
    if any(len(k) != KEY_BYTES for k in m.members):
        return "malformed member key"
    if not m.members or len(m.members) > cap:
        return "pod size above the chip's cap"
    if list(m.members) != sorted(set(m.members)):
        return "members not sorted and unique"
    if len(m.signature) != SIG_BYTES or not verify(mfr_pub, m.digest(), m.signature):
        return "bad manufacturer signature"
    return None


@dataclass(frozen=True)
class Release:
    """Manufacturer authorisation for one chip to leave its pod (flash-bound chips only)."""

    device: bytes
    pod_id: bytes
    generation: int
    signature: bytes

    def digest(self) -> bytes:
        return domain_hash("rfd033/release/v1", self.device, self.pod_id, u64(self.generation))


# --------------------------------------------------------------------------------------------------------------
# Handshake messages
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hello1:
    pod_id: bytes
    device: bytes
    eph: bytes
    nonce: bytes
    manifest: PodManifest | None = None  # lets a stale peer catch up (manifest gossip)


@dataclass(frozen=True)
class Hello2:
    init_nonce: bytes
    device: bytes
    eph: bytes
    nonce: bytes
    signature: bytes
    manifest: PodManifest | None = None


@dataclass(frozen=True)
class Finish:
    transcript: bytes
    signature: bytes


def _transcript(
    pod_id: bytes, a_dev: bytes, b_dev: bytes, a_eph: bytes, b_eph: bytes, a_nonce: bytes, b_nonce: bytes
) -> bytes:
    return domain_hash("rfd033/transcript/v1", pod_id, a_dev, b_dev, a_eph, b_eph, a_nonce, b_nonce)


def _role_digest(role: str, transcript: bytes) -> bytes:
    return domain_hash(f"rfd033/handshake-signature/{role}", transcript)


@dataclass
class Link:
    peer: bytes
    sender: AeadSender
    receiver: AeadReceiver
    established_tick: int


# --------------------------------------------------------------------------------------------------------------
# The chip
# --------------------------------------------------------------------------------------------------------------


class Chip:
    """One chip's pod-membership logic and the non-volatile state that protects it.

    Non-volatile state: a one-way generation counter and (optionally) the pod binding in OTP fuses, plus a
    journaled flash record holding the signed manifest. ``bind_in_fuses=True`` makes the pod binding permanent,
    which rules out release and re-assignment but is immune to flash rollback; ``False`` keeps the binding in
    flash (rollback-able) and allows a manufacturer-signed ``Release``.
    """

    def __init__(
        self,
        device: KeyPair,
        mfr_pub: bytes,
        *,
        serial: int = 0,
        pod_cap: int = 64,
        bind_in_fuses: bool = True,
        reverify_interval: int = 10,
    ) -> None:
        if pod_cap < 1 or reverify_interval < 1:
            raise ValueError("pod_cap and reverify_interval must be positive")
        self.device = device  # what a physical extraction of the secure element would yield
        self.serial = serial
        self._mfr_pub = mfr_pub
        self._cap = pod_cap
        self._bind_in_fuses = bind_in_fuses
        self._reverify_interval = reverify_interval
        self.fuses = OtpFuseBank(FUSE_BITS)
        self._gen = UnaryCounter(self.fuses, offset=0, capacity=GENERATION_FUSES)
        self.flash = FlashStore()
        self._manifest_rec = JournaledRecord(self.flash, "manifest")
        self._bind_rec = JournaledRecord(self.flash, "binding")
        self._manifest: PodManifest | None = None
        self._valid = False
        self._links: dict[bytes, Link] = {}
        self._tick = 0
        self._nonce_rng = make_rng(derive_seed("rfd033-nonce", serial))
        self._pending_init: OrderedDict[bytes, tuple[object, bytes, Hello1]] = OrderedDict()
        self._pending_resp: OrderedDict[bytes, tuple[object, bytes, bytes, bytes]] = OrderedDict()
        self.tamper_log: list[str] = []

    # ---- state -------------------------------------------------------------------------------------------

    @property
    def device_id(self) -> bytes:
        return self.device.public_bytes

    @property
    def generation(self) -> int:
        """The one-way fuse counter: no manifest or release at or below it is ever accepted again."""
        return self._gen.value()

    @property
    def dead(self) -> bool:
        return bool(self.fuses.read(_DEAD_FUSE))

    @property
    def is_member(self) -> bool:
        return self._valid and self._manifest is not None and not self.dead

    @property
    def manifest(self) -> PodManifest | None:
        return self._manifest if self.is_member else None

    @property
    def pod_id(self) -> bytes | None:
        return self._manifest.pod_id if self.is_member and self._manifest else None

    @property
    def bound_pod(self) -> bytes | None:
        if self._bind_in_fuses:
            value = 0
            for i in range(_BIND_BITS):
                value = (value << 1) | self.fuses.read(_BIND_OFFSET + i)
            raw = value.to_bytes(POD_ID_BYTES, "big")
        else:
            raw = self._bind_rec.read() or bytes(POD_ID_BYTES)
        return None if raw == bytes(POD_ID_BYTES) else raw

    def _set_bound_pod(self, pod_id: bytes | None) -> None:
        if self._bind_in_fuses:
            if pod_id is None:
                raise ProvisionError("a fuse binding is permanent")
            current = self.bound_pod
            if current is not None and current != pod_id:
                raise ProvisionError("a fuse binding is permanent")
            wanted = int.from_bytes(pod_id, "big")
            for i in range(_BIND_BITS):
                if (wanted >> (_BIND_BITS - 1 - i)) & 1 and not self.fuses.read(_BIND_OFFSET + i):
                    self.fuses.burn(_BIND_OFFSET + i, verify=True)
        else:
            self._bind_rec.write(pod_id or bytes(POD_ID_BYTES))

    def _burn_to(self, generation: int) -> None:
        while self._gen.value() < generation:
            self._gen.increment()

    # ---- provisioning ------------------------------------------------------------------------------------

    def provision(self, m: PodManifest) -> None:
        """Install a signed manifest (first assignment, update, or catch-up). Raises ``ProvisionError``."""
        if self.dead:
            raise ProvisionError("chip is dead")
        problem = manifest_problem(self._mfr_pub, m, cap=self._cap)
        if problem is not None:
            raise ProvisionError(problem)
        if self.device_id not in m.members:
            raise ProvisionError("this chip is not listed in the manifest")
        bound = self.bound_pod
        if bound is not None and bound != m.pod_id:
            raise ProvisionError("chip is bound to a different pod")
        if m.generation <= self.generation:
            raise ProvisionError("stale or replayed generation")
        if m.generation > self._gen.capacity:
            raise ProvisionError("generation exceeds the fuse counter")
        # flash first, fuses second: a crash in between leaves a manifest ahead of the fuses, which boot() repairs
        self._manifest_rec.write(m.to_bytes())
        self._set_bound_pod(m.pod_id)
        self._burn_to(m.generation)
        self._manifest = m
        self._valid = True

    def release(self, r: Release) -> None:
        """Leave the pod on the manufacturer's say-so. Only chips whose binding lives in flash can be released."""
        if self._bind_in_fuses:
            raise ProvisionError("a fuse-bound chip cannot be released")
        if r.device != self.device_id:
            raise ProvisionError("release is for another device")
        bound = self.bound_pod
        if bound is None or r.pod_id != bound:
            raise ProvisionError("release is for another pod")
        if len(r.signature) != SIG_BYTES or not verify(self._mfr_pub, r.digest(), r.signature):
            raise ProvisionError("bad manufacturer signature")
        if r.generation <= self.generation:
            raise ProvisionError("stale or replayed generation")
        if r.generation > self._gen.capacity:
            raise ProvisionError("generation exceeds the fuse counter")
        self._manifest_rec.write(b"")
        self._set_bound_pod(None)
        self._burn_to(r.generation)
        self._manifest = None
        self._valid = False
        self._links.clear()

    def boot(self) -> None:
        """Power-cycle: rebuild the volatile state from non-volatile storage, trusting only the fuses."""
        self._links.clear()
        self._pending_init.clear()
        self._pending_resp.clear()
        self._manifest = None
        self._valid = False
        if self.dead:
            return
        blob = self._manifest_rec.read()
        if not blob:
            return
        try:
            m = PodManifest.from_bytes(blob)
        except ValueError:
            return
        if manifest_problem(self._mfr_pub, m, cap=self._cap) is not None or self.device_id not in m.members:
            return
        bound = self.bound_pod
        if bound is not None and bound != m.pod_id:
            return
        if m.generation < self.generation:
            return  # rolled back, or invalidated by a tamper burn
        try:  # repair after a crash between the flash write and the fuse burns
            if bound is None:
                self._set_bound_pod(m.pod_id)
            if m.generation > self.generation:
                self._burn_to(m.generation)
        except NvmError:
            return  # fuses cannot be programmed right now: stay out of the pod rather than trust flash alone
        self._manifest = m
        self._valid = True

    def tamper(self, reason: str = "tamper detected") -> None:
        """Tamper response: forget the membership, erase the stored manifest and burn one generation fuse.

        The burn is what makes the response irreversible: afterwards no flash image holding the old manifest is
        acceptable. If the burn does not take (for example because the high-voltage supply is starved) the erase
        can be undone by restoring a flash image; the failure is logged and ``generation`` does not move.
        """
        self.tamper_log.append(reason)
        self._valid = False
        self._manifest = None
        self._links.clear()
        self._pending_init.clear()
        self._pending_resp.clear()
        self._manifest_rec.write(b"")
        try:
            self._gen.increment()
        except CounterExhaustedError:
            self.fuses.burn(_DEAD_FUSE)
        except NvmError:
            self.tamper_log.append("fuse burn did not take effect")

    # ---- time and link state -----------------------------------------------------------------------------

    def tick(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("time does not run backwards")
        self._tick += n

    def link_mode(self, peer: bytes) -> LinkMode:
        link = self._links.get(peer)
        if link is None or not self.is_member or self._manifest is None:
            return LinkMode.LOW
        if peer not in self._manifest.members:
            return LinkMode.LOW
        if self._tick - link.established_tick >= self._reverify_interval:
            return LinkMode.LOW
        return LinkMode.HIGH

    def high_peers(self) -> list[bytes]:
        return sorted(p for p in self._links if self.link_mode(p) is LinkMode.HIGH)

    def _require_high(self, peer: bytes) -> Link:
        if self.link_mode(peer) is not LinkMode.HIGH or peer not in self._links:
            raise LinkDownError("no current high-bandwidth link to this peer")
        return self._links[peer]

    def seal(self, peer: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        return self._require_high(peer).sender.seal(plaintext, aad)

    def unseal(self, peer: bytes, blob: bytes, aad: bytes = b"") -> bytes:
        return self._require_high(peer).receiver.open(blob, aad)

    # ---- handshake ---------------------------------------------------------------------------------------

    def _require_member(self) -> PodManifest:
        if not self.is_member or self._manifest is None:
            raise HandshakeError("not a current pod member")
        return self._manifest

    def _require_peer(self, device: bytes, pod_id: bytes) -> None:
        m = self._require_member()
        if pod_id != m.pod_id:
            raise HandshakeError("peer names a different pod")
        if device == self.device_id:
            raise HandshakeError("peer presents this chip's own identity")
        if device not in m.members:
            raise HandshakeError("peer is not in this chip's manifest")

    def _maybe_adopt(self, m: PodManifest | None) -> None:
        """Manifest gossip: a member of the same pod adopts a newer manufacturer-signed manifest that lists it."""
        if m is None or self._manifest is None or not self.is_member:
            return
        if m.pod_id != self._manifest.pod_id or m.generation <= self._manifest.generation:
            return
        try:
            self.provision(m)
        except ProvisionError:
            return

    def _remember(self, table: OrderedDict, key: bytes, value: tuple) -> None:
        table[key] = value
        while len(table) > MAX_PENDING:
            table.popitem(last=False)

    def begin(self) -> Hello1:
        m = self._require_member()
        priv, pub = x25519_generate()
        hello = Hello1(m.pod_id, self.device_id, pub, self._nonce_rng.bytes(NONCE_BYTES), m)
        self._remember(self._pending_init, hello.nonce, (priv, pub, hello))
        return hello

    def respond(self, h: Hello1) -> Hello2:
        self._require_member()
        self._maybe_adopt(h.manifest)
        self._require_peer(h.device, h.pod_id)
        priv, pub = x25519_generate()
        nonce = self._nonce_rng.bytes(NONCE_BYTES)
        th = _transcript(h.pod_id, h.device, self.device_id, h.eph, pub, h.nonce, nonce)
        sig = self.device.sign(_role_digest("responder", th))
        self._remember(self._pending_resp, th, (priv, h.device, h.eph, h.pod_id))
        return Hello2(h.nonce, self.device_id, pub, nonce, sig, self._manifest)

    def complete(self, h2: Hello2) -> Finish:
        entry = self._pending_init.pop(h2.init_nonce, None)
        if entry is None:
            raise HandshakeError("no handshake in progress for this reply")
        priv, pub, hello = entry
        self._require_member()
        self._maybe_adopt(h2.manifest)
        self._require_peer(h2.device, hello.pod_id)
        th = _transcript(hello.pod_id, self.device_id, h2.device, pub, h2.eph, hello.nonce, h2.nonce)
        if not verify(h2.device, _role_digest("responder", th), h2.signature):
            raise HandshakeError("bad responder signature")
        shared = x25519_exchange(priv, h2.eph)  # type: ignore[arg-type]
        sender, receiver = derive_session(shared, th, initiator=True)
        self._links[h2.device] = Link(h2.device, sender, receiver, self._tick)
        return Finish(th, self.device.sign(_role_digest("initiator", th)))

    def finalize(self, f: Finish) -> None:
        entry = self._pending_resp.pop(f.transcript, None)
        if entry is None:
            raise HandshakeError("no handshake in progress for this transcript")
        priv, initiator, initiator_eph, pod_id = entry
        self._require_member()
        self._require_peer(initiator, pod_id)
        if not verify(initiator, _role_digest("initiator", f.transcript), f.signature):
            raise HandshakeError("bad initiator signature")
        shared = x25519_exchange(priv, initiator_eph)  # type: ignore[arg-type]
        sender, receiver = derive_session(shared, f.transcript, initiator=False)
        self._links[initiator] = Link(initiator, sender, receiver, self._tick)


# --------------------------------------------------------------------------------------------------------------
# The manufacturer / trusted facility
# --------------------------------------------------------------------------------------------------------------


class Manufacturer:
    """Holds the manifest-signing key and the device registry; ``form_pod`` is the disciplined path.

    ``sign_manifest`` and ``sign_release`` are the raw signing operations. They enforce nothing, which is exactly
    what a compromised or careless facility would offer; ``form_pod`` and ``replace_member`` are the disciplined
    wrappers (one pod per chip, generations above every member's fuse counter).
    """

    def __init__(self, seed: bytes = b"rfd033-manufacturer") -> None:
        self._seed = seed
        self._key = KeyPair.from_seed(domain_hash("rfd033/manufacturer-key", seed))
        self._serial = 0
        self.registry: dict[bytes, int] = {}
        self.assignment: dict[bytes, bytes | None] = {}
        self.retired: set[bytes] = set()

    @property
    def public(self) -> bytes:
        return self._key.public_bytes

    def make_chip(
        self, *, pod_cap: int = 64, bind_in_fuses: bool = True, reverify_interval: int = 10
    ) -> Chip:
        serial = self._serial
        self._serial += 1
        key = KeyPair.from_seed(domain_hash("rfd033/device-key", self._seed, u64(serial)))
        chip = Chip(
            key,
            self.public,
            serial=serial,
            pod_cap=pod_cap,
            bind_in_fuses=bind_in_fuses,
            reverify_interval=reverify_interval,
        )
        self.registry[chip.device_id] = serial
        self.assignment[chip.device_id] = None
        return chip

    def pod_id_for(self, label: int) -> bytes:
        return domain_hash("rfd033/pod-id", self._seed, u64(label))[:POD_ID_BYTES]

    def sign_manifest(self, pod_id: bytes, generation: int, members: Sequence[bytes]) -> PodManifest:
        ordered = tuple(sorted(set(members)))
        return PodManifest(pod_id, generation, ordered, self._key.sign(manifest_digest(pod_id, generation, ordered)))

    def sign_release(self, device: bytes, pod_id: bytes, generation: int) -> Release:
        r = Release(device, pod_id, generation, b"")
        return Release(device, pod_id, generation, self._key.sign(r.digest()))

    def form_pod(self, chips: Sequence[Chip], label: int) -> PodManifest:
        """Select chips, assign a pod id, sign the member list and load it into every chip."""
        for chip in chips:
            if self.assignment.get(chip.device_id) is not None or chip.device_id in self.retired:
                raise ValueError("chip is already assigned to a pod")
        pod_id = self.pod_id_for(label)
        generation = max(c.generation for c in chips) + 1
        manifest = self.sign_manifest(pod_id, generation, [c.device_id for c in chips])
        for chip in chips:
            chip.provision(manifest)
            self.assignment[chip.device_id] = pod_id
        return manifest

    def replace_member(self, current: PodManifest, failed: bytes, replacement: Chip) -> PodManifest:
        """Issue the next manifest with ``failed`` swapped for a factory-fresh ``replacement`` chip."""
        if self.assignment.get(replacement.device_id) is not None or replacement.device_id in self.retired:
            raise ValueError("replacement chip is already assigned to a pod")
        if failed not in current.members:
            raise ValueError("failed chip is not in the current manifest")
        members = [m for m in current.members if m != failed] + [replacement.device_id]
        generation = max(current.generation, replacement.generation) + 1
        self.retired.add(failed)
        self.assignment[replacement.device_id] = current.pod_id
        return self.sign_manifest(current.pod_id, generation, members)

    def release_chip(self, chip: Chip) -> Release:
        """Release a flash-bound chip from its pod so that it can be assigned again."""
        pod = self.assignment.get(chip.device_id)
        if pod is None:
            raise ValueError("chip is not assigned")
        release = self.sign_release(chip.device_id, pod, chip.generation + 1)
        self.assignment[chip.device_id] = None
        return release
