"""Where the high-bandwidth decision lives: the RFD's "Bandwidth Limiting Implementation" rows, executed.

Three designs decide whether a high-bandwidth frame is accepted as pod traffic:

* ``FIRMWARE_LIMITER`` ("Firmware"): a pod list and a limit in mutable firmware; frames are classified by source id.
* ``PROTOCOL_ONLY`` ("Protocol"): an authenticated handshake sets a per-peer flag; later frames are accepted by
  source id alone.
* ``KEY_GATED`` ("Cryptographic"): after the handshake every frame must authenticate under the session key.

Four attacker capabilities are tried against each: none, injecting frames with a chosen source id on a link the
attacker can reach, rewriting the mutable firmware of the chips the attacker owns, and extracting device keys from
the secure elements of chips in other pods. A cell is "bypassed" if the attacker gets a frame accepted as pod
traffic by a chip of pod A although it originates outside pod A. Each verdict is the result of running the code
path, not a lookup. The physical row of the RFD (disabling transceivers) cannot be modelled in software.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aisecurity_rfds.common.channel import DecryptionError, ReplayError
from aisecurity_rfds.common.rng import make_rng

from .cluster import build_fleet, establish, impersonate
from .pod import Chip, LinkDownError, LinkMode


JUNK = make_rng(33).bytes(40)  # a frame body the attacker invented


class Design(Enum):
    FIRMWARE_LIMITER = "firmware limiter"
    PROTOCOL_ONLY = "handshake then plain frames"
    KEY_GATED = "handshake then AEAD frames"


class Capability(Enum):
    NONE = "none"
    SPOOF_FRAMES = "spoof source id"
    MODIFY_FIRMWARE = "rewrite firmware"
    EXTRACT_KEYS = "extract device keys"


@dataclass(frozen=True)
class Frame:
    src: bytes
    payload: bytes


class FirmwareCompromisedChip(Chip):
    """A chip whose mutable firmware the attacker rewrote: every software-level link check answers 'high'.

    The secure element, the fuses and the session keys are untouched: firmware cannot conjure a key.
    """

    def link_mode(self, peer: bytes) -> LinkMode:  # noqa: D401 - behaviour is the point
        return LinkMode.HIGH


def compromise_firmware(chip: Chip) -> Chip:
    chip.__class__ = FirmwareCompromisedChip
    return chip


class FirmwareLimiterReceiver:
    """Pod list in mutable firmware, frames classified by source id."""

    def __init__(self, pod_list: set[bytes]) -> None:
        self.config = {"pod_list": set(pod_list)}

    def accept_hbw(self, frame: Frame) -> bool:
        return frame.src in self.config["pod_list"]


class ProtocolOnlyReceiver:
    """Handshake state decides; frames carry no authenticator."""

    def __init__(self, chip: Chip) -> None:
        self.chip = chip

    def accept_hbw(self, frame: Frame) -> bool:
        return self.chip.link_mode(frame.src) is LinkMode.HIGH


class KeyGatedReceiver:
    """A frame is accepted only if it decrypts under the session key shared with its claimed source."""

    def __init__(self, chip: Chip) -> None:
        self.chip = chip

    def accept_hbw(self, frame: Frame) -> bool:
        try:
            self.chip.unseal(frame.src, frame.payload)
        except (LinkDownError, DecryptionError, ReplayError):
            return False
        return True


@dataclass(frozen=True)
class Outcome:
    design: Design
    capability: Capability
    bypassed: bool
    how: str


def attempt(design: Design, capability: Capability) -> Outcome:
    """Can an attacker get pod A's chip ``a1`` to accept outside traffic as pod traffic?"""
    fleet = build_fleet(2, 3, seed=b"gate")
    a1, a2 = fleet.chips[0], fleet.chips[1]
    b1 = fleet.chips[3]
    establish(a1, a2)  # honest traffic inside pod A

    def stolen_a2_bridge() -> Chip:
        emulator = impersonate(fleet.mfr.public, a2.device, fleet.manifests[0], serial=777, pod_cap=3)
        if not establish(emulator, a1):
            raise AssertionError("an emulator holding a pod member's key must be able to link")
        return emulator

    if design is Design.FIRMWARE_LIMITER:
        rx = FirmwareLimiterReceiver({a1.device_id, a2.device_id})
        if capability is Capability.NONE:
            frame, how = Frame(b1.device_id, b"x"), "frame from an outsider carries the outsider's id"
        elif capability is Capability.SPOOF_FRAMES:
            frame, how = Frame(a2.device_id, b"x"), "source id is only a label"
        elif capability is Capability.MODIFY_FIRMWARE:
            rx.config["pod_list"].add(b1.device_id)
            frame, how = Frame(b1.device_id, b"x"), "pod list edited in mutable firmware"
        else:
            frame, how = Frame(b1.device_id, b"x"), "keys play no role in this design"
        return Outcome(design, capability, rx.accept_hbw(frame), how)

    if design is Design.PROTOCOL_ONLY:
        receiver_chip = a1
        if capability is Capability.NONE:
            frame, how = Frame(b1.device_id, b"x"), "no handshake with the outsider"
        elif capability is Capability.SPOOF_FRAMES:
            frame, how = Frame(a2.device_id, b"x"), "flag is per peer id; frames are not authenticated"
        elif capability is Capability.MODIFY_FIRMWARE:
            compromise_firmware(receiver_chip)
            frame, how = Frame(b1.device_id, b"x"), "firmware sets the 'authenticated' flag itself"
        else:
            bridge = stolen_a2_bridge()
            frame, how = Frame(bridge.device_id, b"x"), "bridge passes the handshake with a stolen key"
        return Outcome(design, capability, ProtocolOnlyReceiver(receiver_chip).accept_hbw(frame), how)

    # KEY_GATED
    rx = KeyGatedReceiver(a1)
    if capability is Capability.NONE:
        frame, how = Frame(b1.device_id, JUNK), "no session key shared with the outsider"
    elif capability is Capability.SPOOF_FRAMES:
        frame, how = Frame(a2.device_id, JUNK), "forged frame fails authentication"
    elif capability is Capability.MODIFY_FIRMWARE:
        compromise_firmware(a1)
        frame, how = Frame(b1.device_id, JUNK), "firmware cannot produce a session key"
    else:
        bridge = stolen_a2_bridge()
        frame, how = Frame(bridge.device_id, bridge.seal(a1.device_id, b"x")), "bridge holds a stolen key"
    return Outcome(design, capability, rx.accept_hbw(frame), how)


def bypass_matrix() -> list[Outcome]:
    return [attempt(d, c) for d in Design for c in Capability]
