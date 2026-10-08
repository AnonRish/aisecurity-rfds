"""Licenses, the fault-injection harness and the Ed25519 behaviour the replay logs rely on."""

from __future__ import annotations

import dataclasses

import pytest

from aisecurity_rfds.common.signing import KeyPair, verify
from aisecurity_rfds.rfd_034_license_replay_prevention.hardware import ChipNvm, FaultPlan, PowerLoss
from aisecurity_rfds.rfd_034_license_replay_prevention.licensing import License
from aisecurity_rfds.rfd_034_license_replay_prevention.scenarios import BUDGET, DEVICE_ID, default_authority

ED25519_ORDER = 2**252 + 27742317777372353535851937790883648493


def _license():
    authority = default_authority()
    return authority, authority.issue(DEVICE_ID, sequence=7, budget_flops=BUDGET, challenge=b"c")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_id", b"OTHER"),
        ("sequence", 8),
        ("license_id", bytes(16)),
        ("budget_flops", BUDGET + 1),
        ("challenge", b"d"),
    ],
)
def test_signature_covers_every_field(field, value):
    authority, lic = _license()
    assert verify(authority.public_key, lic.content_digest(), lic.signature)
    forged = dataclasses.replace(lic, **{field: value})
    assert not verify(authority.public_key, forged.content_digest(), forged.signature)


def test_content_digest_ignores_signature_but_full_digest_depends_on_it():
    _, lic = _license()
    padded = lic.with_signature(lic.signature + b"\x00")
    assert padded.content_digest() == lic.content_digest()
    assert padded.full_digest() != lic.full_digest()


def test_digest_encoding_is_injective_across_field_boundaries():
    a = License(b"ab", 1, bytes(16), 1, b"c", b"")
    b = License(b"a", 1, bytes(16), 1, b"bc", b"")
    assert a.content_digest() != b.content_digest()


def test_authority_validates_its_inputs():
    authority = default_authority()
    with pytest.raises(ValueError):
        authority.issue(DEVICE_ID, sequence=-1, budget_flops=1)
    with pytest.raises(ValueError):
        authority.issue(DEVICE_ID, sequence=1, budget_flops=-1)
    with pytest.raises(ValueError):
        authority.issue(DEVICE_ID, sequence=1, budget_flops=1, license_id=b"short")


def test_issuance_is_deterministic_for_a_given_seed():
    first = [default_authority(3).issue(DEVICE_ID, sequence=s, budget_flops=1).license_id for s in range(3)]
    second = [default_authority(3).issue(DEVICE_ID, sequence=s, budget_flops=1).license_id for s in range(3)]
    other = [default_authority(4).issue(DEVICE_ID, sequence=s, budget_flops=1).license_id for s in range(3)]
    assert first == second and first != other


def test_strict_ed25519_rejects_malleated_scalar_and_padded_signatures():
    """Pins the behaviour of the installed `cryptography`: if it ever accepts these, the hash-log scope matters."""
    key = KeyPair.from_seed(b"k" * 32)
    message = b"license digest"
    signature = key.sign(message)
    s = int.from_bytes(signature[32:], "little")
    malleated = signature[:32] + (s + ED25519_ORDER).to_bytes(32, "little")
    assert verify(key.public_bytes, message, signature)
    assert not verify(key.public_bytes, message, malleated)
    assert not verify(key.public_bytes, message, signature + b"\x00")


def test_fault_plan_validates_its_arguments():
    plan = FaultPlan()
    with pytest.raises(ValueError):
        plan.arm(0, "bogus")
    with pytest.raises(ValueError):
        plan.arm(0, "torn")
    with pytest.raises(ValueError):
        plan.arm(-1)


def test_torn_flash_write_keeps_only_a_prefix_of_the_new_value():
    nvm = ChipNvm()
    nvm.plan.arm(0, "torn", 3)
    with pytest.raises(PowerLoss):
        nvm.flash.write("k", b"ABCDEFGH")
    assert nvm.flash.read("k") == b"ABC"


def test_power_loss_before_a_write_leaves_flash_untouched_and_after_completes_it():
    nvm = ChipNvm()
    nvm.plan.arm(0, "before")
    with pytest.raises(PowerLoss):
        nvm.flash.write("k", b"new")
    assert nvm.flash.read("k") is None
    nvm.plan.arm(0, "after")
    with pytest.raises(PowerLoss):
        nvm.flash.write("k", b"new")
    assert nvm.flash.read("k") == b"new"


def test_hv_disruption_silently_drops_flash_writes_and_fuse_burns():
    nvm = ChipNvm()
    nvm.set_hv_disrupted(True)
    nvm.flash.write("k", b"v")
    nvm.otp.increment(verify=False)
    assert nvm.flash.read("k") is None and nvm.otp.value() == 0
    nvm.set_hv_disrupted(False)
    nvm.flash.write("k", b"v")
    nvm.otp.increment(verify=False)
    assert nvm.flash.read("k") == b"v" and nvm.otp.value() == 1


def test_fault_plan_logs_every_persistent_operation_with_its_kind():
    nvm = ChipNvm()
    nvm.flash.write("a", b"12345")
    nvm.otp.increment()
    assert nvm.plan.log == [("flash", 5), ("fuse", 1)]
