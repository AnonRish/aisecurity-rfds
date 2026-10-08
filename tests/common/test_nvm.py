import pytest

from aisecurity_rfds.common.nvm import (
    BatteryBackedSram,
    CounterExhaustedError,
    CounterTamperedError,
    FlashStore,
    JournaledRecord,
    NvmError,
    OtpFuseBank,
    UnaryCounter,
    WriteVerifyError,
)


def test_otp_bits_only_go_one_way():
    bank = OtpFuseBank(8)
    bank.burn(3)
    assert bank.read(3) == 1 and bank.burned() == 1
    bank.burn(3)  # burning again keeps it at 1; there is no API to clear a fuse
    assert bank.read(3) == 1


def test_high_voltage_disruption_is_silent_unless_verified():
    bank = OtpFuseBank(8)
    bank.hv_disrupted = True
    bank.burn(0)  # silently does nothing: the attack in RFD 034
    assert bank.read(0) == 0 and bank.burn_count == 0
    with pytest.raises(WriteVerifyError):
        bank.burn(0, verify=True)  # read-after-write defence


def test_unary_counter_counts_and_exhausts():
    counter = UnaryCounter(OtpFuseBank(5))
    assert counter.value() == 0
    assert [counter.increment() for _ in range(5)] == [1, 2, 3, 4, 5]
    with pytest.raises(CounterExhaustedError):
        counter.increment()


def test_unary_counter_detects_holes():
    bank = OtpFuseBank(8)
    counter = UnaryCounter(bank)
    counter.increment()
    bank.burn(5)  # attacker or partial-burn artefact leaves a hole at 1..4
    with pytest.raises(CounterTamperedError):
        counter.value()


def test_counter_increment_under_disrupted_hv_attack_and_defence():
    bank = OtpFuseBank(8)
    counter = UnaryCounter(bank)
    counter.increment()
    bank.hv_disrupted = True
    assert counter.increment(verify=False) == 2  # claims success...
    assert counter.value() == 1  # ...but nothing was recorded (replay-protection bypass)
    with pytest.raises(WriteVerifyError):
        counter.increment(verify=True)  # defence: read back before enabling the licence


def test_counter_window_offset_and_bounds():
    bank = OtpFuseBank(10)
    counter = UnaryCounter(bank, offset=4, capacity=3)
    counter.increment()
    assert bank.read(4) == 1 and bank.read(0) == 0 and counter.capacity == 3
    with pytest.raises(ValueError):
        UnaryCounter(bank, offset=8, capacity=5)


def test_flash_torn_write_mixes_old_and_new():
    flash = FlashStore()
    flash.write("k", b"AAAAAAAA")
    flash.write("k", b"BBBBBBBB", torn_at=3)
    assert flash.read("k") == b"BBBAAAAA"


def test_flash_rollback_attack_primitive():
    flash = FlashStore()
    flash.write("counter", b"\x01")
    snap = flash.snapshot()
    flash.write("counter", b"\x09")
    flash.physical_restore(snap)
    assert flash.read("counter") == b"\x01"


def test_journaled_record_survives_torn_writes_at_every_position():
    payload_v1, payload_v2 = b"version-one", b"version-two!"
    probe = JournaledRecord(FlashStore(), "probe")
    probe.write(payload_v2)
    max_len = len(FlashStore().read("none") or b"") + 12 + len(payload_v2) + 4  # header + payload + crc
    for torn_at in range(0, max_len + 1):
        store = FlashStore()
        rec = JournaledRecord(store, "r")
        rec.write(payload_v1)
        rec.write(payload_v2, torn_at=torn_at)
        # Either the new value fully landed or the old one is intact: never garbage, never None.
        assert rec.read() in (payload_v1, payload_v2), torn_at


def test_journaled_record_alternates_slots_and_handles_first_write_tear():
    store = FlashStore()
    rec = JournaledRecord(store, "r")
    assert rec.read() is None
    rec.write(b"a", torn_at=2)
    assert rec.read() is None  # nothing was ever committed
    rec.write(b"b")
    rec.write(b"c")
    rec.write(b"d")
    assert rec.read() == b"d"
    assert store.read("r.A") is not None and store.read("r.B") is not None


def test_battery_backed_sram_loses_state_without_battery():
    sram = BatteryBackedSram()
    sram.write("k", b"secret")
    assert sram.read("k") == b"secret"
    sram.remove_battery()
    assert sram.read("k") is None
    with pytest.raises(NvmError):
        sram.write("k", b"x")
    sram.replace_battery()
    assert sram.read("k") is None
