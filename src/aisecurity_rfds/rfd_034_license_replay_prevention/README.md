# RFD 034 - License Replay Prevention

Upstream: <https://projects.aisecurity.forum/rfd/034/> (RAND Corporation, state: idea).

## Status

**Kind: reference-implementation** (a software model on simulated memory). The RFD proposes three ways to stop a
saved, validly signed license from being applied twice: a hash log, sequence numbers and a chip-generated
challenge, plus a hybrid and an external commitment option. This package implements all of them against a
fault-injectable model of OTP fuses and flash, and implements the attacks the RFD names (rollback, high-voltage
disruption, power loss during a write). It compares the designs; it does not pick a winner for real hardware,
and it says nothing about how real fuses or flash behave.

## What is implemented

- `licensing.py`: `License` (Ed25519-signed device id, sequence, id, budget, challenge) and `LicenseAuthority`.
- `hardware.py`: `ChipNvm` (flash + fuses + a unary fuse counter), `FaultPlan` (power loss before/torn/after any
  numbered write), high-voltage disruption, flash imaging and restore.
- `guards.py`: `HashLogGuard` (optionally a ring; digest of signed content or of the whole license),
  `SequenceGuard` (optionally with an OTP rollback floor of one fuse per `stride` numbers), `HybridGuard`
  (sequence window plus recent-id log), `ChallengeGuard` (optionally bound to the OTP counter).
- `ledger.py`: signed `(sequence, timestamp)` commitments and an `ExternalLedger` that flags regressions.
- `scenarios.py`: every attack, the exhaustive power-loss enumeration and the trade-off sweeps below.

## Reproduce

```bash
pytest tests/rfd_034_license_replay_prevention
python -m aisecurity_rfds.rfd_034_license_replay_prevention   # prints every table quoted here (about 4 s)
```

## Findings

Every claim names the test or `__main__` section that backs it. C1-C8 are: hash-log, hash-log/ring4, sequence,
sequence+OTP(S=4), hybrid(W=4), hybrid(W=4)+OTP(S=4), challenge, challenge+OTP-epoch.

1. **Plain replay and replay after a power cycle are blocked by all eight designs** (`test_matrix_matches_the_outcomes_derived_from_each_design`).
2. **A flash rollback defeats every design except the challenge guard bound to the OTP counter.** Replays accepted
   out of 10 credited licenses: 10 for C1, C2, C3 and C5; 2 for C4; 7 for C6; 1 for C7; 0 for C8
   (`test_flash_rollback_replay_counts_follow_the_derivation`). The RFD discusses rollback for the sequence and hash
   approaches but not for the challenge approach, whose pending challenge is also restored by an old flash image.
3. **The OTP floor trades fuses for a bounded replay window.** After 32 licenses, a stride-S floor uses 32/S fuses and
   leaves at most S-1 licenses replayable; for the hybrid it leaves W+S (section 5 of the entry point;
   `test_sequence_floor_trades_one_fuse_per_stride_for_a_replay_window_of_stride_minus_one`,
   `test_hybrid_floor_replay_window_is_window_plus_stride`). The hybrid's tolerance for late licenses is paid for in
   rollback exposure.
4. **High-voltage disruption bypasses all eight designs if budget is credited before the write is read back, and
   none if it is read back first** (`test_hv_disruption_is_defeated_only_by_reading_back_before_crediting`). Read-back
   turns the attack into a lost license, not a free one (`test_fuse_only_disruption_with_read_back_turns_the_attack_into_a_lost_license`).
5. **Power loss never double-credits a license in any of the 17-75 crash points enumerated per design** when the
   commit is atomic (journaled) and crediting happens after it. A license is lost at exactly 1 crash point for
   designs without fuses and 3 with (`test_no_power_loss_point_lets_any_license_be_credited_twice`,
   `test_a_license_is_lost_only_when_the_commit_was_already_durable`). A naive raw little-endian counter in flash
   goes backwards at exactly one torn-write offset, so the enumeration can detect a failure
   (`test_non_atomic_counter_regresses_under_exactly_one_torn_write_offset`).
6. **A bounded hash log forgets**: with a ring of 4, the oldest of 6 licenses is accepted again, as the RFD warns.
7. **Ordering costs liveness.** Delivered as 3, 1, 2, the sequence guard keeps 1 of 3 licenses; hash-log and hybrid keep
   all 3. With delivery shuffled by up to D places, the sequence guard strands 12.5%, 35.8% and 52.8% of licenses at
   D = 2, 4, 8; hybrid(W=4) strands 0%, 0%, 9.0%; a window at least as large as D strands none (`test_hybrid_window_at_least_the_displacement_strands_nothing_and_smaller_windows_do`).
8. **Key the log on the signed content, not on the license as transmitted.** With a verifier that tolerates padding,
   a full-license digest is bypassed and a content digest is not. The installed Ed25519 verifier rejects both
   padding and scalar malleation, pinned by `test_strict_ed25519_rejects_malleated_scalar_and_padded_signatures`.
9. **Challenge approach**: a repeating random-number generator replays licenses unless the challenge is bound to the
   OTP counter; stockpiling 5 licenses works for every design except the epoch-bound challenge, which allows 1;
   the price is one fuse per license, so the model's 512-fuse bank allows 512 licenses
   (`test_challenge_epoch_exhausts_after_one_fuse_per_license`).
10. **External commitments detect only a careless attacker.** After a rollback and 10 replays the attacker holds 25
    credited licenses against 15 granted; the ledger flags a regression only if the replay's commitments are
    reported (`test_ledger_does_not_flag_a_careful_attacker_who_withholds_replay_commitments`).
11. **Storage**: 32 bytes per license (+8) for the hash log, 8 for a sequence counter, `8 + 24*(W+1)` for the hybrid
    (the RFD's pseudo-code needs the sequence stored with each id to prune, so 24 bytes per entry, not 16), and
    double-slot journaling doubles the physical flash footprint (`test_storage_follows_the_rfd_formulas_for_each_approach`).

## Open questions

1. *"What's the expected license volume over a chip's lifetime?"* **Partially.** The volume is a policy input; the model
   gives sizing per license (finding 11, and one fuse per license for the epoch-bound challenge).
2. *"How to handle legitimate scenarios where licenses are issued but never delivered?"* **Partially.** Undelivered
   licenses are harmless to hash-log and hybrid, strand later-numbered licenses in the sequence guard, and are voided
   by any new challenge request in the challenge guard (`test_requesting_a_new_challenge_invalidates_the_previous_outstanding_license`).
3. *"Can sequence number gaps be allowed for fault tolerance without enabling attacks?"* **Answered in model.** Gaps do
   not enable replay; they strand the skipped numbers, and under an OTP floor cost one fuse per stride crossed
   (`test_sequence_gaps_are_safe_for_replay_but_cost_fuses_under_an_otp_floor`). One signed far-future number
   strands every lower license (`test_one_signed_far_future_sequence_number_permanently_strands_lower_licenses`).
4. *"What's the cost of OTP fuses for replay-critical state?"* **Partially.** Fuse counts per design are measured
   (finding 3); silicon cost needs real hardware.
5. *"How to recover from accidental counter corruption without enabling reset attacks?"* **Not addressed.** The
   unary counter detects holes (`CounterTamperedError` in the shared library) but no recovery protocol is modelled.

## Not established

- Nothing here was measured on real OTP fuses, flash or a hardware random-number generator; every result is a
  property of the simulation model in `hardware.py`.
- The license authority, its key management and the governance process for budgets are out of scope; stockpiling
  before a policy change is characterised, not prevented.
- Fault injection beyond power loss, high-voltage starvation and flash imaging (voltage glitches on the signature
  check, invasive probing) was not modelled.
- The commitment option is modelled as signed `(sequence, timestamp)` records; clock integrity and what an audit can
  force a tampered chip to reveal are assumptions, not results.

## Attacks tested

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Same license twice, also across a power cycle | any guard | caught by all 8 |
| Flash image restored, 10 licenses replayed | OTP floor / OTP-bound challenge | caught only by C8; partial for C4 (2/10), C6 (7/10), C7 (1/10); missed by C1, C2, C3, C5 |
| High-voltage supply starved while applying | read-back before crediting | caught with read-back, missed without, for all 8 |
| Power loss at every write, before/torn/after | journaled atomic commit, commit before credit | caught at all points (license lost at 1 or 3); naive counter missed at 1 offset |
| Unsigned license with sequence 2^40 | signature check before any state change | caught by all 8 |
| Oldest of 6 replayed against a ring of 4 | none possible | missed by C2 (documented RFD limitation) |
| Stockpile 5 licenses, apply later | one outstanding challenge bound to the OTP epoch | caught only by C8 |
| Padded signature on a replayed license | hash the signed content | caught; the full-license digest is missed under a lenient verifier |
| Rollback with commitments withheld | external ledger | missed; flagged only when the replay is reported |

## Related RFDs

- RFD 032 (Offline Licensing): defines the license whose replay this prevents.
- RFD 035 (Performance Counter Integrity): the meter counters face the same reset and rollback threats.
- RFD 036 (PUF Enclosures): PUF-protected NVRAM appears in this RFD's storage options; not modelled here.
