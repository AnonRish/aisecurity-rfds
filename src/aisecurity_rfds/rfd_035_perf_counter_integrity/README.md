# RFD 035 - Performance Counter Integrity Protection

Upstream: <https://projects.aisecurity.forum/rfd/035/> (RAND Corporation, state: idea).

## Status

**Kind: simulation** (synthetic telemetry with assumed constants), plus one part that is working protocol code: the
signed readout in `readout.py` uses real Ed25519 from the shared library. The RFD lists six ways to corrupt the
performance counters a compute meter would rely on (EM pulse, voltage glitch, laser, FIB wire cut, unsigned firmware,
aggregation) and six defences (distributed redundancy, sanity checking, physical shielding, distributed aggregation,
counter monotonicity, secure readout). This package builds a model of per-SM counters and chip power, implements all
six attacks as edits of honest readings, implements eight verifier-side checks, measures how much an attacker who knows
those checks can still hide, runs a signed, hash-chained readout protocol with authorised resets against firmware-level
and physical attackers, explores that protocol's code exhaustively at small bounds, and works out redundancy geometry
and some overhead arithmetic. Physical shielding is not modelled. Nothing was measured on a GPU: every detection rate
and hideable fraction below describes the model.

## What is implemented

- `model.py`: synthetic telemetry. 32 SMs with `issued`, `retired`, `flops`, `mem` and `miss` counters, two
  aggregators that each sum a disjoint half of the SMs, and chip power = 100 W idle + 1.5e-3 W per FLOP + 2e-2 W per
  memory access with 2% Gaussian measurement noise. Three workload classes (dense-matmul, decode-memory-bound,
  elementwise) and one class kept out of calibration (`sparse-graph`). Every constant is an assumption.
- `checks.py`: the eight checks (`issued>=retired`, `accesses>=misses`, `throughput-bound`, FLOP- and
  memory-per-instruction bands, robust peer comparison, aggregator cross-check, power consistency), band calibration
  with a margin, an idle-SM rule (an SM that retired nothing cannot have counted FLOPs or accesses), and closed forms
  for the power check. The power statistic is normalised by the *predicted* power, which makes it exactly Gaussian on
  honest data: a two-sided false-positive rate of 2 * Phi(-3.72), about 2e-4 per interval.
- `attacks.py`: the six attack vectors of the RFD, the detection matrix, and the exact probability that the power check
  fires against consistent under-reporting.
- `hiding.py`: an attacker who knows the thresholds and the power model, under-reports FLOPs, adds fake memory traffic
  that holds predicted power constant (K_FLOP / K_MEM = 0.075 accesses per hidden FLOP) and rescales the retired-
  instruction counter to stay inside the bands; a closed form for the largest hideable fraction, checked against a
  brute-force search through the real checks; class-agnostic versus class-aware bands; the band-margin trade-off.
- `readout.py`: `ReadoutBlock` (holds the device key, samples the registers, signs device, epoch, sequence, tick,
  closing flag, counters, previous-readout hash and any ticket), `ResetAuthority` (signed `ResetTicket`s bound to one
  device and one epoch with an expiry), `ReadoutVerifier` (signature, sequence, hash chain, time, epoch, modular delta
  plausibility, tickets) and `protocol_overhead`.
- `device_sim.py`: fourteen end-to-end scenarios of a device running 40 intervals.
- `monotonic.py`: `explore`, a depth-first search over every action sequence of the real block, verifier and authority
  code with invariant checks after each action, plus counter-width helpers.
- `redundancy.py`: `RedundantCounter` with exhaustive fault subsets and a geometric EMP Monte Carlo.

## Reproduce

```bash
pytest tests/rfd_035_perf_counter_integrity
python -m aisecurity_rfds.rfd_035_perf_counter_integrity   # prints every table quoted here (about 12 s)
```

## Findings

Section numbers `[n]` refer to the entry point's output. Thresholds are calibrated on 300 synthetic honest readings
with a 5% margin unless stated.

1. **Each of the six RFD attacks was flagged on every one of 300 attacked readings, but by different checks.** No single
   check covers them (section `[2]`; `test_every_attack_is_flagged_by_some_check_almost_always`). Laser scaling of 4 SMs
   is caught by the peer comparison (100%) and not by power
   (`test_laser_scaling_on_a_few_sms_is_caught_by_peer_comparison_not_by_power`); a FIB wire cut by the FLOP band and
   the peer check (100% each); aggregator tampering only by the aggregator cross-check
   (`test_aggregator_tampering_is_caught_only_by_the_cross_check`); four random bit flips by seven checks at once, the
   best of them (peer comparison) on 78.3%.
2. **Clearing the registers of whole SMs makes them look idle, so only power sees it.** A glitch that clears all counters
   of half the SMs was flagged 100% of the time, by `power-consistency` alone
   (`test_clearing_whole_sms_looks_idle_to_every_check_except_power`). Clearing only the FLOP register is caught by the
   band and peer checks. The idle-SM rule is what stops fake traffic being parked on SMs that report no work (`[5]`,
   `test_fake_traffic_on_idle_sms_is_caught_only_by_the_idle_sm_invariant`).
3. **EMP bit flips are caught often but not always, and the ones that slip through hide almost nothing.** One random flip
   per reading was flagged 82.5% of 1500 readings, two flips 96.9% (`[2]`); the largest FLOP fraction hidden by an
   unflagged flip was 0.3% and 0.4% (`test_emp_flips_that_evade_every_check_hide_almost_nothing`,
   `test_more_emp_flips_raise_the_detection_rate`). The peer check tolerates roughly a 12% deviation of one SM's
   FLOP-per-instruction ratio from the median (six robust deviations, 2% floor); an SM under-reporting by 5% was
   flagged less than 10% of the time and hid under 0.2% of the chip's FLOPs
   (`test_peer_check_bounds_per_sm_hiding_to_about_a_dozen_percent`).
4. **Scaling every counter on every SM (unsigned firmware) is invisible to all structural checks; only power can see it,
   and its power is set by measurement noise.** With 2% assumed noise, a 10% under-count was flagged on 60.7%
   (dense-matmul), 45.0% (decode) and 34.7% (elementwise) of single intervals against exact-formula values of 60.5%,
   44.4% and 36.6%; 5% was flagged on 2.5% to 5.3% (`[3]`,
   `test_consistent_scaling_detection_rate_matches_the_exact_formula`). The 50% detection point is 9.3% to 11.0% of the
   hideable dynamic power for one interval, 1.6% to 1.8% for 40 intervals, and 7.5% to 8.8% for 40 intervals if the
   noise were 10%. A constant 1% under-count escapes a 40-interval test with probability 91.1%, 0.5% with 99.4%, and 5%
   with 0.0% (`[6]`). Small constant under-counts are visible only over many intervals, and only if the power model is
   as accurate as assumed here.
5. **An attacker who knows the thresholds can hide a lot behind class-agnostic bands and little behind class-aware
   ones.** Hideable FLOP fraction with bands calibrated on all workloads: 88.6% (dense-matmul), 60.2% (decode), 29.5%
   (elementwise); with bands calibrated on the declared workload only: 3.3%, 12.2%, 13.9% (`[4]`). The closed form and
   the brute-force search through the real checks agree to under 0.2 percentage points
   (`test_closed_form_matches_brute_force_search_on_ideal_readings`), the attack passes every check and hides what it
   claims (`test_feasible_attack_passes_every_check_and_hides_what_it_claims`), and without the power-matching fake
   traffic the same hiding is caught by power (`test_without_power_matching_the_same_hiding_is_caught_by_power`).
6. **Tight bands flag honest workloads they were not calibrated on, and widening them buys the attacker room.** No
   honest false positive on 300 fresh readings of the calibration classes, but the unseen `sparse-graph` workload was
   flagged 100 of 100 times (`[1]`, `test_the_unseen_class_is_rejected_by_default_bands_on_the_flop_ratio`). Widening
   the margin to 0.60 still flags 98% of it and 0.70 flags none, while the dense-matmul hideable fraction rises from
   87.5% at margin 0 to 97.6% at 0.70 and 99.3% at 0.90 (`[5]`). In this model no margin accepts the unseen workload and
   keeps hiding small. A flag therefore means "outside what was calibrated", not "attack"
   (`test_unattacked_novel_workload_is_flagged_so_flags_alone_do_not_mean_attack`).
7. **The signed, hash-chained readout stops forgery, replay, dropping, rollback of the block and unauthorised resets at
   the channel.** Editing counters in transit is rejected as `bad-signature`, a replayed readout as `replay`, a dropped
   one as `gap`, a forged reset ticket is refused by the block (`bad-ticket-signature`), a used ticket cannot be reused
   in a later epoch (`ticket-epoch-mismatch`) and an expired one is refused (`ticket-expired`); a glitch that clears the
   registers is rejected as an `implausible-delta` (`[6]`, `test_scenario_outcome`,
   `test_rolled_back_or_cloned_block_breaks_the_hash_chain`).
8. **It does not protect the sensor.** A clock-gating stall from interval 20 was accepted by the protocol and caught
   only by the power cross-check (z = 449.3); a constant 1% under-count was accepted and missed (`[6]`). A glitch timed
   so that the register regains its old value is also accepted
   (`test_glitch_timed_to_regain_the_old_value_is_accepted_documented_limitation`).
9. **Legitimate resets lose no work.** A ticketed reset emits a closing readout with the old epoch's final counters,
   zeroes the registers and bumps the epoch; credited work equals true work across the reset, also when a hoarded
   ticket is used late (`test_legitimate_reset_loses_no_work_and_starts_a_new_epoch`,
   `test_a_hoarded_ticket_still_yields_a_closing_readout_with_the_true_final_counters`). The epoch cannot advance without
   a closing readout (`test_epoch_cannot_advance_without_a_closing_readout`).
10. **Bounded exhaustive exploration found no invariant violation and does find the protocol's blind spot.** With ideal
    signatures, 8-bit counters and at most 2 units of work per tick, the hiding alphabet (work, read, glitch, stall)
    at depth 8 explored 470,880 transitions with 84,462 readouts accepted and 12,057 mature windows; the protocol
    alphabet (adds replay, dropped readouts, forged, replayed and future tickets, legitimate reset) at depth 6 explored
    319,718 transitions with 46,206 readouts accepted; both had 0 violations of: no over-credit, no retroactive
    erasure, conservation of work, no accepted regression, epochs advance only through authorised resets, no replay
    accepted twice, refused tickets leave the block unchanged (`[7]`). The same search reports that a whole mature
    window of work can be hidden and accepted; the shortest such trace is
    `work1 -> read -> glitch_clear -> work1 -> read`. Five deliberately broken variants (replays accepted, any ticket
    accepted, no plausibility check, epoch forgotten on reset, huge rate bound) are caught by the same invariants, so
    the search can fail (tests `test_a_verifier_that_accepts_replays_is_caught_by_the_exploration` to
    `test_a_verifier_with_a_huge_rate_bound_accepts_regressions`).
11. **Redundant counters help against local faults only if the copies are physically spread.** Exhaustively over every
    subset of copies: with k=3 a fault reaching one copy is always detected and recovered, two copies always detected but
    not recovered, three not detected; with k=5 up to two copies are recovered, three or four detected but not
    recovered, five not detected (`[8]`, `test_any_fault_short_of_all_copies_is_detected_and_majority_recovers_below_half`).
    Against a random EM disc of radius 0.15 on a unit die, a single copy leaves about 7% undetected corruptions (6.9%
    clustered, 7.0% spread), two or three copies clustered together still 5.7% and 5.6%, two or three spread copies
    0.0% (14.2% and 21.0% detected);
    a larger pulse defeats the spread layout too (`test_a_large_enough_pulse_defeats_even_spread_copies`). A supply glitch
    reaches every copy sharing the supply; whether copies sit in independent power domains is a physical-design
    question this code cannot test.
12. **Protocol overhead and counter width are plain arithmetic.** A readout is
    145 B (241 B with a ticket); 100,000 devices at one readout per second is 14.5 MB/s and 100,000 signature
    verifications per second. A 64-bit FLOP counter at 10^15 FLOP/s wraps after 5.12 hours (0.51 hours at 10^16);
    counting 10^26 FLOPs needs 87 bits; with an 8-bit counter growing at most 2 per tick, readouts must be at most 127
    ticks apart, and if 600 units accumulate between two reads, 512 alias away (`[9]`,
    `test_counter_width_arithmetic`, `test_safe_gap_and_aliasing_are_consistent`).

## Open questions

1. *"What's the overhead (area, power) of redundant counter infrastructure?"* **Not addressed.** Area and power need
   silicon. Only the communication and verification cost of the readout protocol is computed (finding 12), which is
   a different quantity.
2. *"Can existing GPU counter hardware be hardened, or is a redesign needed?"* **Not addressed.** The model shows what
   each defence assumes (a signing block that firmware cannot reach, a power sensor the attacker cannot edit, counters
   in separate regions), not whether any real GPU provides it.
3. *"How to handle legitimate counter resets (e.g., after chip RMA/repair)?"* **Answered in model.** Authorised,
   device- and epoch-bound, expiring tickets, a closing readout, and a hash chain that continues across epochs
   (findings 7 and 9). Who issues tickets for an RMA, and how a repaired chip is re-attested, is a governance question
   left open.
4. *"What's the false positive rate for sanity checks under normal operation?"* **Partially.** In the model: 0 of 300
   on calibrated workloads, 100 of 100 on an unseen workload, 2e-4 per interval for the power check by construction
   (findings 6 and 4). The rate on real workloads depends on how varied they are, which is not known here.
5. *"Can formal verification ensure counter logic has no hidden reset paths?"* **Partially.** Exhaustive exploration of
   the protocol code at small bounds, with mutation tests showing the invariants can fail (finding 10). That is not a
   proof, and it says nothing about hardware counter logic.

## Not established

- Every telemetry value, power coefficient, noise level and workload class is an assumption in `model.py`; nothing was
  measured on a GPU, so detection rates and hideable fractions describe this model, not any chip.
- The power cross-check assumes the verifier knows the power model exactly and that power is sensed by a path the
  attacker cannot touch. Model error and sensor tampering are probed only through the sigma-10% column of `[3]`.
- The fault-injection techniques (EM pulse, voltage glitch, laser, FIB) and the shielding defences are not modelled
  physically; the attacks are abstract edits of readings and the EMP disc is a geometric stand-in.
- The exploration covers this Python implementation at tiny widths and depth with an ideal signature scheme. It is
  bounded model checking, not a verification of counter hardware or of Ed25519.
- The protocol protects the channel only: a sensor that counts less (stall, cut wire, glitch timed to regain the old
  value) yields validly signed, plausible readouts, and only cross-checks can see it.
- The signing block is assumed to be unreachable by firmware and to hold its key in tamper-protected storage; how to
  build that is not addressed here.
- The hiding attacker knows the thresholds and the power model. An adaptive attacker that learns them, or a verifier
  with randomised thresholds, is not modelled.
- Area and power overhead of redundant counters and of a signing block are not estimated.

## Attacks tested

| Attack (RFD difficulty) | Defence under test | Outcome in the model |
| --- | --- | --- |
| EM pulse, random bit flips (Medium) | structural checks | 1 flip caught 82.5%, 2 flips 96.9%, 4 flips 100% of 300; missed flips hide at most 0.4% of FLOPs |
| Voltage glitch clears half the SMs (Medium) | power cross-check | caught 100%, by power alone; structural checks see idle SMs |
| Laser scaling of FLOPs on 4 SMs (High) | peer comparison | caught 100% |
| FIB wire cut on 4 SMs (High) | FLOP band, peer comparison | caught 100% |
| Unsigned firmware scales all counters (Low) | power cross-check | -30% caught 100%; -10% caught 35% to 61% per interval; 1% missed with probability 91.1% over 40 intervals |
| Aggregator under-reports (Medium) | aggregator cross-check | caught 100% |
| Power-matched hiding with fake memory traffic | bands | missed: 88.6%, 60.2%, 29.5% hideable (class-agnostic), 3.3%, 12.2%, 13.9% (class-aware) |
| Fake traffic parked on idle SMs | idle-SM rule | caught (memory-per-instruction band) |
| Edited counters, forged readout | signature | caught (`bad-signature`) |
| Replayed or dropped readout | sequence number, hash chain | caught (`replay`, `gap`) |
| Rolled-back or cloned block | hash chain | caught |
| Forged, replayed, expired or wrong-epoch reset ticket | block and verifier | refused, block left unchanged |
| Glitch that clears registers | modular delta plausibility | caught (`implausible-delta`) |
| Clock-gating stall | power cross-check | protocol accepts; power catches (z = 449.3) |
| Glitch timed to regain the old value; stall | protocol | missed by the protocol (documented limitation, shortest trace in finding 10) |
| Constant 1% under-count | power cross-check | missed (z = 1.1; 91.1% escape over 40 intervals) |

## Related RFDs

- RFD 032 (Offline Licensing for Compute Rationing): a licence meter is only as good as the counters it reads.
- RFD 034 (License Replay Prevention): the same rollback and reset threats against persistent state.
- RFD 024 (Distributed FLOP Counting): any FLOP count rests on counters like these; not yet in this repository.
- RFD 012 (Analog Sensors for Compute Verification): the power cross-check here is one analog signal checked against
  digital counters; not yet in this repository.
- RFD 036 (Tamper-Respondent PUF Enclosures): shielding and key protection for the signing block are not modelled here.
