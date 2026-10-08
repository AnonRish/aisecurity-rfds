# RFD 032 - Offline Licensing for Compute Rationing

Upstream: <https://projects.aisecurity.forum/rfd/032/> (RAND Corporation, state: idea).

## Status

**Kind: reference-implementation** for the on-chip license protocol, plus simulations whose parameters are
assumptions. The RFD proposes chips that ship throttled and run at full speed only while signed, device-specific
licenses keep on-chip meters above zero. This package implements the verifier and meter block on real Ed25519 with
adversarial tests, and adds small models of the throttle actions, of what happens when a job outlives its license,
and of stockpiling. The protocol results are properties of the code. The throttle and policy numbers are properties
of the stated assumptions and say nothing about real GPUs. Replay and rollback protection is deliberately simple
here (a sequence number in flash); RFD 034 is the detailed treatment and shows where that is weak.

## What is implemented

- `licensing.py`: `ComputeLicense` (serial, sequence, `add`/`set` mode, per-meter budgets) and `LicenseAuthority`.
- `chip.py`: `OfflineLicenseChip`: signature, serial binding, sequence, canonical-form, overflow and balance-cap
  checks applied atomically; per-meter enforcement; fail-safe checkpointing across power cycles.
- `throttle.py`: the RFD's four throttle actions, a `HardwareProfile` of assumed parameters and `run_job`.
- `policy_sim.py`: mid-job exhaustion policies, stockpiling against the chip model, power-cycle experiment.

## Reproduce

```bash
pytest tests/rfd_032_offline_licensing
python -m aisecurity_rfds.rfd_032_offline_licensing   # prints every table quoted here
```

## Findings

1. **Protocol checks hold, atomically.** A license for another serial (`wrong-device`), an edited budget
   (`bad-signature`), a replay or older sequence (`replay-or-stale`), an unenforced meter (`unknown-meter`),
   duplicate meter entries that were genuinely signed (`malformed`), a top-up past 2^64 (`overflow`) and a top-up above
   the cap (`balance-cap`) are all rejected, and a rejected license changes no state, including when only one of
   its meters is at fault (`test_balance_cap_rejects_the_whole_license_not_just_one_meter`,
   `test_duplicate_meter_entries_are_malformed_not_double_counted`).
2. **Every enforced meter needs funding.** A license that tops up FLOPS but not memory transfer leaves the chip
   throttled (`test_unlicensed_chip_is_throttled_until_every_enforced_meter_is_funded`); the RFD's "multiple meters
   can be active" implies each license must cover each enforced meter.
3. **Checkpoints can be fail-safe.** Writing a checkpoint of `remaining - interval` *before* work that would pass the
   old checkpoint means a power cycle restores a balance that is never larger than the true one. In the sweep (budget
   10^6, 400 random steps) the largest gain was 0, -4 and -112 at intervals 10, 100 and 1000, and the largest loss
   equalled the interval. Torn checkpoint and license writes restore either the old or the new state, never a mix
   (`test_torn_checkpoint_write_never_restores_more_than_the_true_balance`). Writes occur about once per
   `interval + 1` units. As arithmetic with an assumed 10^5 flash writes, the RFD's 10^18-FLOP example needs an
   interval of at least 10^13 FLOPs, losing at most 10^-5 of the budget per power cycle.
4. **Throttle choice decides who is hurt** (assumed profile: 1% clock, tensor-disabled work at 1/16 speed, memory
   bandwidth at 5%). Clock reduction slows gaming, training and inference identically (0.01), so it cannot spare
   non-AI use. Disabling tensor units leaves gaming at 1.0 and AI at 0.0625, so AI keeps running at the assumed ratio.
   A bandwidth limit gives 0.149 for gaming and training and 0.062 for inference. The ratios are assumptions;
   the structure (clock reduction cannot discriminate between workloads) is not (`test_clock_reduction_slows_every_workload_equally_including_gaming`).
5. **Mid-job policy results match closed forms** (job sizes lognormal, sigma 0.5, delay D = 10 job-times: assumptions).
   Pausing at exhaustion gives slowdown (B + D) / B; throttling to baseline gives (B + D) / (B + rD), and the baseline's
   progress matters only when B is comparable to rD. For B = 1, 5, 20, 100 pausing measures 10.998, 2.999, 1.499, 1.099
   against 11, 3, 1.5, 1.1. A hard stop with no checkpointing never finishes jobs larger than a license (48.9% at
   B = 1) and discards 33.1%, 17.3%, 3.7% and 0.8% of work at B = 1, 5, 20, 100. In this model the reissue delay, not the
   baseline speed, determines how bad an interruption is.
6. **A balance cap limits bursts, not totals.** With 5 pre-issued top-ups of 100, the attacker obtains 500 at caps of
   100, 300 and 500, with peak balance equal to the cap (`test_balance_cap_bounds_the_burst_but_not_the_total_a_stockpile_provides`).
   A revocation (`set` to 0 at a higher sequence) gives 0 usable if applied first and 500 if the attacker withholds it.
7. **License size trades overhead for exposure.** With D = 10 job-times, reissue overhead D/B is 100%, 20%, 10%, 2%
   for B = 10, 50, 100, 500, while 3 pre-issued licenses expose 3B.
8. **Known gap**: flash rollback restores the older, larger balance and sequence number
   (`test_flash_rollback_restores_an_older_larger_balance_documented_limitation`).

## Open questions

1. *"Who is the license authority? (Chip vendor? US government? International body?)"* **Out of scope.** Institutional.
2. *"What governance process determines allocation of compute budgets?"* **Out of scope.** Institutional.
3. *"How to prevent license stockpiling before a policy change?"* **Partially.** Finding 6: caps do not bound a drawer of
   licenses and revocation needs delivery. RFD 034 finds that a chip-issued challenge bound to a fuse counter limits
   stockpiling to one outstanding license, at the cost of a round trip per license.
4. *"What's the right granularity for license duration? (Days? Months? Per-job?)"* **Partially.** Finding 7 gives the
   overhead and exposure curves under assumed D; the right point depends on D and on tolerable exposure.
5. *"How to handle legitimate workloads that exceed licensed compute mid-job?"* **Partially.** Finding 5 compares pause,
   baseline throttle and hard stop. Overdraft and cooperative checkpoint signalling were not modelled.
6. *"Can licenses be made non-fungible to prevent gray markets?"* **Partially.** Serial binding makes a license
   unusable on any other chip (tested); a chip with remaining balance can still be sold.
7. *"How to set the throttled baseline without making chips useless for legitimate non-AI uses?"* **Partially.** Finding 4,
   under assumed ratios.

## Not established

- No silicon was involved: throttle speeds, job sizes and reissue delays are assumed parameters, so the policy and
  throttle tables characterise those models, not any GPU or deployment.
- Flash rollback is not defended here; fuse floors and counter integrity are the subject of RFD 034 and RFD 035.
- The license authority, budget allocation and signing-key protection are institutional questions and are not modelled.
- Side-channel extraction, wire-cutting bypass of the throttle and fault injection into meter registers (named in
  the RFD's security table) were not modelled.

## Attacks tested

| Attack | Defence | Outcome |
| --- | --- | --- |
| License moved to another chip | serial inside the signed digest | caught |
| Budget edited after signing | Ed25519 over the full digest | caught |
| Replay or older license | strictly increasing sequence | caught (rollback is not, see finding 8) |
| Duplicate meter entries to double count | canonical-form check | caught |
| Top-up past 2^64 | overflow check | caught |
| Hoarding balance on the chip | per-meter cap | caught for bursts, missed for totals |
| Power cycle to restore a spent balance | checkpoint before work | caught (gain never positive) |
| Torn write during checkpoint or license | journaled record, atomic apply | caught |
| Pre-issued licenses applied after a policy change | cap, set-to-zero revocation | missed unless revocation is delivered first |
| Flash image restored | none here | missed |

## Related RFDs

- RFD 034 (License Replay Prevention): the replay and rollback analysis this package defers to.
- RFD 035 (Performance Counter Integrity): protects the meter registers this package assumes are trustworthy.
- RFD 033 (Fixed Set Cluster Size Limiting): a sibling RAND mechanism that limits aggregation rather than usage.
