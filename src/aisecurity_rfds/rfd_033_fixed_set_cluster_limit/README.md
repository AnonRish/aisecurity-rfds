# RFD 033 - Fixed Set Cluster Size Limiting

Upstream: <https://projects.aisecurity.forum/rfd/033/> (RAND Corporation, state: idea).

## Status

**Kind: reference-implementation** for the membership protocol, the link handshake and the egress limiter (real
signatures and authenticated encryption on simulated fuses and flash), **plus a closed-form model** for the bandwidth
analysis (every input of which is an assumption). The RFD proposes restricting high-bandwidth links to a preauthorised
"pod" of chips and holding everything else to about 1 GB/s. This package builds the pod-formation protocol the RFD
sketches, attacks it, builds the limiter that would enforce the cap, and asks numerically what a cap costs someone who
trains a frontier-scale model anyway. It addresses four of the RFD's six open questions inside the model and none of
them for real hardware. It does not show that a fixed set limit would hold in practice, and it says nothing about
manufacturing cost or retrofit.

## What is implemented

- `pod.py`: `Chip` (device key; a one-way 120-fuse generation counter; an optional pod binding in fuses; a journaled
  flash manifest; boot-time repair; tamper response; links that expire), `Manufacturer` (signs manifests and
  releases; `sign_manifest` enforces nothing, like a careless facility, while `form_pod`, `replace_member` and
  `release_chip` are the disciplined path) and the three-message handshake (`Hello1`, `Hello2`, `Finish`): Ed25519
  signatures over a transcript hash, X25519 key exchange, AES-GCM sessions from `aisecurity_rfds.common`.
- `cluster.py`: fleets of pods, the real handshake as the edge test, connected components of the high-bandwidth
  graph, and `impersonate` / `bridge_cluster_size` for the key-extraction attack.
- `gate.py`: three places to take the "is this pod traffic?" decision (a firmware limiter, handshake then plain
  frames, handshake then AEAD frames) against four attacker capabilities.
- `limiter.py`: an integer token bucket and `ChipPort`, which charges pod peers to one bucket and everything else to
  a single shared external bucket.
- `scenarios.py`: the attack table and the update-window experiment.
- `primitives.py`: tiny executable versions of a ring all-reduce, a counted matrix multiply, a pipeline boundary and a
  decoder block, so the analysis's constants are counted rather than remembered.
- `training.py`, `podsize.py`, `inference.py`: per-step compute and traffic for a model split over K pipeline stages
  and R data-parallel replicas of S-chip pods, break-even caps and ceilings in closed form, a plan search that finds
  the fewest chips meeting a deadline, the "defeat bandwidth" bisection, pod compute and its erosion by algorithmic
  progress, and serving traffic per token.

## Reproduce

```bash
pytest tests/rfd_033_fixed_set_cluster_limit
python -m aisecurity_rfds.rfd_033_fixed_set_cluster_limit   # prints every table and number quoted below (about 8 s)
python scripts/mutation_check.py 33                          # 44 seeded bugs; every one must be caught
```

Section numbers (`§N`) below are the numbered sections of that program's output. Figures in backticks are copied
from it, and a test (`test_rfd033_zz_readme.py`) fails if one is not printed there.

## Findings

Every claim names the test or output section that backs it. All bandwidth results are properties of the model under
the assumptions in §7: chips of `1e+15` FLOP/s at `40%` sustained use with `80 GB` of memory, pods of `64` chips,
a `1e+12`-parameter dense model trained on `2e+13` tokens in one year with `1.6e+07`-token batches, `16` bytes of
training state per parameter, `2`-byte gradients and activations, and perfect overlap of communication with compute.

1. **Membership attacks (§1).** The attack table runs `15` attacks against the protocol code. `12` are blocked, `1`
   is partly blocked and `2` succeed for the attacker. Blocked: a chip of another pod claiming membership, a manifest
   signed by anyone but the manufacturer, one pod's manifest loaded into another pod's chips, a validly signed
   oversize pod offered to size-1 chips, an old manifest replayed after an update, a pre-update flash image restored,
   a flash image restored after a tamper response, a recorded handshake replayed, a chip's own hello reflected back
   at it, an initiator identity swapped in flight, a replayed release, and a flash wipe followed by a manifest for
   another pod on a chip whose binding is in fuses (`test_attack_table_outcomes_are_pinned` pins every row).

2. **Two limitations an attacker can use (shown as passing tests, not fixed).** The tamper response is irreversible
   only if its fuse burn takes effect: with the high-voltage supply starved, the generation does not move and a
   restored flash image brings the chip back into its pod
   (`test_limitation_starving_the_burn_supply_lets_a_flash_restore_undo_a_tamper_response`); without the restore an
   ordinary reboot stays out (`test_a_tamper_response_under_a_starved_burn_supply_still_survives_an_ordinary_reboot`).
   And a chip whose pod binding lives in flash, which is what lets the manufacturer release it for reassignment, can
   have the binding wiped by anyone who can write its flash; a chip bound in fuses refuses but can never be released
   (`test_limitation_a_rolled_back_flash_binding_lets_a_careless_facility_move_a_chip_but_fuses_do_not`).

3. **Updatable membership (open question 3) works in the model, with a window (§2).** A chip accepts a manifest only if
   it verifies under the fused manufacturer key, lists the chip, is no larger than the chip's fused size cap, names the
   pod the chip is bound to, and carries a generation strictly above the one-way fuse counter. Old manifests and
   pre-update flash images are dead (`test_flash_rollback_cannot_bring_back_a_superseded_manifest`), a crash between
   the flash write and the fuse burn is repaired at boot (`test_a_crash_between_the_flash_write_and_the_fuse_burn_is_repaired_at_boot`),
   and a failed member can be replaced (`test_replacing_a_failed_member_isolates_it_once_the_new_manifest_spreads`).
   The window: if the facility updates one member to remove a chip, the removed chip still links with every member
   that has not heard. Right after the update `6`, `14`, `30` and `62` members accept it, in pods of `8`, `16`, `32`
   and `64`. The new manifest spreads inside the handshakes members run anyway (each hello carries the sender's
   manifest; a chip adopts a newer signed manifest that lists it). With one handshake per member per round, every
   member has adopted after a mean of `2.3`, `2.9`, `3.8` and `4.4` rounds (at most `4`, `4`, `5`, `5` over ten
   seeds). The wall-clock exposure is those rounds times the re-verification interval, which the RFD does not give.
   The counter is finite: `120` fuses, one or more burned per update or tamper response
   (`test_tamper_with_an_exhausted_counter_kills_the_chip`).

4. **The pod-size bound holds only while device keys stay secret (§3).** An attacker device holding one stolen key from
   each of m pods of `8` chips passes the handshake as each of them, because it needs only a copy of the victim pod's
   signed manifest, which every member holds. It merges exactly the pods that lost a key: `16`, `24`, `32` and `64`
   chips for m = 2, 3, 4 and 8; extracting all eight keys of one pod adds nothing beyond that pod
   (`test_extracting_one_key_from_each_of_m_pods_joins_exactly_m_pods`). The cluster-size limit therefore reduces to
   the strength of the secure element, which this package assumes and does not measure.

5. **Where the high-bandwidth decision lives matters (§4).** No design survives every capability. Only the design that
   authenticates every frame under the handshake's session key holds against spoofed source ids and rewritten
   firmware; the firmware limiter holds only against key extraction (it uses no keys) and the handshake-then-plain-frames
   design holds only against an attacker with no capability at all. The model is harsher on the middle design than the
   RFD's "Medium" rating, because its attacker can inject frames and rewrite firmware. The matrix:

```
| design                      | none  | spoof source id | rewrite firmware | extract device keys |
|-----------------------------|-------|-----------------|------------------|---------------------|
| firmware limiter            | holds | BYPASSED        | BYPASSED         | holds               |
| handshake then plain frames | holds | BYPASSED        | BYPASSED         | BYPASSED            |
| handshake then AEAD frames  | holds | holds           | holds            | BYPASSED            |
```

6. **The limiter's guarantee is exact (§5).** Over any window the bucket grants at most the burst plus the rate times
   the window's length. A greedy sender polling every millisecond gets `10001000000` bytes against a bound of
   `10001000000`; across `285600` random windows the highest ratio of bytes carried to bytes allowed is `0.9965`
   (`test_no_window_ever_carries_more_than_burst_plus_rate_times_its_length`). Two details carry security weight and
   are tested: the bucket's clock never moves backwards, and the external bucket is per chip, not per destination, so
   fifty destinations share it (`2001000` bytes in two seconds against a bound of `2001000`;
   `test_splitting_traffic_over_many_destinations_does_not_multiply_the_external_limit`).

7. **At the RFD's example cap, read per chip, the reference run costs almost nothing extra (§8).** The
   communication-free ideal is `9506` chips; the cheapest plan at `1 GB/s` per chip is `4 stages x 38 replicas = 9728 chips`,
   a multiplier of `1.02x`, at utilisation `1.00`. Read instead as a cap on the whole pod, the same number gives
   `250x`. The RFD's own comparison (1 GB/s against 900 GB/s for NVLink) reads as per chip, and its diagram labels
   the pod's exit "<=1 GB/s", which reads as per pod; the choice changes the answer by the pod size. The RAND
   suggestion that about 1 GB/s may be sufficient is not reproduced or refuted here: its basis is not in the RFD.

8. **How low the cap must go (open question 2) is tens of MB/s per chip in this model (§9).** The cap below which the
   cheapest run needs more than `10x` the ideal chips is `74.9 MB/s` per chip with no reduction of
   data-parallel traffic, `7.79 MB/s` if gradient compression times local steps shrinks that traffic `100`-fold, and
   it plateaus near `2.41 MB/s` because pipeline traffic is untouched by those tricks. Dividing the RFD's `1 GB/s` by
   those thresholds gives `13.3`, `128` and `415`. Read per pod, the thresholds are `4.79 GB/s`, `499 MB/s` and
   `154 MB/s`. In §8 a `1 GB/s` per-pod cap costs `2.48x` against `100`-fold reduction and `1.53x` against `10000`-fold.
   "None" in the §8 tables means no layout with at most `512` stages and `4096` replicas finishes in a year, not a proof
   of impossibility. The reductions are free in this model; in practice they cost tokens or quality, so the thresholds
   err toward the attacker.

9. **What the threshold depends on (§10).** Smaller batches, faster chips, shorter deadlines and bigger models all raise
   it, so a cap fixed today binds harder later; communication-efficient training moves it the other way. A `4x` smaller batch and a
   `4x` faster chip give the same threshold, `150 MB/s` against `74.9 MB/s` at baseline, because it is set by
   data-parallel traffic, whose break-even depends on chip speed over batch size. A model that fits one pod needs only
   data-parallel traffic: for `1e+11` parameters the threshold at `100`-fold reduction falls to `124 kB/s`.

10. **Closed forms (§11).** Pure data parallelism breaks even (communication takes as long as compute) at a per-chip
    cap of `(R-1) g F / (3 c H B)`, independent of model size and pod size: `1.65 GB/s` for `R = 100` replicas, `16.5 MB/s`
    with `100`-fold reduction. At `1 GB/s` per chip it tops out at `3840` chip-equivalents however many pods are added.
    These forms are checked against the plan search by brute force
    (`test_the_plan_search_equals_brute_force_over_the_same_grid`).

11. **Pod size (open question 1) is a statement about compute per year, and a fixed limit erodes (§12).** A pod of `65`
    chips sustains `8.2e+23` FLOP per year at these assumptions, enough for a compute-optimal model of `82.7` billion
    parameters at twenty tokens per parameter. The years until such a pod, helped only by algorithmic efficiency,
    matches the reference run's `1.2e+26` FLOP: for `64` chips `3.6`, `7.2` and `14.4` years at doubling times of
    `0.5`, `1` and `2` years (`1.6`, `3.2`, `6.4` for `1024` chips). A year of compute stays under `1e+24`, `1e+25` and `1e+26`
    FLOP for pods of up to `79`, `792` and `7922` chips. The table holds the target fixed; frontier compute itself
    grows, which it ignores. Which compute level is dangerous is a policy input the model cannot supply.

12. **Multi-pod workloads (open question 4) are served by pipelining, not by tensor parallelism (§13).** At `1 GB/s`
    per chip the link of a pipelined server of `1e+13` parameters (`2` pods for int8 weights) carries `264x` the tokens
    per second the chips can compute, and `122x` for `1e+14` parameters (`20` pods); a tensor-parallel server gets
    `0.356x` and `0.0403x`, so the cap, not the chips, sets its speed. The reference model's weights fit one pod, but its
    training state needs `4` pods because of memory alone. Training sends exactly twice the boundary bytes of serving:
    `88.1` against `44` kB per token for the reference model. A bandwidth cap alone therefore does not separate
    pipelined serving from pipelined training; what differs is the data-parallel gradient traffic.

13. **The constants are counted (§6).** A ring all-reduce sends `1.5000` and `1.7500` times the vector per worker for
    `4` and `8` workers; a dense train step costs `6.0000` FLOPs per parameter per token; a pipeline boundary carries
    `2.0000` activation-sized vectors per token; a decoder block has `12.0000` d^2 parameters
    (`test_counted_training_flops_are_six_per_parameter_per_token`).

## Open questions

1. *"What pod size threshold prevents frontier training while enabling legitimate use?"* **Partially.** The model turns
   a size into compute per year and years until algorithmic progress erodes it (finding 11); choosing the threshold is a
   policy decision it cannot make.
2. *"How low must external bandwidth be to defeat communication-efficient training?"* **Answered in the model, under
   its assumptions:** tens of MB/s per chip against the reference run, and lower against reductions and smaller models
   (findings 8 and 9). Not answered for real training systems.
3. *"Can pod membership be updatable (for hardware replacement) without creating vulnerabilities?"* **Answered in
   the model, with residual risk:** signed generations above a fuse floor keep rollback and replay out, and the cost is
   a window for removed chips plus a finite counter (findings 2 and 3).
4. *"How to handle legitimate multi-pod workloads (e.g., large inference)?"* **Partially.** Pipelined serving fits under
   the cap and tensor-parallel serving does not; the cap does not separate pipelined serving from pipelined training
   (finding 12). No protocol for authorising a larger pod is modelled.
5. *"What's the manufacturing cost of implementing fixed set mechanisms?"* **Not addressed.** It needs silicon design
   and fabrication data.
6. *"Can fixed set be retrofitted to existing chip designs, or only new generations?"* **Not addressed.** The
   authenticated-frame design needs a secure element, fuses and an inline encryption engine on the link; whether
   existing designs have them is a question for their designers.

## Not established

- Nothing here was measured on real chips, fuses, links or training runs. The bandwidth analysis is a closed-form model
  whose inputs (chip speed, utilisation, memory, batch size, deadline, state size) are assumptions meant to be varied,
  and it lets the attacker shrink communication for free.
- The model covers synchronous layouts of K stages by R replicas. Pipeline bubbles, latency, stragglers, failures,
  mixture-of-experts, sharded optimiser state and asynchronous or decentralised schemes designed for slow links (which
  the RFD names as a direct threat) are represented only through the reduction factor and the activation size.
- The pod-size bound holds only while device keys stay secret (finding 4). How well real secure elements resist
  extraction is out of scope here (see RFDs 026, 028 and 036).
- The handshake uses standard primitives and was attacked by the tests listed; it was not formally verified. Role-separated
  signatures and refusal of a chip's own identity are redundant defences against reflection in this model, and
  `test_a_responder_signature_is_not_valid_as_an_initiator_signature_under_the_same_key` pins only the first.
- The RFD's physical row (disabling transceivers), the link layer and any timing side channel are not modelled.
- Fuse behaviour, including the high-voltage starvation attack, is a model (`aisecurity_rfds.common.nvm`), not a
  measurement of any process.
- Manufacturing cost and retrofit (open questions 5 and 6) are not addressed, and nothing here evaluates the RFD's
  claim that chips can be exported "with confidence" under a fixed set limit.

## Attacks tested

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Chip of another pod claims membership | peer must appear in the signed manifest | blocked |
| Manifest signed by a key that is not the manufacturer's | manufacturer key in fuses | blocked |
| One pod's manifest loaded into another pod's chips | chip must be listed in the manifest | blocked |
| Validly signed 2-chip pod offered to size-1 chips | size cap in fuses | blocked |
| Old manifest replayed after an update | generation floor in fuses | blocked |
| Flash image from before an update restored | generation floor in fuses | blocked |
| Flash image restored after a tamper response | tamper burns a generation fuse | blocked |
| The same, with the burn supply starved | none (the burn did not take) | attacker succeeds |
| Recorded handshake replayed | fresh nonces and ephemeral keys in a signed transcript | blocked |
| Chip's own hello reflected back at it | own identity refused | blocked |
| Initiator identity swapped in flight | identities inside the signed transcript | blocked |
| Release authorisation replayed, also after the chip rejoined | generation floor in fuses | blocked |
| Removed chip contacts a member that has not updated | gossip of the newer manifest | partial |
| Flash wiped, then a manifest for another pod, flash-bound chip | none (binding lived in flash) | attacker succeeds |
| The same, fuse-bound chip, also with the manifest written straight into flash | pod binding in fuses | blocked |
| One key extracted from each of m pods | none (keys assumed secret) | attacker merges m pods |
| Spoofed frame source, rewritten firmware | frame authentication under the session key | firmware and plain-frame designs bypassed; AEAD design holds |
| Captured AEAD frame replayed | receive counter | blocked |
| Traffic split over many destinations | one shared external bucket | blocked |
| Clock wound back to mint token credit | clock high-water mark | blocked |

## Related RFDs

- RFD 029 (NIC-Based Bandwidth Limiting & Interconnect Verification) and RFD 022 (NIC Repurposing for Guarantee
  Processor): the NIC is a natural enforcement point for this package's limiter and gate; they are modelled here in
  isolation, not as NIC designs.
- RFD 026 (Location Verification for AI Compute), RFD 028 (Retrofittable Tamper Detection System) and RFD 036
  (Tamper-Respondent PUF Enclosures): the strength of the key-extraction bound and the tamper trigger come from them;
  here the trigger is an input and the keys are assumed to stay secret.
- RFD 034 (License Replay Prevention) and RFD 032 (Offline Licensing for Compute Rationing): the same fuse-held rollback
  floor, high-voltage starvation attack and manufacturer-signed objects.
- RFD 031 (Chip Registry for AI Governance): the manufacturer's device registry is the registry this protocol relies on.
- RFD 037 (Secure Chip Binning and Feature Lockout): the pod size cap and manufacturer key are fused features of the
  kind it discusses.
