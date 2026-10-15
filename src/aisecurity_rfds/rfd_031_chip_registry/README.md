# RFD 031 - Chip Registry for AI Governance

Upstream: <https://projects.aisecurity.forum/rfd/031/> (Oxford Martin AI Governance Initiative, state: idea).

## Status

**Kind: reference-implementation.** A working registry protocol with simulated hardware under it. The RFD asks for a
registry of AI chips and lists five technical requirements (a unique identifier, a registry database, an update
protocol, a query interface and an immutable audit trail) and two families of identifier: hardware-based (on-chip keys,
PUFs, remote attestation) and physical inspection (glitter patterns, laser engravings, photographs, tamper-evident
seals). This package implements the protocol side as real code: identifiers bound to a device key, a sparse Merkle map
of ownership records, a signed append-only Merkle log, signed requests for enrolment, sale, relocation and retirement
that one deterministic function decides, clients that verify an answer, an auditor that replays the log, and witnesses
that cosign tree heads so that an operator cannot show two histories. It runs 40 attacks against it (17 against the
rules, 23 against an operator that cheats) and reports the four that pass unnoticed. It also models the identifiers
(copied-key detection by an attestation counter, PUF identification and key derivation, a glitter tag) and location
(salted commitments, a dictionary attack on them, a sampling audit for declared chips that are absent). Five of the
RFD's six open questions are addressed inside those models, one fully and four in part. The sixth, how robust on-chip
keys are against a state actor with physical access, is not addressed. No real chip, PUF, tag or registry was measured,
and the cryptography says who asked for a change and whether the record is consistent, not where a chip is.

## What is implemented

- `smt.py`: the registry database, a sparse Merkle map keyed by chip id. A key's position is the first 128 bits of a
  domain-separated hash of it, empty subtrees hash to precomputed defaults, and a proof carries a bitmap and only the
  siblings that are not defaults. A proof of absence is the same object as a proof of presence checked against an empty
  leaf, so a client can verify "this chip is not registered" instead of taking the operator's word for it.
- `identity.py`: the three identifier styles the RFD names. `DeviceKeyChip` signs a verifier's challenge together with
  a counter that only rises, and `CounterTracker` flags a counter that does not; `clone_detection_probability` gives
  closed forms for a copy that starts level with the original and one that starts ahead, and
  `clone_detection_by_enumeration` checks them against every sequence run through real chips. `PufChip` derives its key
  at every boot from the simulated PUF with a code-offset fuzzy extractor (`aisecurity_rfds.common.puf`) and can fail to
  attest when the PUF drifts; `puf_separation` measures same-chip and different-chip distances. `random_tag`,
  `photograph`, `match_fraction`, `glitter_error_rates` and `simulate_glitter` model an adhesive tag as points on a unit
  torus with Gaussian placement noise.
- `registry.py`: records (owner, status, jurisdiction, an optional location commitment, a version), signed requests
  that name the version they apply to, and `apply_request`, the one deterministic function that decides whether a
  request is valid: enrol, start a transfer, complete it, cancel it, relocate, retire. The operator (`Registry`) applies
  requests, appends each with the record it produced to a Merkle log, keeps the map, and signs tree heads that commit to
  both roots and the time. A client runs `verify_query` (signature, age, skew, proof) and `verify_history` (replay of
  one chip's entries); an auditor runs `audit_log` (replay of the whole log, with optional earlier heads).
  `Registry.observe_attestation` raises alerts for unknown chips, attestations that do not verify, copies and retired
  chips that attest.
- `witness.py`: `Witness` cosigns a head only if it extends the last head it cosigned (a consistency proof) and never
  cosigns two different heads of one size; `Quorum` accepts a head cosigned by `t` of `n` known witnesses;
  `CompromisedWitness` cosigns anything; `fork_by_enumeration` tries every way of showing two conflicting heads to real
  witness objects with a given number of colluders.
- `location.py`: salted location commitments, a dictionary attack on them, the number of distinct commitments an
  observer sees, the hypergeometric detection probability of a ghost-inventory audit and the number of chips an
  inspector must challenge, a simulation of it, and an audit run with real attestations against a `Facility` that holds
  only some of the chips declared for it.
- `scenario.py` and `attacks.py`: a deterministic cast (a manufacturer, two organisations, an attacker, an operator, an
  accredited recycler) and the catalogue of 40 attacks, each executed against the real code; the outcome in every row is
  what the code did.
- `__main__.py`: prints four numbered sections with every table and number quoted below. §1 is identifiers
  (technical requirement 1, open question 1), §2 the update protocol (requirement 3, open questions 5 and 6), §3 a
  dishonest operator and the witnesses (requirements 2, 4 and 5, open question 4) and §4 location (open questions 2
  and 3).

## Reproduce

```bash
pytest tests/rfd_031_chip_registry
python -m aisecurity_rfds.rfd_031_chip_registry   # prints every table and number quoted below (about 9 s)
python scripts/mutation_check.py 31                # 106 seeded bugs; every one must be caught
```

Section numbers (`§N`) below are the numbered sections of that program's output. Figures in backticks are copied from
it, and a test (`test_rfd031_zz_readme.py`) fails if one is not printed there.

## Findings

Every claim names the test or output section that backs it. All results are properties of models under assumed inputs.
The cast in §2 and §3 is a manufacturer, two organisations, an attacker, an operator and an accredited recycler. Their
keys are derived from their names, so anyone can recompute them, which is right for a demonstration and wrong for
anything else. The policy window for an offer is 14 days (`test_policy_defaults`).

1. **A registry id is a hash of a chip's public key, so an entry can be checked against the chip it names (§2).** The id
   is a domain-separated hash of the device public key, and an enrolment under any other id is refused with
   `bad-chip-id`. An attestation is a signature over the chip id, a verifier's challenge and a counter, so it verifies
   for one challenge and one key only
   (`test_a_chip_id_is_a_domain_separated_hash_of_its_public_key`,
   `test_the_chip_id_must_be_the_hash_of_the_chips_public_key`,
   `test_an_attestation_verifies_for_its_challenge_and_not_another`,
   `test_an_attestation_does_not_verify_under_another_chips_key`,
   `test_the_attestation_message_binds_chip_challenge_and_counter`). This shows that whoever attests holds the key. It
   does not show that the key is still inside the chip that was enrolled (finding 2) or that the chip is where its owner
   says it is (findings 14 and 15).

2. **A copied key is told apart only by the attestation counter, and only if both chips attest to the registry (§1).** A
   key extracted from a chip gives a copy whose signatures are indistinguishable from the original's, and which starts
   at the original's counter (`test_a_copy_has_the_same_identity_and_starts_at_the_originals_counter`). The registry's
   tracker flags a counter that fails to rise
   (`test_the_tracker_accepts_rising_counters_and_flags_repeats_and_drops`). If each attestation the registry sees is the
   copy's with probability q, a copy that starts level with the original is caught at the first switch between the two
   chips, and a copy that starts ahead only when the original attests after it. The closed forms agree to `1e-12` with
   every sequence of up to `6` attestations run through real chips
   (`test_the_closed_forms_equal_a_sum_over_every_sequence`, `test_the_closed_forms_at_known_points`).

   ```
   | attestations seen | share by the copy | copy starts level | copy starts ahead |
   |-------------------|-------------------|-------------------|-------------------|
   | 2                 | 0.05              | 0.0950            | 0.0475            |
   | 2                 | 0.20              | 0.3200            | 0.1600            |
   | 2                 | 0.50              | 0.5000            | 0.2500            |
   | 5                 | 0.05              | 0.2262            | 0.1832            |
   | 5                 | 0.20              | 0.6720            | 0.5632            |
   | 5                 | 0.50              | 0.9375            | 0.8125            |
   | 10                | 0.05              | 0.4013            | 0.3680            |
   | 10                | 0.20              | 0.8926            | 0.8568            |
   | 10                | 0.50              | 0.9980            | 0.9893            |
   | 20                | 0.05              | 0.6415            | 0.6216            |
   | 20                | 0.20              | 0.9885            | 0.9846            |
   | 20                | 0.50              | 1.0000            | 1.0000            |
   ```

   With the copy making `0.05` of the attestations seen, `5` attestations catch a level copy with probability `0.2262`
   and `20` catch it with `0.6415`; when it makes half of them, `5` catch it with `0.9375`. A copy that starts ahead is
   never easier to catch (`0.8125` against `0.9375` at that point;
   `test_a_careful_copy_is_never_easier_to_catch_than_a_level_one`), and a copy that is never used, or that is the only
   chip ever used, is never caught (`test_a_copy_that_is_never_or_always_used_is_never_caught`). This is a tripwire that
   makes using a copy risky. It does not make copying hard.

3. **A PUF separates chips by a wide margin, and the key derived from it is what stress breaks (§1).** Raw responses of
   1024 cells from 2000 simulated devices, where a response within 25% of the enrolled one is accepted:

   ```
   | condition | mean intra-chip BER | worst intra-chip BER | false rejects | mean inter-chip BER |
   |-----------|---------------------|----------------------|---------------|---------------------|
   | 25C new   | 0.0447              | 0.0654               | 0 of 2000     | 0.5000              |
   | 85C 5y    | 0.0608              | 0.0898               | 0 of 2000     | 0.5001              |
   | 125C 20y  | 0.0979              | 0.1299               | 0 of 2000     | 0.5001              |
   | 150C 40y  | 0.1462              | 0.1855               | 0 of 2000     | 0.5003              |
   ```

   Same-chip distance rises from `0.0447` when new to `0.1462` after 40 years at 150C, the worst chip stays under the
   threshold at `0.1855`, and an unrelated chip sits at `0.5000`. With no false rejects in 2000 the 95% upper bound on
   the false-reject rate is `0.0018`, and an unrelated chip is accepted with probability `2.9e-60` for 1024 cells,
   `2.4e-16` for 256 and `6.4e-09` for 128
   (`test_puf_responses_of_the_same_chip_are_close_and_of_different_chips_are_far`,
   `test_the_puf_false_accept_probability_is_a_binomial_tail`). Identification is not the weak point. Deriving a 128-bit
   key from the same PUF at every boot is, because the code that corrects the noise has to succeed every time. The share
   of 200 attempts that attest:

   ```
   | repetition | cells | 25C new | 85C 5y | 125C 20y | 150C 40y |
   |------------|-------|---------|--------|----------|----------|
   | 3          | 384   | 0.350   | 0.220  | 0.000    | 0.000    |
   | 5          | 640   | 0.865   | 0.515  | 0.190    | 0.000    |
   | 9          | 1152  | 1.000   | 1.000  | 0.995    | 0.845    |
   | 15         | 1920  | 1.000   | 1.000  | 1.000    | 1.000    |
   ```

   With `3` repetitions only `0.350` of boots attest even when new, and none after 20 years at 125C; with `5` the share
   after 20 years at 125C is `0.190`; with `9` the chip attests in `0.845` of boots after 40 years at 150C; `15`
   repetitions (`1920` cells) attest in all of them
   (`test_a_puf_chip_with_too_little_redundancy_cannot_attest_under_stress`,
   `test_more_redundancy_survives_conditions_that_defeat_less`,
   `test_a_puf_chip_attests_and_its_key_is_stable_across_boots`). A chip that cannot attest is a false reject, which the
   registry sees as a missing or invalid attestation and not as an attack. The PUF is the simulated one in
   `aisecurity_rfds.common.puf`; its noise model is an assumption.

4. **A glitter tag's error rates are steep in the noise, and the same curve is a forger's success rate (§1).** A tag of
   `30` flakes on the unit torus is accepted when `80%` of them match within radius `0.012`. The false-reject rate of the
   model climbs from `3.1e-08` at camera noise sigma `0.004` through `1.1e-03` and `9.6e-02` to `8.9e-01` at `0.008`,
   and 3000 simulated photographs agree (`0.0000`, `0.0010`, `0.1040`, `0.9033`):

   ```
   | sigma | false reject (model) | false reject (3000 photos) | 95% interval     | forger passes (model) | false accept (model) |
   |-------|----------------------|----------------------------|------------------|-----------------------|----------------------|
   | 0.004 | 3.1e-08              | 0.0000                     | [0.0000, 0.0013] | 1.0000                | 7.2e-40              |
   | 0.005 | 1.1e-03              | 0.0010                     | [0.0003, 0.0029] | 0.9989                | 7.2e-40              |
   | 0.006 | 9.6e-02              | 0.1040                     | [0.0936, 0.1154] | 0.9038                | 7.2e-40              |
   | 0.008 | 8.9e-01              | 0.9033                     | [0.8922, 0.9134] | 0.1088                | 7.2e-40              |
   ```

   Read the same sigma as the combined placement and camera error of a forger who copies the registry's photograph, and
   1 minus the false-reject rate is the chance that the forgery passes: `0.9038` at `0.006` and `0.1088` at `0.008`. A
   tag therefore resists forgery only as far as placing flakes within about half the match radius is hard, which is a
   question about printing and materials and not about the registry. An unrelated tag is accepted with probability
   `7.2e-40`, and among `10,000,000` tags the expected number of confusable pairs is `3.6e-26`. Those tails come from an
   independence approximation that is only approximate where it can be measured, `0.2117` against `0.2062` simulated,
   `0.4260` against `0.4400` and `0.7000` against `0.6937`
   (`test_the_false_accept_model_is_close_to_simulation_where_it_can_be_measured`), so they are extrapolations and not
   measurements (`test_the_false_reject_model_agrees_with_simulated_photographs`,
   `test_glitter_rates_for_a_single_flake_are_closed_forms`,
   `test_the_acceptance_count_is_rounded_up_without_float_noise`,
   `test_a_noisier_camera_rejects_more_and_a_larger_radius_accepts_more_of_both`). A tag identifies a package, not the
   silicon: nothing here stops a tag being moved to another chip.

5. **Every rule of the update protocol is enforced by the registry itself, in a fixed order (§2).** The catalogue runs
   `17` events against the real code: `16` are refused by a rule with a distinct code, `1` (a chip recorded as lost that
   attests to an inspector) raises an alert, and `0` are accepted
   (`test_every_protocol_attack_has_the_recorded_defence_and_outcome`,
   `test_the_rules_refuse_every_protocol_attack_that_is_a_request`; the rows are in the table under "Attacks tested"). A
   refused request changes nothing and is not logged (`test_a_rejected_request_changes_nothing`). The order of the checks
   decides which code is reported: staleness comes before authorisation, and the owner's signature before the state
   (`test_staleness_is_reported_before_authorisation`,
   `test_only_the_owner_can_start_a_transfer_and_that_is_checked_before_the_state`). Every request names the version it
   applies to, which is why a replayed offer fails with `stale-version` and two competing requests cannot both succeed
   (`test_a_request_for_another_version_of_an_active_chip_is_stale`,
   `test_a_request_for_another_version_of_a_chip_in_transit_is_stale`), and a request that names the same field twice is
   refused (`test_requests_that_name_a_field_twice_or_an_unknown_kind_are_refused`).

6. **A sale completes only with the buyer, a fresh attestation by the chip and an offer that has not lapsed (§2).** The
   chip signs a challenge bound to the chip, its record version, both parties and the deadline, so an attestation made
   for one offer fails for another
   (`test_the_challenge_binds_chip_version_both_parties_and_the_deadline`,
   `test_completion_needs_the_chip_to_sign_this_transfers_challenge`). A buyer can complete up to the deadline and not
   after it, the buyer can decline at any time, and the owner can withdraw only once the offer has lapsed, so a seller
   cannot pull an offer from under a buyer who has paid
   (`test_the_buyer_can_complete_up_to_the_expiry_and_not_after`,
   `test_the_buyer_can_decline_at_any_time_and_the_chip_returns_to_active_unchanged`,
   `test_the_owner_can_withdraw_only_after_the_offer_has_lapsed`). A copy that attested to an inspector first makes the
   genuine chip's completion fail with `counter-regression` and logs nothing
   (`test_the_transfer_counter_check_rejects_a_chip_that_went_backwards_and_logs_nothing`), which is finding 2's tripwire
   at the one moment a sale needs the chip. A sale into an embargoed jurisdiction is refused, but the jurisdiction is
   whatever the buyer declares (`test_completion_into_an_embargoed_or_blank_jurisdiction_is_refused`).

7. **Destruction, recycling and repurposing are one final state with a reason, and "destroyed" is a claim that someone
   accredited signed (§2).** A retirement records one of four reasons (destroyed, recycled, repurposed, lost), clears the
   location commitment and is final: nothing changes afterwards and the chip cannot be enrolled again
   (`test_retirement_is_final_and_records_why`, `test_a_retired_chip_cannot_be_enrolled_again`). When the policy names
   accredited witnesses, destruction needs one of their signatures
   (`test_destruction_needs_an_accredited_witness_when_the_policy_names_some`,
   `test_destruction_needs_no_witness_when_the_policy_names_none`). A retired chip that later attests to an inspector
   raises a `zombie` alert that carries the recorded reason
   (`test_a_retired_chip_that_attests_is_a_zombie_and_the_reason_is_reported`). The registry cannot tell a chip that was
   destroyed from one that was kept. It can make the owner's claim signed, witnessed and open to contradiction.

8. **One chip's whole life verifies on the client with a few hundred bytes of proof (§2).** The life below (enrolment,
   two sales, a relocation and a retirement) is replayed by the client from the log entries the operator supplies, each
   checked against the signed head, and the result is checked against the record that the map proves:

   ```
   | version | request           | status     | owner | jurisdiction |
   |---------|-------------------|------------|-------|--------------|
   | 1       | enroll            | active     | fab   | CA           |
   | 2       | start-transfer    | in-transit | fab   | CA           |
   | 3       | complete-transfer | active     | alice | US           |
   | 4       | relocate          | active     | alice | US           |
   | 5       | start-transfer    | in-transit | alice | US           |
   | 6       | complete-transfer | active     | bob   | GB           |
   | 7       | retire            | retired    | bob   | GB           |
   ```

   The client replays the `7` entries and finds them consistent with the map, and the proof material for that answer is
   `656 bytes` (7 log proofs and one map proof)
   (`test_every_stage_of_a_chips_life_verifies_for_a_client_and_an_auditor`,
   `test_a_chips_history_replays_to_the_record_the_map_proves`). The operator, the client and the auditor decide validity
   with the same function, so a client that replays a history reaches the state the rules give and not the state the
   operator reports (`test_a_replica_built_from_the_log_has_the_same_roots_and_state`,
   `test_replication_revalidates_every_request`, `test_replicas_that_apply_different_requests_diverge`).

9. **A dishonest operator is caught by an auditor, a client that replays history or a witness, depending on what it
   changes, and four things it can do pass unnoticed (§3).** The catalogue runs `23` attacks on the operator's own log,
   map and heads: `7` are detected by an auditor, `2` are refused by a witness, `6` are rejected by a client, `4` are
   attempts to show two clients different histories (finding 12) and `4` pass the check named. Rewriting an old entry
   and re-signing the head is detected by replay (entry `2` "records a state its request does not produce") and refused
   by a witness that cosigned the earlier head (`inconsistent`); appending a transfer the owner never signed is detected
   as an invalid request; a change made to the map but not to the log passes a single query, because the proof is valid
   for the doctored root, and is caught by the chip's history replay and by a whole-log audit
   (`test_a_doctored_map_is_caught_by_the_history_but_not_by_a_single_query`). An auditor that keeps earlier heads also
   catches a history rewritten into another valid one, a changed map at an earlier size, and an entry dated before a head
   that was published without it
   (`test_earlier_heads_expose_a_history_that_was_rewritten_into_another_valid_one`,
   `test_earlier_heads_catch_a_changed_map_at_an_earlier_size`,
   `test_earlier_heads_bound_how_far_an_entry_can_be_backdated`). The four that pass are a map change seen by a single
   query; two unrelated entries swapped, to an auditor with no earlier head; an entry dated earlier but still after the
   head published without it; and a request that the operator never logged. The last has no defence in this package: a
   log that omits a request is indistinguishable from one in which the request was never made, and a signed receipt,
   which would turn omission into evidence, is not modelled
   (`test_the_attacks_that_pass_unnoticed_are_the_four_the_readme_admits_to`).

10. **An operator that skips its counter check is caught from the log, but only for the counters the log carries (§3).**
    If the operator lets a copy complete a sale whose attestation counter is no higher than the genuine chip's last, a
    whole-log audit reports the entry ("completes a transfer with a counter that did not rise") and a client that
    replays the chip's history rejects it with `counter-regression`
    (`test_an_audit_reports_a_transfer_whose_attestation_counter_did_not_rise`,
    `test_a_history_whose_transfer_counter_did_not_rise_is_refused`,
    `test_an_honest_operator_refuses_the_sale_that_an_operator_without_the_counter_check_logs`). A copy that started
    ahead passes until the genuine chip completes a sale, and counters are judged per chip
    (`test_a_copy_that_started_ahead_passes_until_the_genuine_chip_completes_a_sale`,
    `test_attestation_counters_of_different_chips_are_judged_apart`). What an auditor cannot see is the attestations that
    inspectors presented: the operator's tracker uses them and the log does not carry them, so a copy seen only by an
    inspector is visible to the operator's alerts and not to the log
    (`test_a_repeated_or_lower_counter_is_a_clone_alert_with_the_evidence`).

11. **Heads age out and cannot be post-dated (§3).** A client accepts a head only if the operator's signature verifies,
    it is no older than the client's `max_age` and it is not more than `max_skew` seconds ahead of the client's clock.
    The attack rows show `stale-head` for a head from before a recent transfer and `future-head` for one dated far
    ahead (`test_a_query_head_must_be_neither_too_old_nor_from_the_future`). A head up to `max_age` old can still hide a
    transfer made in that interval, so the lag is a parameter that the client chooses, and the clock that the client
    trusts is outside the model.

12. **Witnesses turn "trust the operator not to fork" into "trust enough witnesses not to collude" (§3).** A witness
    cosigns a head only if it is genuine, not from the future, not older than the last it cosigned, not shorter, and
    linked to the last by a consistency proof; it never cosigns two different heads of one size
    (`test_a_witness_extends_its_head_only_with_a_consistency_proof`,
    `test_a_witness_will_not_go_back_to_a_shorter_log`,
    `test_a_witness_will_not_cosign_two_different_heads_of_the_same_size`,
    `test_a_witness_notices_a_changed_map_root_under_an_unchanged_log`). A client accepts a head cosigned by `t` of `n`
    witnesses (`test_a_head_is_accepted_with_exactly_the_threshold_of_distinct_witnesses`,
    `test_the_same_witness_counts_once`). Two conflicting heads of one size can then both gather a quorum only if the
    colluders cover the overlap, which is `max(0, 2t - n)`, and the number of witnesses that may be offline before no
    head can gather a quorum is `n - t`. Each row below was checked by trying every way of showing the two heads to real
    witness objects, with every possible number of colluders
    (`test_enumerating_every_way_to_show_two_heads_to_real_witnesses_matches_the_formula`):

    ```
    | witnesses | threshold | colluders needed to fork | witnesses that may be offline | matches enumeration |
    |-----------|-----------|--------------------------|-------------------------------|---------------------|
    | 3         | 1         | 0                        | 2                             | yes                 |
    | 3         | 2         | 1                        | 1                             | yes                 |
    | 3         | 3         | 3                        | 0                             | yes                 |
    | 4         | 2         | 0                        | 2                             | yes                 |
    | 5         | 3         | 1                        | 2                             | yes                 |
    | 5         | 4         | 3                        | 1                             | yes                 |
    | 5         | 5         | 5                        | 0                             | yes                 |
    | 7         | 4         | 1                        | 3                             | yes                 |
    | 7         | 5         | 3                        | 2                             | yes                 |
    | 7         | 6         | 5                        | 1                             | yes                 |
    ```

    Three witnesses with a threshold of `2` can be forked by `1` colluder and tolerate `1` offline; five with `4` need
    `3` colluders and tolerate `1`; seven with `6` need `5` and tolerate `1`. The two quantities trade against each
    other: a threshold equal to the number of witnesses needs all of them to collude and tolerates none being offline,
    and a threshold of `1` needs `0` colluders, which is no protection at all. The attack rows confirm it on real heads:
    with `3` witnesses and `2` needed, `1` colluder gets both heads accepted, and with `5` witnesses and `4` needed it
    takes `3` (`test_the_enumerated_heads_really_conflict`, `test_the_quorum_formulas_at_known_points`). A witness checks
    the log and not the map, so it prevents equivocation and not a wrong state in one head; the auditor's replay does
    that. All of this assumes the witnesses are independent, which is a fact about who runs them and not about code.

13. **A proof is about log2(n) hashes, so a client's cost is small (§3).** Measured on registries of `128`, `1024` and
    `4096` chips, as the mean over 32 chips:

    ```
    | chips | map siblings | map proof bytes | log hashes | log proof bytes |
    |-------|--------------|-----------------|------------|-----------------|
    | 128   | 7.2          | 248             | 7.0        | 224             |
    | 1024  | 10.2         | 341             | 10.0       | 320             |
    | 4096  | 12.3         | 411             | 12.0       | 384             |
    ```

    At `4096` chips a map proof carries `12.3` siblings (`411` bytes, including a 16-byte bitmap) and a log proof `12.0`
    hashes (`384` bytes). Extrapolated and not measured, about log2(n) hashes in each structure gives `20` hashes
    (`640` bytes) at `1,000,000` chips, `24` (`768`) at `10,000,000` and `30` (`960`) at `1,000,000,000`
    (`test_proofs_carry_about_log2_n_siblings`, `test_proof_size_is_the_bitmap_plus_32_bytes_per_sibling`,
    `test_every_sibling_in_a_proof_is_a_stored_node_and_default_siblings_are_omitted`). These sizes say nothing about the
    operator's storage, query load or the cost of replaying a whole log, which grows with its length.

14. **A location commitment hides a facility only as far as its salt is unguessable and unshared (§4).** A commitment
    opens to the facility and salt it was made with and to nothing else, and it is domain-separated
    (`test_a_commitment_opens_only_to_the_facility_and_salt_it_was_made_with`,
    `test_a_commitment_is_domain_separated_and_does_not_blur_facility_and_salt`). With `100` candidate facility names a
    dictionary attack recovers the facility from a commitment with no salt after `100` hash evaluations and with a
    one-byte salt after `25,600`; a two-byte salt costs `6,553,600` evaluations, not tried, and a 16-byte one `3.4e+40`
    (`test_a_short_salt_falls_to_a_dictionary_and_the_right_opening_is_recovered`,
    `test_dictionary_work_grows_with_candidates_and_256_per_salt_byte`):

    ```
    | salt bytes | hash evaluations to exhaust | facility recovered |
    |------------|-----------------------------|--------------------|
    | 0          | 100                         | yes                |
    | 1          | 25,600                      | yes                |
    | 2          | 6,553,600                   | not tried          |
    | 16         | 3.4e+40                     | not tried          |
    ```

    What an observer of the registry learns also depends on how salts are shared: for `1000 chips in 10 facilities` the
    registry shows `10 distinct commitments` with a salt per facility, so an observer can group chips by facility without
    learning where it is, and `1000` with a salt per chip, so it cannot
    (`test_distinct_commitments_counts_what_an_observer_could_group`,
    `test_the_same_facility_with_different_salts_gives_unlinkable_commitments`). The choice between no location, a
    jurisdiction only, a commitment and plaintext is the RFD's policy question. The model shows what each leaks and not
    which to choose.

15. **A logical location is only as good as the audit behind it (§4).** If some declared chips are not where they are
    declared, or do not exist, an inspector who challenges a random sample finds one with the hypergeometric
    probability. For `10,000` declared chips:

    ```
    | ghosts | share | chips to inspect for 95% | P(detect) inspecting 58 |
    |--------|-------|--------------------------|-------------------------|
    | 1000   | 10.0% | 29                       | 0.9978                  |
    | 500    | 5.0%  | 59                       | 0.9494                  |
    | 100    | 1.0%  | 294                      | 0.4427                  |
    | 10     | 0.1%  | 2588                     | 0.0565                  |
    ```

    Finding a 10% problem takes `29` chips, a 1% problem `294` and a 0.1% problem `2588`, and inspecting `58` chips finds
    that last one with probability `0.0565` (`test_the_known_audit_sizes`,
    `test_detection_probability_is_the_exact_hypergeometric_complement`,
    `test_the_chips_to_inspect_is_the_smallest_sample_that_reaches_the_confidence`). A simulation agrees with the formula:
    for `500` declared chips, `30` absent and `20` inspected, the model gives `0.7170` and the simulated audits `0.7217`,
    within `[0.7154, 0.7279]` (`test_simulation_agrees_with_the_closed_form_and_is_reproducible`). An audit run with real
    attestations (`200` declared, the first `12` absent, `60` challenged) fails exactly the absent chips in the sample,
    which are `2` (`test_an_audit_returns_exactly_the_absent_chips_it_sampled`,
    `test_a_facility_produces_the_chips_it_holds_and_not_the_ones_it_does_not`). A logical location can therefore stand
    in for a physical one only where its owner accepts audits of this size, and where a chip that answers a challenge is
    taken to be at the place that sent it, which a relayed challenge would defeat (not modelled; see RFD 026).

## Open questions

1. *"How robust are current on-chip keys (NVIDIA GPUs, security modules, PUFs) against sophisticated state actors with
   physical access?"* **Not addressed.** The simulated device key and PUF say nothing about what a state actor with
   physical access can extract. The package shows only what the registry can do after a key has leaked: the counter
   tripwire catches a copy that is used alongside the original, and never catches one that replaces it (findings 2
   and 10).
2. *"Should location information be included given transparency-security tradeoffs for military hardware?"*
   **Partially.** The model shows what each option reveals: a jurisdiction only, a commitment that groups chips by
   facility when salts are shared per facility and does not when they are per chip, and a commitment that a dictionary
   defeats when the salt is short (finding 14). Whether military hardware should be recorded at all, and at which of
   those levels, is a policy judgement that the package does not make.
3. *"Can "logical location" (institutional responsibility) substitute for physical location tracking?"* **Partially.**
   A logical location holds only as far as an audit can test it. Finding 15 gives the sample sizes (`29` to `2588`
   chips of `10,000`) and shows that a challenge fails exactly for absent chips, but a chip that answers is not shown to
   be where the inspector asked, and the model says nothing about whether institutional responsibility has legal or
   deterrent force.
4. *"What governance structure should manage the registry? (International body, consortium, distributed?)"*
   **Partially.** The technical part is answered: an operator alone can be caught rewriting, forking or backdating in all
   but four of the cases tried, witnesses move the assumption to a quorum, and a fork needs `max(0, 2t - n)` colluders
   while `n - t` witnesses may be offline (findings 9 to 12). The policy part is not: who holds the operator, witness,
   manufacturer and destruction-witness keys, and who sets the embargo list.
5. *"How to handle chip destruction, recycling, or repurposing?"* **Partially.** Retirement is a final state with a
   reason, destruction needs an accredited witness's signature, and an inspector who sees a retired chip attest raises
   an alert (finding 7). The registry cannot verify physical destruction, and the model has no state for a chip that is
   repurposed and stays in use outside the registry's scope.
6. *"What happens when chips are sold or transferred between organizations?"* **Answered in the model, with residual
   risk:** a sale needs the owner's offer, the buyer's signature, the chip's attestation bound to that offer and an
   unexpired window, and a copy that attested first is stopped at completion (findings 5, 6 and 10). The residual risk is
   physical: a chip sold without any registry request is noticed only by an audit (finding 15), and the buyer's
   declared jurisdiction is not checked.

## Not established

- Nothing here was measured on real chips, PUFs, tags or registries. The device key, the PUF and the glitter tag are
  simulated (the PUF in `aisecurity_rfds.common.puf`, the tag geometry in `identity.py`) and their parameters are
  assumptions. Open question 1 is not addressed: nothing models extraction of a key by a state actor, so the value of
  the counter tripwire (finding 2) against such an actor is unknown.
- The cryptography establishes who asked for a change and that the log and the map are consistent. It does not
  establish that a chip is where its record says, that a declared jurisdiction is true, that a destroyed chip was
  destroyed, or that a request the operator dropped was ever submitted (finding 9). A chip sold without a registry
  request is outside the protocol until an audit finds it.
- Time is an input. The operator's clock, the client's clock and the age limit decide what counts as stale; a client
  that trusts a wrong clock accepts a stale head up to its limit, and nothing detects an operator that is consistently
  wrong about the time (finding 11).
- Governance is reduced to counts: how many witnesses must collude, and who may sign for manufacturers, destruction and
  embargoes. Who should hold those roles and who sets the policy is not answered (open question 4).
- The counter check on a sale is partly the operator's alone. Its tracker also learns from attestations that inspectors
  present, which the log does not carry, so an auditor can check only the counters the log holds (finding 10).
- The audit figures assume that absent chips are chosen without knowledge of which will be inspected, that an inspector
  can challenge a chip, and that a chip that answers is where the inspector is. A relayed challenge (RFD 026), a chip
  borrowed from another site for a day and a facility told in advance which chips will be inspected are not modelled.
- Scale and operations are not tested. Proof sizes at ten million and a billion chips are extrapolations of log2(n);
  key rotation and compromise for the operator, witnesses and manufacturers, availability, rate limits, denial of service
  and storage of the log are outside the model.

## Attacks tested

The first 17 rows are the update-protocol attacks of §2 and the next 23 the operator attacks of §3, copied from the
program's output, where the outcome in each row is what the code did. The rows after them are attacks on the
identifiers and on location from §1 and §4, with the figures that those sections print.

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Enrolment signed by a key that is not an approved manufacturer | manufacturer allow-list | rejected: unauthorised |
| Enrolment under an id that is not the hash of the chip's key | chip id binding | rejected: bad-chip-id |
| A second enrolment of a registered chip | unique chip id | rejected: duplicate |
| A transfer offered by someone who does not own the chip | owner's signature | rejected: unauthorised |
| An old offer replayed after the record has moved on | expected version | rejected: stale-version |
| A second offer while the first is pending | transfer state machine | rejected: bad-state |
| Completion by a party other than the named buyer | buyer's signature | rejected: unauthorised |
| Completion after the offer has lapsed | offer expiry | rejected: expired |
| Completion that reuses the chip's attestation from an earlier offer | challenge bound to the offer | rejected: bad-attestation |
| The genuine chip completes a sale after a copy of it attested | attestation counter | rejected: counter-regression |
| The seller withdraws an offer before it lapses | offer window | rejected: not-expired |
| A sale into an embargoed jurisdiction | policy list | rejected: embargoed |
| Any change to a retired chip | retirement is final | rejected: retired |
| Destruction recorded without an accredited witness | witness signature | rejected: unwitnessed |
| A relocation while a transfer is pending | transfer state machine | rejected: bad-state |
| A request that names the same field twice | canonical fields | rejected: bad-field |
| A chip recorded as lost attests to an inspector | attestation monitoring | alert: zombie |
| Rewrite an old entry and re-sign the head | audit replay | detected: entry 2 records a state its request does not produce |
| The same rewrite shown to a witness that cosigned the earlier head | consistency proof | refused: inconsistent |
| Append a transfer the owner never signed | audit replay | detected: entry 4 is not a valid request (unauthorised) |
| Operator lets a copy complete a sale: the whole log is replayed | audit replay | detected: entry 4 completes a transfer with a counter that did not rise |
| The same: the chip's history is replayed | history replay | rejected: counter-regression |
| Change a record in the map but not in the log: a single query | map proof | accepted |
| The same change: the chip's history is replayed | history replay | rejected: history-disagrees-with-map |
| The same change: the whole log is replayed | audit replay | detected: map root differs from the signed head |
| Swap two unrelated entries and re-sign the head | audit replay | accepted: audit passes |
| The same, to an auditor that kept the earlier head | audit replay with earlier heads | detected: log root at size 3 differs from an earlier head |
| Date an entry before a head that was published without it | audit replay with earlier heads | detected: entry 3 is dated before a head that was published without it |
| Date an entry earlier, but after the head published without it | audit replay with earlier heads | accepted: audit passes |
| Serve a head from before a recent transfer | head age limit | rejected: stale-head |
| Serve a head dated far in the future so it stays fresh | clock skew limit | rejected: future-head |
| Show two clients different histories: 3 witnesses, 2 must cosign, 0 colluding | witness quorum | no way found to get both accepted |
| Show two clients different histories: 3 witnesses, 2 must cosign, 1 colluding | witness quorum | both heads accepted |
| Show two clients different histories: 5 witnesses, 4 must cosign, 1 colluding | witness quorum | no way found to get both accepted |
| Show two clients different histories: 5 witnesses, 4 must cosign, 3 colluding | witness quorum | both heads accepted |
| Drop the newest entries and sign a shorter head | witness | refused: shrinking |
| The same, to an auditor holding the shorter log and the longer head | audit replay | detected: log has 3 entries but the head covers 4 |
| Never log a request that was submitted | none | accepted: audit passes |
| Claim that a registered chip is not registered | non-membership proof | rejected: bad-proof |
| Answer a query with a different chip's record | record and chip id check | rejected: wrong-record |
| A key copied out of a chip and used for a `0.05` share of the attestations the registry sees | attestation counter | caught with probability `0.2262` in 5 attestations and `0.6415` in 20 if it starts level, `0.1832` and `0.6216` if it starts ahead |
| A key copied out of a chip and used for every attestation | attestation counter | never caught (finding 2) |
| A forger places glitter flakes from the registry's photograph with combined error `0.006` | 80% flake match within `0.012` | passes with probability `0.9038` |
| The same forger with combined error `0.008` | 80% flake match within `0.012` | passes with probability `0.1088` |
| An unrelated glitter tag is presented as the registered one | 80% flake match within `0.012` | accepted with probability `7.2e-40` (model, extrapolated) |
| An unrelated chip's PUF response is presented as the registered chip's | raw response within 25% of the enrolled one | accepted with probability `2.9e-60` for 1024 cells and `6.4e-09` for 128 |
| A dictionary attack on a location commitment with a one-byte salt, 100 candidate facilities | salted commitment | facility recovered after `25,600` hash evaluations |
| The same with a two-byte salt | salted commitment | `6,553,600` evaluations (not tried) |
| The same with a 16-byte salt | salted commitment | `3.4e+40` evaluations (not tried) |
| 100 of 10,000 declared chips are not where declared, 58 chips challenged | ghost-inventory audit | detected with probability `0.4427` |
| 12 of 200 declared chips are absent, 60 challenged with attestation challenges | audit with real attestations | `2` failed, exactly the absent chips in the sample |

## Related RFDs

- RFD 037 (Secure Chip Binning and Feature Lockout): the accuracy of the registry is invariant I2 there. This package
  shows what keeps a record accurate (signed, versioned requests that every party replays, and audits) and what it
  cannot (a sale that never reaches the registry).
- RFD 034 (License Replay Prevention) and RFD 032 (Offline Licensing for Compute Rationing): the same monotonic-counter
  idea. Here the counter reveals a copied key, there it stops a licence being reused.
- RFD 026 (Location Verification for AI Compute): what a chip's answer to a challenge says about where it is, which the
  audit in finding 15 assumes and does not model.
- RFD 036 (Tamper-Respondent PUF Enclosures) and RFD 028 (Retrofittable Tamper Detection System): what it costs to
  extract a device key or a PUF response, which is open question 1 here and is not modelled.
- RFD 038 (Containerized Data Centers for Mobile/Military AI): by its title, the mobile and military installations that
  make the location question (open question 2) hard.
