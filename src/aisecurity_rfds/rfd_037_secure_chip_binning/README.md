# RFD 037 - Secure Chip Binning and Feature Lockout

Upstream: <https://projects.aisecurity.forum/rfd/037/> (RAND Corporation, state: idea).

## Status

**Kind: simulation.** Four executable models, none of them a measurement. The RFD observes that features switched off
on a binned chip often stay on the die, fused off or disabled in firmware, and that an attacker with enough resources
could switch them back on. It lists four secure approaches (separate dies, irreversible disablement, cryptographic
binding, PUF-bound features) and five ways to verify that circuits are absent. This package (1) models what binning
leaves latent and what dies built for each tier would cost in silicon, (2) runs the four approaches against attackers
with explicit budgets of fault injections and circuit edits, (3) searches every reachable state of a small model of
upgrades, downgrades, die replacement and token replay, and (4) computes how large a destructive sampling audit has to
be. It addresses four of the RFD's five open questions inside those models and none of them for real hardware. Every
yield, cost, demand and detection input is an assumption, and the attacker budgets are counts of actions, not costs.

## What is implemented

- `binning.py`: the yield and cost model. Clustered (negative-binomial) defects over a die made of one uncore region
  that must be defect free and identical blocks that may be fused off; the distribution of good blocks per die comes
  from scaled Gauss-Laguerre quadrature and is checked against closed forms, an independent integration and Monte
  Carlo. Planners: `plan_binning` (chips per wafer from a supply-and-demand bound for nested tiers, and the smallest
  possible latent top share), `plan_separate` (a die per lower tier, spare blocks capped by `max_safe_spares`),
  `plan_hybrid` (top dies that fell short of the top count sold lower as by-products), `silicon_premium`,
  `breakeven_defect_density`, `total_cost_per_unit` and `breakeven_volume`. Costs are in wafer equivalents.
- `mechanisms.py`: the lockout designs behind one interface, and an `Attacker` with budgets of fault injections and
  circuit edits (privileged software is always assumed). `FirmwareFlag`; `FuseGate` with 1 to 7 redundant copies read
  by majority; `AntifuseRail`, whose self-test reads a rail; `TokenGate` in three variants (bare, device-bound,
  device-bound with an epoch counter) using real Ed25519 signatures and a unary counter from `aisecurity_rfds.common`;
  `PufBound`, a real fuzzy extractor over a simulated PUF with public helper data on the chip and a manufacturer
  database that holds only the key of the purchased feature; and `SeparateDie`. An attack ends blocked (feature stays
  off), visible (feature works and the chip's report says so) or silent (works and the report says off); only a circuit
  edit can forge the report.
- `lockout.py`: the matrix over twelve attacker profiles, a search for the smallest budget that works, the chance
  that random faults flip a redundant fuse vote, and PUF key failure rates under temperature and age with
  Clopper-Pearson bounds.
- `rma.py`: a bounded model checker for two chips and three tiers. Each chip has a physical cap, an entitlement, a
  registry record, a running tier and an epoch counter. Actions are upgrade, downgrade, die swap and an attacker who
  holds every token ever minted and replays any of them. It reports the shortest sequence that breaks each invariant
  and searches to a fixed point.
- `audit.py`: hypergeometric miss probability with an assumed per-chip detection probability, chips to inspect for a
  confidence level, the share of the lot destroyed, how many cheated chips escape, repeated lots, a sampling
  simulation, and a design-file comparison that reads only what the manufacturer declares.
- `__main__.py`: prints four numbered sections (§1 to §4) with every table and number quoted below.

## Reproduce

```bash
pytest tests/rfd_037_secure_chip_binning
python -m aisecurity_rfds.rfd_037_secure_chip_binning   # prints every table and number quoted below (about 6 s)
python scripts/mutation_check.py 37                      # 85 seeded bugs; every one must be caught
```

Section numbers (`§N`) below are the numbered sections of that program's output. Figures in backticks are copied
from it, and a test (`test_rfd037_zz_readme.py`) fails if one is not printed there.

## Findings

Every claim names the test or output section that backs it. All results are properties of models under assumed inputs.
The yield and cost model (§1) uses a die of `224 mm2` of uncore plus `144` blocks of `4 mm2` (`800 mm2` in all, `64`
dies per `300 mm` wafer), tiers of `132`, `114` and `96` blocks with demand of `20%`, `30%` and `50%`, and defect
clustering alpha `3`. Costs are thousandths of a wafer per chip, so no price is invented.

1. **Binning leaves the top tier latent in nearly every chip sold below it (§1).** Even for the allocation that
   minimises it, the share of lower-tier chips whose die reached the top block count is `100.0%` up to a defect
   density D0 of `0.5` per cm2, `99.4%` at `1`, `96.8%` at `2` and `92.5%` at `4`. Fusing blocks off does not remove
   them, and at plausible defect densities almost every fused-down chip could have run at the top tier
   (`test_latent_share_is_total_at_low_defect_density_and_falls_as_defects_rise`; the minimum over all allocations is
   checked against a linear program in `test_latent_top_share_is_the_least_any_allocation_can_achieve`). The yields
   under it are checked three ways: for the uncore surviving at D0 `1.0` the closed form gives `0.1877`, the quadrature
   `0.1877` and Monte Carlo `0.1853`; for a die reaching the top count at D0 `4.0`, the quadrature gives `0.0148` and
   Monte Carlo `0.0147`
   (`test_quadrature_reproduces_the_closed_form_survival_of_the_whole_die_and_of_the_uncore`,
   `test_every_entry_of_the_pmf_agrees_with_an_independent_integration`,
   `test_the_pmf_agrees_with_a_monte_carlo_of_the_same_defect_model`).

2. **At these tiers separate dies use less silicon than binning, not more (§1).** Lower tiers get smaller dies, which
   yield better and fit more to a wafer, and the planner adds spare blocks up to the number that keeps each die short
   of the tier above (spares are latent capability too; `test_max_safe_spares_and_plan_separate_refuse_a_die_that_reaches_the_tier_above`,
   `test_best_safe_spares_is_the_cheapest_choice_by_brute_force`).

   ```
   | D0   | binned chips/wafer | latent top share | binned cost | spares (mid, low) | separate cost | hybrid cost | hybrid premium |
   |------|--------------------|------------------|-------------|-------------------|---------------|-------------|----------------|
   | 0.05 | 57.3               | 100.0%           | 17.44       | 2, 3              | 14.27         | 14.27       | -18.2%         |
   | 0.1  | 51.6               | 100.0%           | 19.39       | 4, 3              | 15.93         | 15.93       | -17.9%         |
   | 0.2  | 42.2               | 100.0%           | 23.72       | 4, 3              | 19.61         | 19.61       | -17.3%         |
   | 0.5  | 24.7               | 100.0%           | 40.47       | 6, 6              | 33.95         | 33.94       | -16.1%         |
   | 1    | 12.0               | 99.4%            | 83.26       | 10, 8             | 70.86         | 70.79       | -15.0%         |
   | 2    | 4.1                | 96.8%            | 242.19      | 12, 11            | 210.60        | 209.47      | -13.5%         |
   | 4    | 1.0                | 92.5%            | 990.03      | 15, 13            | 877.64        | 866.04      | -12.5%         |
   ```

   The hybrid saves `18.2%` of the silicon per chip at D0 `0.05` and `12.5%` at D0 `4` (`17.44` against `14.27`, and
   `990.03` against `866.04`). This answers the RFD's cost question for one assumed product; it does not say separate
   dies are cheaper in practice, because one-off cost and flexibility are outside it (findings 3 and 6).

3. **The sign flips when the tiers are close together (§1).** With the mid and low tiers `gap` and 2 x `gap` blocks
   below the top, the room for yield spares is the gap, so a small gap leaves little.

   ```
   | tier gap (blocks) | safe spares | premium at D0 0.1 | at D0 1 | at D0 4 | break-even D0 |
   |-------------------|-------------|-------------------|---------|---------|---------------|
   | 18                | 17          | -17.9%            | -15.0%  | -12.5%  | none up to 20 |
   | 12                | 11          | -13.6%            | -10.2%  | -6.3%   | none up to 20 |
   | 6                 | 5           | -9.0%             | +2.5%   | +33.5%  | 0.85          |
   | 3                 | 2           | -6.0%             | +62.0%  | +189.4% | 0.22          |
   ```

   At a gap of `6` the hybrid costs `2.5%` more at D0 `1` and `33.5%` more at D0 `4`, with break-even at D0 `0.85`;
   at a gap of `3` it costs `62.0%` and `189.4%` more, with break-even at `0.22`. At gaps of `18` and `12` there is
   none up to `20`. The premium the RFD asks about is therefore a function of how far apart the sold tiers are
   (`test_silicon_premium_changes_sign_with_the_spacing_of_the_tiers`).

4. **Salvaging defective top dies helps by about one per cent (§1).** The hybrid also sells top dies that fell short of
   the top block count as lower-tier chips, which is safe because the missing blocks are defective, not fused. Against
   plain separate dies it saves `70.79` against `70.86` at D0 `1` and `866.04` against `877.64` at D0 `4`, and
   nothing visible at low D0 (`test_hybrid_never_costs_more_than_separate_dies_and_saves_where_defective_top_dies_are_plentiful`,
   `test_hybrid_only_sells_dies_that_fall_short_of_the_top_count_below_the_top_tier`; the hybrid's cost equals a linear
   program over all placements of by-product dies in `test_hybrid_cost_equals_a_linear_program_over_all_placements_of_by_product_dies`).

5. **Defect clustering moves yield much more than it moves the premium (§1).** At D0 `1`, the chance that the uncore is
   defect free is `0.427` at alpha `0.5`, `0.309` at `1`, `0.188` at `3` and `0.109` at `100` (close to Poisson),
   while the hybrid's saving stays between `14.5%` and `16.1%` and the latent top share between `99.39%` and
   `99.83%`.

6. **One-off cost decides, and the model has no number for it (§1).** At D0 `0.1`, two extra designs that cost `1,000`
   wafer equivalents break even at `288,409` chips, `10,000` at `2,884,093` and `100,000` at `28,840,935`. At
   `100,000` chips the three cases are `33.7%`, `497.8%` and `5138.7%` dearer than binning; at `10,000,000` chips they
   are `17.4%` cheaper, `12.7%` cheaper and `33.7%` dearer (`test_nre_arithmetic_and_break_even_volume`). Real design,
   mask and validation costs, and the loss of being able to shift wafer starts between tiers, are not modelled.

7. **A firmware flag falls to software alone, and every in-silicon gate falls to a few faults or edits (§2).** The
   matrix below gives, for each design and attacker, B (feature stays off), V (feature works and the chip's report says
   so) or S (works and the report says off). The attackers are privileged software only (`soft`), `N` fault injections
   (`gN`), `N` circuit edits (`eN`), one of each (`e1g1`), a token minted for another chip (`tok_other`), this chip's
   token from before a downgrade (`tok_old`), the manufacturer's signing key and database (`mfr`) and a readout of the
   chip's PUF response (`puf`).

   ```
   | mechanism         | soft | g1 | g3 | e1 | e2 | e3 | e1g1 | e1g3 | tok_other | tok_old | mfr | puf |
   |-------------------|------|----|----|----|----|----|------|------|-----------|---------|-----|-----|
   | firmware_flag     | S    | S  | S  | S  | S  | S  | S    | S    | S         | S       | S   | S   |
   | fuse_x1           | B    | V  | V  | V  | S  | S  | S    | S    | B         | B       | B   | B   |
   | fuse_x3           | B    | B  | V  | B  | V  | S  | V    | S    | B         | B       | B   | B   |
   | antifuse_rail_st1 | B    | B  | B  | B  | S  | S  | S    | S    | B         | B       | B   | B   |
   | antifuse_rail_st3 | B    | B  | B  | B  | S  | S  | B    | S    | B         | B       | B   | B   |
   | token_unbound     | B    | V  | V  | V  | S  | S  | S    | S    | V         | V       | V   | B   |
   | token_bound       | B    | V  | V  | V  | S  | S  | S    | S    | B         | V       | V   | B   |
   | token_bound_epoch | B    | V  | V  | V  | S  | S  | S    | S    | B         | B       | V   | B   |
   | puf_bound         | B    | V  | V  | V  | S  | S  | S    | S    | B         | B       | B   | V   |
   | separate_die      | B    | B  | B  | B  | B  | B  | B    | B    | B         | B       | B   | B   |
   ```

   Only `separate_die`, which has no circuit to enable, is blocked in every column
   (`test_a_separate_die_blocks_every_budget_and_only_the_wrong_die_fails`; a die with the top tier's blocks present is
   silent at once). The smallest budgets that work are `1` fault injection or `1` edit for a single fuse and for the
   token and PUF gates, `2` and `2` for three fuse copies, and `2` edits for the antifuse rail, which a single edit only
   bricks and which has no fault-injection route in the model
   (`test_a_firmware_flag_falls_to_software_alone_and_the_forged_report_hides_it`,
   `test_a_single_fuse_read_falls_to_one_fault_or_one_edit_and_the_chip_reports_it`,
   `test_a_majority_of_three_fuses_needs_two_flipped_reads`,
   `test_one_edit_on_the_antifuse_rail_bricks_the_chip_and_two_hide_it`,
   `test_the_budget_search_agrees_with_running_the_attacker_directly`). These are counts. Whether one edit can do what
   the model lets it do is not established.

8. **Hiding a bypass costs one more edit than making it (§2).** The edits needed to hide a bypass from the chip's own
   report are `2` for a single fuse, the token gates and the PUF gate, `3` for three fuse copies and `2` for the
   antifuse rail: only a circuit edit can forge a report, so faults and replayed tokens leave a visible trace
   (`test_hiding_a_fuse_bypass_costs_one_more_edit`, `test_only_a_circuit_edit_ever_hides_a_bypass_from_the_chips_own_report`,
   `test_more_budget_never_makes_an_attack_worse`).

9. **Redundancy buys protection against random faults, not against an attacker who can edit or keep rebooting (§2).**
   With each read corrupted at probability `0.05` per boot, `3` copies flip the vote with probability `7.25e-03` per
   boot but `0.9993` over `1000` boots, and `7` copies with `1.94e-04` and `0.1760`. The closed form for `3` copies at
   `0.05` is `0.00725` against a Monte Carlo of `0.00710` (interval `0.00640` to `0.00787`)
   (`test_random_fault_bypass_probability_matches_exact_binomial_values`,
   `test_more_redundant_reads_lower_the_chance_that_random_faults_flip_the_vote`,
   `test_simulated_random_fault_boots_agree_with_the_closed_form`). The antifuse self-test has the same trade: each
   extra read it needs is one more fault to inject (`test_the_self_test_must_be_fooled_on_every_read_so_its_read_count_sets_the_fault_cost`).

10. **Tokens fail in layers, and a leaked signing key defeats all of them but not PUF binding (§2).** A bare token
    enables any chip (`tok_other` is `V`), binding it to a device blocks that, a device-bound token from before a
    downgrade still works until an epoch counter advances (`tok_old`), and the manufacturer's key and database open every
    token design (`mfr`) but not the PUF-bound feature, which falls to a readout of the PUF response (`puf`)
    (`test_an_unbound_token_enables_every_chip_and_a_bound_token_enables_one`,
    `test_a_device_bound_token_stays_valid_after_a_downgrade_until_the_epoch_counter_advances`,
    `test_a_leaked_signing_key_opens_every_token_design_but_not_a_puf_bound_feature`,
    `test_a_puf_bound_feature_falls_only_to_a_readout_of_the_puf_or_to_an_edit`). The decision a token or PUF check
    makes is a single one, so one fault injection flips it, and reading it three times raises the price to two
    (`test_a_single_decision_gate_falls_to_one_fault_and_redundancy_raises_the_price`).

11. **PUF binding is limited by field stability as much as by attack (§2).** A PUF-bound key that fails to reproduce
    locks a legitimate customer out. In 300 simulated devices enrolled at 25 C, three repetitions of the code fail
    `161 of 300` new, `226 of 300` after 5 years at 85 C and `291 of 300` after 20 years at 125 C; fifteen repetitions
    fail `0 of 300`, `0 of 300` and `3 of 300`, an upper bound of `2.9%`
    (`test_puf_keys_fail_more_often_under_stress_and_with_less_error_correction`,
    `test_the_field_failure_bound_is_a_clopper_pearson_upper_end`). The simulated device's noise model is an
    assumption, not data from any silicon PUF.

12. **Tier changes need binding, an epoch counter and a registry that is updated on every change (§3).** Searching every
    reachable state of the model:

    ```
    | design                                    | states | all explored | in-place upgrade | I1 runs <= registry | I2 registry = entitlement |
    |-------------------------------------------|--------|--------------|------------------|---------------------|---------------------------|
    | binned dies, bare tokens                  | 385    | yes          | yes              | broken in 1         | holds                     |
    | binned dies, device-bound tokens          | 225    | yes          | yes              | broken in 2         | holds                     |
    | binned dies, bound tokens with revocation | 4,624  | yes          | yes              | holds               | holds                     |
    | same, registry not updated on changes     | 18,225 | yes          | yes              | broken in 1         | broken in 1               |
    | per-tier dies, bare tokens                | 9      | yes          | no               | holds               | holds                     |
    ```

    With bare tokens `replay(0,token 1:2@0)` breaks I1 in one step; with binding alone `downgrade(1,0) -> replay(1,token 1:2@0)`
    does in two (the RFD's reusable stolen token, as a returned chip's old token); an unsynchronised registry breaks
    both invariants at the first sale (`upgrade(0,1)`); binding with revocation and a synchronised registry has no
    violation in `4,624` states. Per-tier dies hold both invariants in `9` states but cannot be upgraded in place, so
    every tier change is a die swap, and the model assumes the replaced die is destroyed
    (`test_bare_tokens_let_another_chips_token_raise_a_low_tier_chip_in_one_step`,
    `test_device_binding_alone_still_lets_a_token_from_before_a_downgrade_come_back`,
    `test_binding_with_revocation_and_a_synced_registry_has_no_violation_anywhere_in_the_model`,
    `test_a_registry_that_is_not_updated_breaks_both_invariants_at_the_first_sale`,
    `test_per_tier_dies_hold_the_invariants_but_cannot_be_upgraded_in_place`). The shortest violations equal those found
    by enumerating every action sequence, and a separately written search visits the same number of states
    (`test_the_search_finds_the_same_shortest_violations_as_enumerating_every_action_sequence`,
    `test_an_independent_search_visits_the_same_number_of_worlds`). A leased unlock can be sold and ended repeatedly
    without breaking either invariant, but each end of lease uses one epoch, and when the counter is full only a die swap
    lowers the chip (`test_a_leased_unlock_can_be_sold_and_ended_repeatedly_without_breaking_either_invariant`,
    `test_a_full_epoch_counter_removes_in_place_downgrade_but_not_replacement`).

13. **Auditing absence by sampling is cheap when cheating is common and costly when it is rare (§4).** For a lot of
    100,000 chips and 95% confidence:

    ```
    | cheated chips | share of lot | detect 1.0 | detect 0.9 | detect 0.5 | detect 0.2 | lot destroyed at 1.0 |
    |---------------|--------------|------------|------------|------------|------------|----------------------|
    | 1,000         | 1.00%        | 298        | 331        | 597        | 1,495      | 0.30%                |
    | 100           | 0.10%        | 2,950      | 3,278      | 5,902      | 14,756     | 2.95%                |
    | 10            | 0.01%        | 25,886     | 28,762     | 51,773     | none       | 25.89%               |
    ```

    Inspecting `298` chips finds a 1% cheat, destroying `0.30%` of the lot, and finding `10` cheated chips takes
    `25,886`, `25.89%` of it; if inspecting a cheated chip reveals it only `0.2` of the time, no sample size finds `10`
    (`test_inspections_needed_is_the_exact_threshold`, `test_no_amount_of_inspection_suffices_when_detection_is_too_weak`,
    `test_a_threshold_that_is_an_exact_tie_is_not_pushed_up_by_rounding`). From the other side, with `300` chips inspected
    a manufacturer can ship `992` cheated chips and escape with probability at least 5%, or `230` at 50%; with `3,000`
    inspected, `98` and `22` (`test_the_evasion_budget_is_the_largest_cheat_that_still_escapes`). Cheating on `100` chips
    per lot is missed with probability `0.7404` in one audited lot and `0.0495` across 10, so repeat audits matter more
    than one large one (`test_repeated_lots_compound_the_chance_of_never_being_caught`). A simulated audit agrees with the
    closed form, `0.4630` against `0.4571` for one case (`test_a_simulated_audit_finds_cheating_as_often_as_the_closed_form_says`).
    Comparison with design files passes a lot that declares the reference design whatever was fabricated, by
    construction in this model, because it reads only what the manufacturer declares
    (`test_design_file_comparison_reads_only_what_the_manufacturer_declares`). Detection probability is an input, so the
    model cannot say whether X-ray, decapping, delayering or functional testing reach any of these values.

## Open questions

1. *"What's the cost premium for separate dies vs. binning?"* **Answered in the model, under its assumptions:** no
   silicon premium at the illustrative tiers (a saving of `12.5%` to `18.2%`), a premium when the tiers are close
   together, and a one-off cost whose size the model cannot supply (findings 1 to 6). No real prices or yields were used.
2. *"Can verification authorities realistically inspect chip layouts?"* **Partially.** For an assumed per-chip
   detection probability the model says how many chips an authority must destroy and how much a manufacturer can hide
   (finding 13). Whether any of the RFD's five techniques reaches that probability, and whether authorities can obtain
   chips at that volume, is not answered.
3. *"How to handle chips that are legitimately sold at one tier but later RMA'd to another?"* **Answered in the model,
   with residual risk:** an in-place change needs device binding, an epoch counter and a registry updated on every
   change; per-tier dies need a die swap and rely on the replaced die being destroyed (finding 12).
4. *"Are there legitimate use cases for software-limited features that governance should accommodate?"* **Not
   addressed.** This is a policy question; the models treat any feature governance needs locked as one to lock and give
   no view on which features qualify.
5. *"How to handle feature unlocking that's legitimately sold (like Intel On Demand)?"* **Answered in the model, with
   residual risk:** a leased unlock is the same machinery as a tier change. Each end of lease uses one epoch of a
   finite counter, after which only a die swap lowers the chip, and the manufacturer's signing key is a single point of
   failure for every token design (findings 10 and 12).

## Not established

- Nothing here was measured on real dies, fuses, fabs, prices or inspection tools. Defect density, floorplan, tier
  block counts, demand mix, spare policy, wafer cost and one-off design cost are assumptions, and the sign of the
  silicon premium flips with the spacing of the tiers (finding 3).
- The attacker budgets (fault injections, circuit edits) are counts, not costs. How hard a real fuse bridge, rail
  reconnection or decision-line edit is, and whether one edit can force a given decision, is not modelled, so the matrix
  shows what needs what and not how expensive it is. Probing, side channels and bugs in the unlock path are outside it.
- Binning here is limited by defects only. Real binning also sorts by frequency, leakage and power, and the flexibility
  lost by committing wafer starts to separate dies is not modelled.
- The audit takes the per-chip detection probability as an input. How well X-ray, decapping, delayering or functional
  testing reveal absent or dormant circuits on real chips is not established, nor whether an authority can obtain and
  inspect chips at the volumes computed.
- The tier-change search covers two chips, three tiers and an epoch counter of two. It assumes a replaced die is
  destroyed and the registry is a trusted record; return logistics, grey-market resale and a compromised registry are
  outside it. A search with no violation is a statement about this model, not about hardware or larger fleets.
- The signatures and the fuzzy extractor are real code, but the fuses, the antifuse rail and the PUF are simulated
  (`aisecurity_rfds.common.nvm`, `aisecurity_rfds.common.puf`); the PUF field-failure rates describe that noise model,
  not any silicon PUF. Open question 4 (legitimate software-limited features) is not addressed.

## Attacks tested

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Privileged software sets the feature flag | firmware flag | bypassed, and the report says off |
| One fault injection on a single fuse read | one fuse copy | bypassed, chip reports it |
| Two fault injections against three fuse copies read by majority | three copies | bypassed, chip reports it |
| One circuit edit on a single fuse; two to hide it | one fuse copy | bypassed with one edit, hidden with two |
| One circuit edit on the antifuse rail | rail read by a self-test | chip bricked, feature stays off |
| Two circuit edits on the antifuse rail | rail read by a self-test | bypassed and hidden |
| A token minted for another chip presented to this one | bare token | bypassed |
| The same | device-bound token | blocked |
| This chip's own token from before a downgrade | device-bound token | bypassed |
| The same | device-bound token with epoch counter | blocked |
| Leaked signing key and manufacturer database | any token design | bypassed |
| The same | PUF-bound key | blocked |
| Readout of the chip's PUF response | PUF-bound key | bypassed |
| Up to three edits and three faults | separate die with no circuit | blocked |
| Another chip's token replayed (tier-change model) | bare tokens | breaks I1 in one step |
| Token from before a downgrade replayed (tier-change model) | device-bound tokens | breaks I1 in two steps |
| The same sequences | binding, revocation and a synchronised registry | no violation in the model |
| A sale or return without a registry update | unsynchronised registry | breaks I1 and I2 at the first sale |
| A cheating lot declares the reference design | comparison with design files | passes the lot |
| 100 cheated chips in a lot, 300 inspected | sampling audit, perfect detection | missed with probability `0.7404` |

## Related RFDs

- RFD 031 (Chip Registry for AI Governance): the registry whose accuracy is invariant I2 here, and whose lag the
  unsynchronised-registry design shows.
- RFD 034 (License Replay Prevention) and RFD 032 (Offline Licensing for Compute Rationing): the same epoch counter and
  manufacturer-signed tokens, used here to gate a feature rather than a quota.
- RFD 033 (Fixed Set Cluster Size Limiting): its pod binding and size cap are fused features of the kind lockout is
  meant to protect.
- RFD 036 (Tamper-Respondent PUF Enclosures) and RFD 028 (Retrofittable Tamper Detection System): what a fault
  injection or an edit costs, and whether it is noticed, is their subject; here both are counts.
