# RFD 002 - MoE Expert Fingerprinting

Upstream: <https://projects.aisecurity.forum/rfd/002/> (Gabriel Kulp, state: idea).

## Status

**Kind: simulation.** The RFD proposes that hardware observations of a mixture-of-experts model during inference might
reveal which expert subnetworks were activated, and so hint at the tokens being processed, without any software access.
It asks three open questions: how many experts need to be distinguishable for practical token recovery, whether
fingerprinting transfers across model versions or quantization levels, and whether routing noise can be added without
degrading model quality. This package has no GPU, no real mixture-of-experts model and no sensor. It builds the model
that such an experiment would be compared against. A small mixture-of-experts language model is trained here from
scratch, so that its router is learned and its quality is a measured loss. Every expert is given an assumed physical
signature, and a sensor reports the sum of the signatures of the experts that ran in a block, plus Gaussian noise. An
attacker who holds an open copy of a model builds a table of how often each token made each pair of experts run, and
then names the token behind a position from what the sensor said about it. All three open questions get an answer
inside this model and none gets one about real hardware. Every number printed is a property of the toy model, of its
synthetic language and of the assumed sensor, and not a measurement.

## What is implemented

- `moe.py`: a mixture-of-experts language model in numpy with hand-written gradients (checked against finite
  differences in the tests) and an Adam trainer. It predicts the next token from the current one. An embedding plus a
  scaled random context vector, which stands in for everything else a real router sees, goes through 3 blocks. Each
  block routes a token to its top 2 of 16 small ReLU MLPs through a learned linear router, mixes their outputs with the
  softmax of the chosen router logits, adds the result to a residual stream and is trained with a load-balancing loss.
  `Chain` is the synthetic first-order language that the model learns, `quantize` rounds weights per tensor to a given
  number of bits, and `routing_noise` supplies Gumbel noise for the router logits: added before a top-1 choice it samples
  each expert with probability softmax(logits / tau). The trained toy reaches an excess loss of 0.065 nats over the
  entropy of its language (0.775 nats), against 3.013 with the experts removed and 3.384 for a uniform guess, and its
  router spreads the work: the least busy expert gets 0.030 of the dispatches, the busiest 0.103, and an even share
  would be 0.062.
- `signatures.py`: the assumed sensor. Every expert has a signature vector of measurable features, and experts may
  share a signature, so the number of distinct signatures is the number of distinguishable experts. A block's reading is
  the sum of the signatures of its two experts plus Gaussian noise of a given sd. The module has the nearest-signature
  decoder, the exact pairwise confusion probabilities, closed-form bounds on decoding which pair ran (the largest single
  confusion below, the union bound above), the accuracy that no sensor can exceed when signatures repeat, and signature
  drift, where another build of the same experts measures differently.
- `attack.py`: the attacker. `build_profile` runs a model on every token in 200 random contexts and counts which pair of
  experts ran in each block, with Laplace smoothing. `attack` draws positions of the target model, reads the sensor on
  the first blocks and returns the token with the highest naive-Bayes posterior (a uniform prior over tokens, blocks
  independent given the token, the signatures and the sensor noise known). `route_change` measures the share of (token, block)
  positions at which two models choose different experts, and `deterministic_ceiling` the share of tokens that have a route of their own when
  no context vector is present.
- `__main__.py`: prints the assumptions and five numbered sections with every table and number quoted below. §1 is how
  many experts must be distinguishable (open question 1), §2 what the sensor must resolve, §3 routes that depend on
  context, §4 transfer to quantized, fine-tuned and retrained models and to signature drift (open question 2), and §5
  routing noise (open question 3).
- `aisecurity_rfds.common.report` formats the tables, `aisecurity_rfds.common.rng` seeds every draw, and
  `aisecurity_rfds.common.stats` supplies the Wilson intervals.

A token accuracy below is the share of 800 uniformly drawn tokens, each at a random context, that the attacker names
correctly from one position's sensor readings on the first 1, 2 or 3 blocks. Chance is 1/64, which prints as 0.016.
Sensor noise is an sd in units of the spread of one signature feature, and the features of a signature are drawn from
N(0, 1).

## Reproduce

```bash
pytest tests/rfd_002_moe_expert_fingerprinting
python -m aisecurity_rfds.rfd_002_moe_expert_fingerprinting   # prints every table quoted here (about 26 s)
python scripts/mutation_check.py 2                # 131 seeded bugs; every one must be caught
```

Add `--jobs 2` to the last command to test two mutated copies at a time.

Section numbers (`§N`) below are the numbered sections of that program's output. Figures in backticks are copied from
it, and a test (`tests/test_readme_consistency.py`) fails if one is not printed there.

## Findings

1. **Token recovery needs several blocks or many distinguishable experts (§1).** The attacker sees exact routes, with no
   sensor noise. A distinguishable expert here is one whose signature differs from every other expert's, and when fewer
   than 16 are distinguishable some experts share a signature, so a block's pair is known only up to those classes. With
   three blocks observed the share of tokens named is `0.256` for 2 distinguishable experts, `0.704` for 4, `0.939` for 8
   and `0.971` for 16, against chance `0.016`. A single block with all 16 distinguishable gives `0.507`, and two blocks
   give `0.901`. One block's pair carries 6.9 bits, more than the 6.0 that naming one of 64 tokens needs, so capacity is
   not what limits it. What limits it is that a token's route changes with the context vector and that several tokens
   can share a route. With the context vector removed, the share of tokens that have a route of their own is `0.734`
   after 1 block and `0.984` after 2, and an exact attacker reaches those ceilings exactly. Backed by
   `test_token_recovery_rises_with_the_number_of_distinguishable_experts`,
   `test_token_recovery_rises_with_the_number_of_blocks_observed` and
   `test_with_no_context_vector_the_exact_attack_reaches_exactly_the_share_of_tokens_with_their_own_route`.

   ```
   | distinguishable experts | distinct set signatures | 1 block | 2 blocks | 3 blocks |
   |-------------------------|-------------------------|---------|----------|----------|
   | 2                       | 3                       | 0.049   | 0.124    | 0.256    |
   | 4                       | 10                      | 0.126   | 0.424    | 0.704    |
   | 8                       | 36                      | 0.287   | 0.780    | 0.939    |
   | 16                      | 120                     | 0.507   | 0.901    | 0.971    |
   ```

2. **A noisy sensor loses the pair of experts before it loses the token (§2).** The table gives, for naming the pair of
   experts in one block, the simulated accuracy, an accuracy that the union bound over pairwise confusions guarantees,
   an accuracy that the largest single confusion says cannot be exceeded, and the token accuracy of the Bayes attacker
   over three blocks. At noise sd 0.5 only `0.541` of the pairs are named correctly (the bounds allow `0.154` to `0.789`, and
   the simulation sits between them in every row), yet `0.901` of the tokens are named, because the decoder keeps a
   likelihood for every pair instead of committing to one and combines three blocks. Token accuracy falls to `0.586` at
   noise sd 1.0 and to `0.225` at 2.0. Backed by `test_set_decoding_collapses_before_token_recovery_does`,
   `test_simulated_set_accuracy_sits_between_the_two_bounds` and `test_token_recovery_falls_as_the_sensor_gets_noisier`.

   ```
   | noise sd | set accuracy | guaranteed at least | cannot exceed | token accuracy |
   |----------|--------------|---------------------|---------------|----------------|
   | 0.1      | 0.998        | 0.998               | 0.999         | 0.961          |
   | 0.25     | 0.897        | 0.867               | 0.931         | 0.970          |
   | 0.5      | 0.541        | 0.154               | 0.789         | 0.901          |
   | 1.0      | 0.189        | 0.000               | 0.661         | 0.586          |
   | 2.0      | 0.061        | 0.000               | 0.583         | 0.225          |
   ```

3. **Measuring more features helps as much as lowering the noise (§2).** At noise sd 0.5 token accuracy is `0.295` with
   1 feature per signature, `0.594` with 2, `0.916` with 4 and `0.968` with 8. Each row of every table uses its own
   random signatures, and the second table shows how much that alone moves a figure: over 20 other draws of the
   4-feature signatures, set accuracy averages `0.570` with an sd of `0.063` (from `0.444` to `0.690`) and token accuracy
   averages `0.900` with an sd of `0.022`. That sd includes the sampling error of 2000 and 800 trials. Differences of
   about that size between rows that use different signatures are therefore not findings. Backed by
   `test_measuring_more_features_raises_token_recovery` and
   `test_set_accuracy_is_reproducible_and_rises_with_the_number_of_features`.

   ```
   | features | set accuracy | token accuracy |
   |----------|--------------|----------------|
   | 1        | 0.041        | 0.295          |
   | 2        | 0.148        | 0.594          |
   | 4        | 0.630        | 0.916          |
   | 8        | 0.926        | 0.968          |
   ```

   ```
   | quantity       | mean  | sd    | min   | max   |
   |----------------|-------|-------|-------|-------|
   | set accuracy   | 0.570 | 0.063 | 0.444 | 0.690 |
   | token accuracy | 0.900 | 0.022 | 0.863 | 0.943 |
   ```

4. **A router that sees more than the token is harder to read (§3).** The context vector stands in for everything else
   that a router sees, and scaling it up makes a token's route vary more from position to position. With three exact
   blocks token accuracy falls from `0.971` at scale 0.5 to `0.851` at 1 and `0.461` at 2, and with one block from
   `0.521` to `0.371` and `0.176`. In this toy the context is noise to the language model, because it says nothing about
   the next token, so a larger scale also costs quality: the excess loss rises from `0.065` to `0.166` and `0.732`. In a
   real model the router's other inputs are informative and that cost has no counterpart here. The finding is how
   accuracy depends on how much of the router's input is not the token, and not the quality figures. The attacker also
   treats positions as independent, and does not use the other tokens of a sequence. Backed by
   `test_a_larger_context_vector_lowers_recovery_and_costs_quality`.

   ```
   | context scale | excess loss | 1 block | 3 blocks |
   |---------------|-------------|---------|----------|
   | 0.5           | 0.065       | 0.521   | 0.971    |
   | 1             | 0.166       | 0.371   | 0.851    |
   | 2             | 0.732       | 0.176   | 0.461    |
   ```

5. **A table built on one model works on its quantized and fine-tuned copies but not on an independently trained model
   (§4).** The attacker builds the table on the original model and attacks another one. Quantizing to int8 with the
   router kept changes the experts in `0.005` of the blocks (`0.017` with every weight quantized), and the old table
   names `0.974` of the tokens, against `0.969` for the original, which is no difference given the sampling error. With
   every weight at int4, `0.301` of the blocks choose other experts and the old table still names `0.958`, because the
   blocks that kept their experts carry the decision. At int3 `0.586` of the blocks change and the old table falls to
   `0.829`, while a table rebuilt on the target names `0.949`. Fine-tuning on a mixture with a second language changes
   `0.240` of the blocks (10% new language) and `0.371` (50%), with `0.950` and `0.912` from the old table and `0.973`
   and `0.965` from a rebuilt one. At 50% the model's loss on the original language has risen to an excess of `0.770`.
   A model trained again from another seed has the same quality (`0.065`) but chooses other experts in `0.992` of the
   blocks, the old table names `0.004` of its tokens, which is below chance, and a table rebuilt on it names `0.969`. A
   derived copy therefore keeps its fingerprint and a retrained model does not, and an attacker needs an open copy of
   the target, or of something it was derived from, to rebuild the table. Backed by
   `test_a_table_built_on_the_original_works_on_an_int8_copy`,
   `test_coarser_quantization_changes_more_routes_and_a_rebuilt_table_repairs_the_damage`,
   `test_a_table_survives_fine_tuning_with_a_loss_of_some_accuracy_and_a_rebuilt_table_loses_none` and
   `test_a_table_is_useless_on_an_independently_trained_model_until_it_is_rebuilt_on_it`.

   ```
   | target model                    | blocks with other experts | excess loss, original language | table from the original | table rebuilt on the target |
   |---------------------------------|---------------------------|--------------------------------|-------------------------|-----------------------------|
   | the original                    | 0.000                     | 0.065                          | 0.969                   | 0.970                       |
   | int8, router kept               | 0.005                     | 0.066                          | 0.974                   | 0.976                       |
   | int8, all weights               | 0.017                     | 0.066                          | 0.975                   | 0.978                       |
   | int4, router kept               | 0.108                     | 0.114                          | 0.966                   | 0.968                       |
   | int4, all weights               | 0.301                     | 0.129                          | 0.958                   | 0.969                       |
   | int3, router kept               | 0.218                     | 0.286                          | 0.965                   | 0.975                       |
   | int3, all weights               | 0.586                     | 0.419                          | 0.829                   | 0.949                       |
   | fine-tuned on 10% new language  | 0.240                     | 0.139                          | 0.950                   | 0.973                       |
   | fine-tuned on 50% new language  | 0.371                     | 0.770                          | 0.912                   | 0.965                       |
   | trained again from another seed | 0.992                     | 0.065                          | 0.004                   | 0.969                       |
   ```

6. **Signature drift costs accuracy only once it is a sizeable part of the signature (§4).** Another build of the same
   experts is measured with signatures that differ from the attacker's table by Gaussian noise whose sd is a share of
   the root-mean-square signature, here with sensor noise sd 0.25. At a drift of 0.1 token accuracy is `0.974`, against
   `0.958` with none, which is sampling error, and at 0.3 it is `0.739`. The drift model moves every entry of every
   signature independently, and real differences between chips, firmware or temperatures are more likely to be
   structured, for instance a common gain and offset that a calibration step could remove. Backed by
   `test_signature_drift_lowers_recovery_once_it_is_a_sizeable_part_of_the_signature`.

   ```
   | drift (share of rms signature) | token accuracy |
   |--------------------------------|----------------|
   | 0.0                            | 0.958          |
   | 0.1                            | 0.974          |
   | 0.3                            | 0.739          |
   ```

7. **Routing noise costs little only while it changes little, and then it hides little (§5).** Gumbel noise of scale tau
   is added to the router logits before the top-2 choice, and the mixing weights still use the clean logits. The
   quality cost is the first table. The trained model's excess loss is `0.065` with no noise, `0.086` at tau 0.25,
   `0.119` at 0.5, `0.441` at 1.0 and `1.289` at 2.0. An attacker who knows tau and builds its table under the noise
   names `0.965` of the tokens at tau 0.25 (from `0.976` with no noise), `0.901` at 0.5, `0.610` at 1.0 and `0.176` at
   2.0, and an attacker who ignores the noise names `0.953`, `0.836`, `0.463` and `0.147`. So the noise that costs under
   0.03 nats takes one or two points of accuracy. Tau 1.0, the smallest level tested that takes the ignorant attacker
   below one half, costs 0.376 nats over the noiseless loss, which multiplies the perplexity by 1.46. Backed by `test_small_routing_noise_costs_almost_no_quality_and_large_noise_costs_a_lot`,
   `test_an_attacker_who_ignores_the_noise_loses_little_at_small_tau_and_most_of_the_accuracy_at_large_tau`,
   `test_a_table_that_knows_the_noise_recovers_part_of_the_loss` and
   `test_routing_noise_changes_the_expert_choice_more_and_more_often_as_tau_grows`.

   ```
   | tau  | excess loss: trained | control | noise-trained |
   |------|----------------------|---------|---------------|
   | 0.0  | 0.065                | 0.034   | 0.037         |
   | 0.1  | 0.078                | 0.043   | 0.048         |
   | 0.25 | 0.086                | 0.049   | 0.052         |
   | 0.5  | 0.119                | 0.070   | 0.064         |
   | 1.0  | 0.441                | 0.340   | 0.283         |
   | 2.0  | 1.289                | 1.225   | 1.053         |
   ```

8. **An attacker who can watch the same token several times averages the noise away (§5).** When the same token in the
   same context is seen 4 times, each time with fresh routing noise, token accuracy is `0.993` at tau 0.5, which is above the `0.976` with no noise, then `0.925` at 1.0 and `0.396` at 2.0, against `0.176` for a single
   look. A plausible reading of the gain over the noiseless case is that noisy choices reveal the router's runner-up
   experts, and this package does not test that. Whether an attacker can obtain repeats is a property of the deployment:
   it needs the same token in the same context to be processed again with independent noise. Backed by
   `test_seeing_the_same_token_several_times_defeats_moderate_noise` and
   `test_more_repeats_never_hurt_the_attacker_much_and_help_most_when_the_noise_is_large`.

   ```
   | tau  | trained, table without noise | trained, table knows the noise | trained, same token seen 4 times | control, table knows the noise | noise-trained, table knows the noise |
   |------|------------------------------|--------------------------------|----------------------------------|--------------------------------|--------------------------------------|
   | 0.0  | 0.970                        | 0.976                          | 0.978                            | 0.963                          | 0.970                                |
   | 0.1  | 0.960                        | 0.965                          | 0.981                            | 0.966                          | 0.975                                |
   | 0.25 | 0.953                        | 0.965                          | 0.985                            | 0.954                          | 0.964                                |
   | 0.5  | 0.836                        | 0.901                          | 0.993                            | 0.900                          | 0.894                                |
   | 1.0  | 0.463                        | 0.610                          | 0.925                            | 0.641                          | 0.645                                |
   | 2.0  | 0.147                        | 0.176                          | 0.396                            | 0.200                          | 0.221                                |
   ```

9. **Training with the noise buys back some quality and no secrecy (§5).** The control is the trained model fine-tuned
   for 300 more steps without noise, and the noise-trained model is the same fine-tuned for 300 steps at tau 0.5. At
   tau 1.0 the noise-trained model's excess loss is `0.283`, the control's `0.340` and the original's `0.441`, and at
   tau 2.0 they are `1.053`, `1.225` and `1.289`. Most of the improvement over the original comes from the extra steps,
   and what the noise adds is the gap between the last two columns, about 0.06 nats at tau 1.0 and 0.17 at 2.0. The
   attacker's accuracy against the noise-trained model is not measurably lower than against the control: `0.645` against `0.641` at tau
   1.0 and `0.221` against `0.200` at 2.0, within a sampling error of about 0.02. Noise-aware training lowers the loss
   and not the leak. Backed by
   `test_noise_aware_training_lowers_the_loss_under_noise_compared_with_equally_long_plain_training`.

## Open questions

1. *"How many experts need to be distinguishable for practical token recovery?"* **Partially.** In the model, with the
   exact routes of three blocks, 2 distinguishable experts of 16 name 0.256 of the tokens, 4 name 0.704, 8 name 0.939 and
   16 name 0.971, and fewer observed blocks need more of them: one block with all 16 distinguishable names 0.507
   (finding 1). Once the sensor's noise reaches about half the spread of a signature feature the limit becomes the noise
   and not the number of distinguishable experts, and a second feature or a lower noise helps more than a bigger model
   (findings 2 and 3). A router that depends more on context than on the token lowers every figure (finding 4). The
   threshold in a real model depends on its vocabulary, its number of blocks, how much of the router's input is the
   token and the sensor's noise relative to the differences between expert signatures, and whether real experts have
   distinguishable signatures at all is not established.
2. *"Does expert fingerprinting transfer across model versions or quantization levels?"* **Partially.** In the model a
   table built on the original keeps most of its accuracy on quantized copies (0.974 at int8, 0.958 at int4 with every
   weight quantized, 0.829 at int3) and on copies fine-tuned on a mixture with a second language (0.950 and 0.912), and
   it is useless on a model trained again from another seed (0.004), where a table rebuilt on the model gets 0.969
   (finding 5). Signatures that differ between builds cost accuracy only once the difference is about a third of the
   signature (finding 6). Whether a real fine-tune, distillation or pruning of a real model keeps its routing, and
   whether a real expert's signature survives another chip, firmware or temperature, is not established.
3. *"Can routing noise be added without degrading model quality?"* **Partially.** In the model noise up to tau 0.25
   costs under 0.03 nats of excess loss and takes one or two points off the attacker's accuracy, tau 1.0, the smallest
   level tested that brings the ignorant attacker below one half, costs 0.376 nats, and an attacker who can observe the
   same token repeatedly recovers most of what the noise hides (findings 7 and 8). Training with the noise lowers the loss but not the leak
   (finding 9). The cost in the loss of a real model, and other ways of removing the signal such as constant-power
   experts, padding or randomised placement, are not studied.

## Not established

- Nothing was measured on a GPU or on a real mixture-of-experts model. The toy (64 tokens, 3 blocks of 16 experts, width
  24, a first-order synthetic language) and every property of the sensor (random 4-feature signatures, additive
  Gaussian noise, known block boundaries, one token in flight) are assumptions. Whether real experts have physical
  signatures that a real sensor can tell apart, and at what noise, is the central empirical question of the RFD and is
  not established here.
- The toy has no attention, no batching and no sequence. A token's route varies with a random context vector that says
  nothing about the next token, whereas a real router sees a contextual hidden state, and concurrent requests in one
  batch would mix their experts. Recovering text from a real stream, with the other tokens of a sequence and a language
  model's priors, is not modelled, and it would probably recover more than the position-by-position decoder here.
- The attacker is a naive-Bayes template attacker with a uniform prior, exact white-box tables and exactly known
  signatures and noise. It is not shown to be the best attacker, so its accuracies are neither a bound on what an
  attacker can reach in the model nor a worst case for a defender, and the assumption of exact signatures and noise
  flatters it where a real sensor would have to be calibrated.
- Every figure is one trained model per configuration (one other seed appears in the transfer table, where the two
  seeds agree), 800 tokens per accuracy, 2000 trials per set accuracy and one draw of signatures per row, except the
  spread table. A sampling error is about 0.01 to 0.02, the draw-to-draw sd of set accuracy is 0.06, and differences
  smaller than that are not resolved. Other seeds move the details, and the spread over training seeds is not reported.
- Transfer is tested only on per-tensor rounding of the toy, on fine-tuning with a mixture of two synthetic languages
  and on a model trained again from another seed. Group-wise or calibrated quantization schemes, distillation, pruning,
  a different number of experts, router retraining and real model versions are not modelled.
- Routing noise is Gumbel noise on the router logits, and quality is held-out cross-entropy on the synthetic language.
  Padding, constant-power experts, randomised placement, downstream task quality and the effect on a real model's loss
  are not studied, and the repeated-observation attacker's access to repeats is an assumption, not a measurement.
- Defences (RFD 004), covert use of expert choice by a trained model (RFD 003) and the physical sensors themselves (RFD
  001, RFD 012) are outside the package. The only defence-like variables are the sensor's noise and features, the router's
  context dependence, quantization and the routing noise.

## Attacks tested

Each row is an attack run against the toy, the property of the model or sensor that is under test as the defence, and the
figures the program prints for it. Unless stated, the sensor is exact, 16 experts are distinguishable and 3 blocks are
observed.

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Token from the exact routes of 3 blocks, 2 distinguishable experts | Experts that share signatures | `0.256`, chance `0.016` |
| The same with all 16 experts distinguishable, 1 block | A single observed block | `0.507`, and `0.971` with 3 blocks |
| Token from a noisy sensor, noise sd 1.0 | Sensor noise | `0.586` of tokens, with `0.189` of the pairs named correctly |
| The same at noise sd 2.0 | A noisier sensor | `0.225` of tokens |
| One feature measured instead of four, noise sd 0.5 | A less informative sensor | `0.295`, against `0.916` with four |
| A model trained with context scale 2 | Routes that vary with context | `0.461` |
| Table from the original on its int8 copy, router kept | Quantization | `0.974` |
| The same on an int3 copy, every weight quantized | Coarse quantization | `0.829`, and `0.949` with a rebuilt table |
| The same on a copy fine-tuned on 50% new language | Fine-tuning | `0.912` |
| The same on a model trained again from another seed | Retraining | `0.004`, and `0.969` with a rebuilt table |
| Signatures that differ by 0.3 of their rms, noise sd 0.25 | A different physical build | `0.739` |
| Table that ignores the routing noise, tau 1.0 | Gumbel noise on the router | `0.463` |
| Table that knows the noise, tau 1.0 | The same, attacker adapted | `0.610` |
| The same token seen 4 times, tau 1.0 | The same, attacker repeats | `0.925` |
| Table that knows the noise against a noise-trained model, tau 1.0 | Noise-aware training | `0.645`, and `0.641` for the control |

## Related RFDs

- RFD 001, Side-Channel Leakage from LLM Inference: the physical channel (power, EM, acoustic, thermal and software
  sensors) and a topic attack on a fixed, assumed router. This package takes the sensor as a signature vector, trains
  the router and asks for tokens and not topics. Whether a real sensor reports expert signatures at the noise assumed
  here is what RFD 001's experiment would show. No code is shared.
- RFD 003, Trained Covert Communication via Side-Channels: a model that chooses its experts to send a message is the
  active counterpart of the passive leak here, and the 6.9 bits that one block's pair of experts can carry is the kind
  of quantity its bandwidth question would use. Not modelled.
- RFD 004, Functional Invariants for Side-Channel Defense: routing noise here is a crude defence, and an invariant that
  makes the signatures of experts indistinguishable would remove the signal instead of randomising the choice. Nothing
  of that kind is implemented.
- RFD 012, Analog Sensors for Compute Verification: the defender's use of analog sensors. The sensor that reads an
  expert signature here is the kind of instrument that a verifier would also use, and nothing here implements a verifier.
