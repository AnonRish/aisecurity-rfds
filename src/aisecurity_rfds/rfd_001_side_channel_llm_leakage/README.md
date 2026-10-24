# RFD 001 - Side-Channel Leakage from LLM Inference

Upstream: <https://projects.aisecurity.forum/rfd/001/> (Gabriel Kulp, state: idea).

## Status

**Kind: simulation.** The RFD hypothesises that a GPU running an LLM leaks information through its power draw,
electromagnetic emissions, sound, heat and the power sensors that software can read, and that an attacker could use this
to recover prompt text, token presence, output logits, semantic categories and possibly the true response behind a
refusal. It proposes an experiment on a frontier accelerator with oscilloscopes, EM probes and software sensors, and
asks three open questions: which channel has the best signal-to-noise ratio for which target, how much a
mixture-of-experts architecture changes the leakage surface, and whether attacks survive virtualization in the cloud.
This package has no GPU and no sensor. It builds the model that such an experiment would be compared against. One
multiplication leaks the Hamming weight of its product; a channel reports the mean leakage over a window of operations
plus noise; five assumed channels (an oscilloscope on the power rail, an EM probe, a microphone, a software power sensor
and a thermal sensor) differ in gain, noise and sample rate. On that model it implements leakage detection (a Welch t
test), a correlation attack on one activation, token identification with error bounds, expert-routing leakage with a
Bayes attacker for the topic of a prompt, a shared sensor for the cloud question, and a response hidden behind a
refusal. Each analysis tool is checked against a closed form, a reference implementation or a hand-worked example on
synthetic traces. All three open questions get an answer inside the model, none gets one about real hardware, and every
number printed is a property of the assumptions and not a measurement.

## What is implemented

- `leakage.py`: the two assumptions that everything else rests on, written as code. A multiplication leaks the Hamming
  weight of its 16-bit two's-complement product (`hw16`, `mac_leakage`). A `Channel` reports the mean leakage over a
  window of operations (the operation rate divided by its sample rate), with one noise source added before the
  averaging, which averages down, and one after it, which does not. `ToyLayer` is a first layer with known int8 weights
  (12 x 24) and an int8 embedding for each of 64 tokens, so one token's leakage is a vector of 288 values. The five
  assumed channels are `SCOPE`, `EM_PROBE`, `ACOUSTIC`, `SOFTWARE` and `THERMAL`, with round-number gains, noise and
  sample rates; `Channel.with_tenants` adds the noise of other workloads on a shared sensor in quadrature.
- `analysis.py`: tools that take plain arrays, so they could be pointed at measured traces, though here they are run
  only on simulated ones. `welch_t` and `tvla_traces_needed` are the fixed-versus-random leakage test and its
  closed-form trace count. `cpa_recover` is a correlation attack on one int8 activation, `correlation_detection_traces`
  its Fisher-transform detection count, and `closest_confusable` finds, for one activation, the other that leaks most
  nearly the same. `identification_error_bounds` brackets the error of naming one of many templates from a noisy
  reading (the worst single confusion below, a union bound above), `identification_accuracy` simulates it,
  `repeats_needed` finds how many averaged runs the bound guarantees, and `noise_free_identification_accuracy` gives the
  ceiling that remains when identical templates cannot be told apart.
- `routing.py`: a mixture-of-experts router as a table, P(expert pair | topic), and a fixed power shift for each expert,
  so that the pair running in a slot shifts that slot's mean power. It has the mutual information between pair and topic,
  the accuracy of reading the pair from one noisy level, and a Bayes attacker who infers the topic from the levels of
  successive tokens.
- `scenarios.py`: the leakage test and a dense-model topic attacker on the toy layer, the search for the number of tokens
  an attacker needs, and the refusal pipeline with three modes (no refusal, a filter applied after decoding, a check
  applied before decoding).
- `__main__.py`: prints the assumptions and seven numbered sections with every table and number quoted below. §1 is
  leakage detection, §2 the correlation attack, §3 naming a token, §4 the channel against target matrix (open question
  1), §5 dense against mixture-of-experts (open question 2), §6 a shared sensor (open question 3) and §7 the refusal.
- `aisecurity_rfds.common.report` formats the tables, `aisecurity_rfds.common.rng` seeds every draw, and
  `aisecurity_rfds.common.stats` supplies the Wilson intervals.

The channels, as the program prints them first. Every value is an assumption chosen as a round number:

```
| channel                  | gain | noise per op | noise per sample | samples per second | ops per sample | samples per token |
|--------------------------|------|--------------|------------------|--------------------|----------------|-------------------|
| power rail, oscilloscope | 1    | 12           | 0                | 1,000,000,000      | 1              | 20,000,000        |
| EM near-field probe      | 0.4  | 4            | 0                | 1,000,000,000      | 1              | 20,000,000        |
| acoustic, microphone     | 0.2  | 5            | 0.002            | 50,000             | 20,000         | 1,000             |
| software power sensor    | 1    | 0            | 0.01             | 50                 | 20,000,000     | 1                 |
| thermal sensor           | 0.05 | 0            | 0.004            | 10                 | 100,000,000    | 0.2               |
```

## Reproduce

```bash
pytest tests/rfd_001_side_channel_llm_leakage
python -m aisecurity_rfds.rfd_001_side_channel_llm_leakage   # prints every table quoted here (about 11 s)
python scripts/mutation_check.py 1                # 82 seeded bugs; every one must be caught
```

Section numbers (`§N`) below are the numbered sections of that program's output. Figures in backticks are copied from
it, and a test (`tests/test_readme_consistency.py`) fails if one is not printed there.

## Findings

1. **The leakage test does what its formula says (§1).** A fixed-versus-random Welch t test flags a sample when |t| is
   above 4.5. For a planted shift of d standard deviations, the closed-form trace count should give 90% power, and the
   simulation agrees: it flags `0.889`, `0.900`, `0.889` and `0.902` of 2000 simulated tests at the counts below (one
   standard error at 2000 tests is 0.007). With no leak, both groups random tokens and 10,000 traces, the oscilloscope
   trace gives `0 of 288 samples flagged, largest |t| 3.0`. Backed by
   `test_the_t_test_flags_a_planted_leak_at_the_rate_the_formula_promises`,
   `test_the_t_test_flags_almost_nothing_when_there_is_no_leak` and `test_welch_t_matches_scipy`.

   ```
   | effect d (standard deviations) | traces per group for 90% power | flagged in 2000 simulated tests |
   |--------------------------------|--------------------------------|---------------------------------|
   | 0.5                            | 268                            | 0.889                           |
   | 0.2                            | 1,672                          | 0.900                           |
   | 0.1                            | 6,686                          | 0.889                           |
   | 0.05                           | 26,742                         | 0.902                           |
   ```

2. **A fast channel sees the token in every sample; a slow one averages it away (§1).** The table gives each channel's
   typical best effect (the largest per-sample difference between a fixed token and a random one, in standard
   deviations, median over fixed tokens) and the traces per group a leakage test needs for 90% power. The oscilloscope
   and the EM probe sample every operation: effects of `5.4e-1` and `6.3e-1`, `234` and `167` traces, and with token 0
   fixed they flag `2 / 94 / 217 of 288` and `1 / 114 / 238 of 288` samples after 100, 1,000 and 10,000 traces. The
   microphone, the software sensor and the thermal sensor average each token's 288 multiplications into windows of
   `20,000`, `20,000,000` and `100,000,000` operations. A window longer than the token's operations reports their sum
   divided by the window (`test_a_window_longer_than_the_signal_dilutes_it_by_signal_length_over_window`), so the effect
   falls to `1.1e-2`, `1.9e-4` and `4.8e-6` and the traces needed rise to `5.7e5`, `1.8e9` and `2.9e12`, with the
   trace count growing as the inverse square of the effect
   (`test_tvla_traces_scale_with_the_inverse_square_of_the_effect_and_grow_with_power`). That dilution is a property of
   the toy, which leaves the rest of each window idle; a real decode step is busy throughout it (see Not established),
   so the slow rows say what this model does and not what a real thermal sensor would see.

   ```
   | channel                  | ops per sample | typical best effect d | traces per group to flag (90%) | flagged after 100 / 1,000 / 10,000 traces |
   |--------------------------|----------------|-----------------------|--------------------------------|-------------------------------------------|
   | power rail, oscilloscope | 1              | 5.4e-1                | 234                            | 2 / 94 / 217 of 288                       |
   | EM near-field probe      | 1              | 6.3e-1                | 167                            | 1 / 114 / 238 of 288                      |
   | acoustic, microphone     | 20,000         | 1.1e-2                | 5.7e5                          | 0 / 0 / 0 of 1                            |
   | software power sensor    | 20,000,000     | 1.9e-4                | 1.8e9                          | 0 / 0 / 0 of 1                            |
   | thermal sensor           | 100,000,000    | 4.8e-6                | 2.9e12                         | 0 / 0 / 0 of 1                            |
   ```

3. **Detecting that a secret leaks is cheaper than recovering it (§2).** The correlation attack tries all 256 values of
   one int8 activation against known random weights, one measurement per multiplication, and keeps the value whose
   predicted Hamming-weight leakage correlates best. The Fisher-transform formula says how many samples make the
   correlation of the *correct* prediction exceed the detection threshold with 90% power, which is what a defender who
   knows the secret would need to certify the leak: `17`, `37` and `112` samples at noise of 1, 2 and 4 LU, and the
   simulation flags it `0.963`, `0.940` and `0.925` of the time (the formula is a little conservative). Recovering the
   secret takes more. Activation 77 needs `18`, `40` and `205` samples, and a random activation `137`, `308` and
   `1557`, which is about 8 to 14 times the detection count. The gap comes from ghost hypotheses: a multiplication by
   twice the activation shifts the product by one bit and so leaks almost the same Hamming weight, and the predicted
   leakages of `5 ~ 10 (0.998)` and `100 ~ 50 (0.988)` are nearly identical
   (`test_the_closest_confusable_activation_of_a_doubling_is_the_double`). Activation 0 can never be recovered, because
   its predicted leakage is constant (`test_a_zero_activation_cannot_be_recovered_because_its_prediction_is_constant`).
   The recovery counts are rungs of a ladder that grows by a factor of 1.5, each judged on 100 trials, so they are good
   to about one rung and no better.

   ```
   | noise sd (LU) | rho (secret 77) | samples to flag the leak | flagged in simulation | samples to recover 77 (90%) | samples to recover a random secret (90%) |
   |---------------|-----------------|--------------------------|-----------------------|-----------------------------|------------------------------------------|
   | 1.0           | 0.919           | 17                       | 0.963                 | 18                          | 137                                      |
   | 2.0           | 0.760           | 37                       | 0.940                 | 40                          | 308                                      |
   | 4.0           | 0.504           | 112                      | 0.925                 | 205                         | 1557                                     |
   ```

4. **Naming a token has closed-form bounds that sandwich the simulation, and slow sensors hit a ceiling (§3).** The
   attacker compares one noisy reading of the first-layer product with the 64 token templates. From one oscilloscope run
   the simulation names the token with accuracy `0.838`; the pairwise-error bounds guarantee at least `0.596` and rule
   out more than `0.981` (`test_the_bounds_sandwich_the_simulated_error_at_several_noise_levels`). Averaging 2 identical
   runs gives `0.986` and 4 runs `1.000`. A guaranteed 99% takes 3 averaged runs on the oscilloscope and 2 on the EM
   probe; on the other three channels no number of runs reaches it. A slow sensor reports one number per token, the sum
   of its leakages, and the 64 tokens have only `55 distinct sums`, so even with no noise the best any attacker can do
   is `0.859` (`test_a_noiseless_attacker_reaches_exactly_the_ceiling_it_is_given`).

   ```
   | averaged runs | simulated accuracy | guaranteed at least | cannot exceed |
   |---------------|--------------------|---------------------|---------------|
   | 1             | 0.838              | 0.596               | 0.981         |
   | 2             | 0.986              | 0.982               | 0.998         |
   | 4             | 1.000              | 1.000               | 1.000         |
   | 8             | 1.000              | 1.000               | 1.000         |
   | 16            | 1.000              | 1.000               | 1.000         |
   ```

   ```
   | channel                  | averaged runs for a guaranteed 99% | ceiling even with no noise |
   |--------------------------|------------------------------------|----------------------------|
   | power rail, oscilloscope | 3                                  | 1.000                      |
   | EM near-field probe      | 2                                  | 1.000                      |
   | acoustic, microphone     | never (>10^7 runs)                 | 0.859                      |
   | software power sensor    | never (>10^7 runs)                 | 0.859                      |
   | thermal sensor           | never (>10^7 runs)                 | 0.859                      |
   ```

5. **Which channel is best depends on the target, and the ranking follows the assumed numbers (§4).** One run, prompts of
   16 tokens, 28 expert pairs (8 experts, 2 active) and 4 topics. The oscilloscope and the EM probe name tokens
   (`0.854` and `0.949`) and so recover a response hidden behind a post-hoc filter (`0.841` and `0.952`). The
   microphone, the software sensor and the thermal sensor are at chance for tokens: `0.021`, `0.021` and `0.019` against
   `0.016`. A topic is different. Through expert routing the slow sensors still work, with `0.813` on the microphone and
   `0.677` on the software sensor against chance `0.250`; the thermal sensor takes 0.2 samples per token and cannot
   resolve a token slot (`-` in the table). A dense model's topic can be read only by naming tokens, so the three slow
   channels give `0.248`, `0.252` and `0.258`, which is chance. The EM probe beats the oscilloscope here only because the
   assumed values give it the better gain-to-noise ratio (0.4 over 4 LU, against 1 over 12), and accuracy follows the
   assumed gain (`test_a_channel_gain_scales_the_levels_and_therefore_the_accuracy`), so another assumption can reverse
   the order. What the table supports is the form of the answer, the target's time scale set against the sensor's window
   and noise, and not a measured ranking.

   ```
   | channel                  | token | expert pair | topic (routing) | topic (dense) | hidden response |
   |--------------------------|-------|-------------|-----------------|---------------|-----------------|
   | power rail, oscilloscope | 0.854 | 0.560       | 0.955           | 1.000         | 0.841           |
   | EM near-field probe      | 0.949 | 0.606       | 0.965           | 1.000         | 0.952           |
   | acoustic, microphone     | 0.021 | 0.369       | 0.813           | 0.248         | 0.019           |
   | software power sensor    | 0.021 | 0.231       | 0.677           | 0.252         | 0.014           |
   | thermal sensor           | 0.019 | -           | -               | 0.258         | 0.014           |
   ```

6. **Expert routing adds a topic leak that reaches slow sensors; a dense model has none (§5).** In the toy router each
   topic favours a few of the 28 expert pairs (Dirichlet concentration 0.3) and each expert shifts the slot's mean power
   by a fixed amount. The expert pair a token uses reveals `0.880 bits about the topic` (at most 2.000) and, read
   exactly, one token gives `0.658 topic accuracy`; a dense model makes no such choice
   (`test_a_dense_model_is_a_router_with_one_choice_and_leaks_nothing`). A Bayes attacker who reads one noisy level per
   token needs 16 tokens on the oscilloscope and the EM probe, 32 on the microphone and 64 on the software sensor to
   reach 90% topic accuracy. The dense attacker, who has to name tokens, needs 4 and 1 tokens on the two fast channels
   and more than 1024 (the search limit) on the others. In the dense case a topic is a block of 16 of the 64 tokens, an
   assumption that sets this comparison as much as the router's concentration does. So routing is an extra low-bandwidth
   leak: it reaches sensors that cannot name a token, and it does not make the fast-channel leak larger, since a dense
   model already gives up its tokens there.

   ```
   | channel                  | tokens for 90% topic accuracy: routing | tokens for 90%: dense, naming tokens |
   |--------------------------|----------------------------------------|--------------------------------------|
   | power rail, oscilloscope | 16                                     | 4                                    |
   | EM near-field probe      | 16                                     | 1                                    |
   | acoustic, microphone     | 32                                     | >1024                                |
   | software power sensor    | 64                                     | >1024                                |
   | thermal sensor           | -                                      | >1024                                |
   ```

7. **The size of the routing leak is set by sensor noise against the expert's power shift (§5).** Holding the software
   sensor at 50 Hz and varying its noise, with each expert shifting the mean level by 0.05 LU (one standard deviation),
   the tokens needed for 90% topic accuracy go from 16 when the noise is 0.05 of the shift to 256 at 0.4 of it and 1024
   at 0.8, and past the search limit at 1.6 (`test_a_noisier_sensor_is_worse`). The table turns "does expert routing leak
   through a slow sensor" into one ratio that a measurement would have to supply, and the package does not measure it.

   ```
   | sensor noise (LU) | noise / expert shift | tokens for 90% topic accuracy |
   |-------------------|----------------------|-------------------------------|
   | 0.0025            | 0.05                 | 16                            |
   | 0.005             | 0.1                  | 32                            |
   | 0.01              | 0.2                  | 64                            |
   | 0.02              | 0.4                  | 256                           |
   | 0.04              | 0.8                  | 1024                          |
   | 0.08              | 1.6                  | >1024                         |
   ```

8. **Co-tenants and the exposed sample rate move the routing attack in opposite directions (§6).** A shared software
   sensor adds the other workloads' noise in quadrature (0.005 LU each, an assumption) and exposes some sample rate. At
   50 Hz the tokens needed for 90% topic accuracy are 64, 128 and 256 with 1, 4 and 16 tenants; at 200 Hz they are 32,
   64 and 128; at 1,000 Hz they are 16, 16 and 32. A faster exposed sensor buys back what co-tenants take away
   (`test_other_tenants_slow_the_routing_attack_and_a_faster_exposed_rate_speeds_it_up`). It does not let a slow sensor
   name tokens: through the 1000 Hz sensor token accuracy is `0.017` against chance `0.016`, because each sample
   covers `1,000,000 operations, of which 288 carry the token`
   (`test_a_faster_exposed_rate_does_not_let_a_slow_sensor_name_tokens`).

   ```
   | exposed rate (Hz) | samples per token | 1 tenant | 4 tenants | 16 tenants |
   |-------------------|-------------------|----------|-----------|------------|
   | 50                | 1                 | 64       | 128       | 256        |
   | 200               | 4                 | 32       | 64        | 128        |
   | 1,000             | 20                | 16       | 16        | 32         |
   ```

9. **A refusal applied after decoding hides the text and not the tokens (§7).** The pipeline has three modes: no
   refusal; a post-hoc filter, which decodes the response and then replaces the text with a refusal; and a pre-check,
   which refuses before any decoding. The attacker names each decoded token from its decode step. With the post-hoc
   filter the fraction of a 32-token hidden response that is recovered equals the fraction with no refusal at all
   (`0.841` on the oscilloscope, `0.952` on the EM probe, `0.019` and `0.014` on the slow channels), and with the
   pre-check it is `0.000` on all four. This holds by construction: the model has the accelerator embed each token as it
   is decoded, so nothing applied afterwards can remove those steps
   (`test_hiding_the_text_after_decoding_does_not_hide_the_tokens_from_the_attacker`,
   `test_the_three_refusal_modes`). What the model cannot say is how much of a real serving stack's behaviour that
   decode step exposes.

   ```
   | refusal applied | power rail, oscilloscope | EM near-field probe | acoustic, microphone | software power sensor |
   |-----------------|--------------------------|---------------------|----------------------|-----------------------|
   | no refusal      | 0.841                    | 0.952               | 0.019                | 0.014                 |
   | post-hoc filter | 0.841                    | 0.952               | 0.019                | 0.014                 |
   | pre-check       | 0.000                    | 0.000               | 0.000                | 0.000                 |
   ```

## Open questions

1. *"Which side-channel (power, EM, acoustic, thermal) offers the best SNR for different attack targets?"*
   **Partially.** In the model the answer has a form. A target that needs the identity of one token is reachable only
   by a channel that samples faster than the token's arithmetic, here the oscilloscope and the EM probe (findings 2, 4
   and 5). A target that is an aggregate, such as which experts ran, is reachable through a slow sensor (findings 5 and
   6). The EM probe comes out ahead of the oscilloscope only because its assumed gain-to-noise ratio is higher. Whether
   any assumed gain, noise level or sample rate is true of a real GPU, and so which channel is best in practice, is not
   established.
2. *"How much does model architecture (dense vs MoE) affect leakage surface?"* **Partially.** The model gives a
   direction and a size. A mixture-of-experts router adds a topic leak that survives slow sensors and a dense model adds
   none, while on fast channels the dense model already gives up its tokens (findings 5 and 6). How large the extra leak
   is on a real model depends on the ratio of an expert's power shift to the sensor's noise, which finding 7 tabulates and
   which nothing here measures.
3. *"Can attacks work through virtualization layers in cloud environments?"* **Partially.** The model reduces a
   virtualization layer to two numbers, the noise that co-tenants add to a shared sensor and the sample rate that the
   hypervisor exposes. With them the routing attack still reaches 90% topic accuracy at 16 tenants, in 256 tokens at 50
   Hz and 32 at 1,000 Hz, and token naming fails through a 1,000 Hz sensor (finding 8). Whether a real GPU partitioning
   scheme, power cap or hypervisor lets a tenant read such a sensor is the actual question, and it is not answered.

## Not established

- Nothing was measured on a GPU. The leakage law (the Hamming weight of each multiplication), every channel's gain, noise
  and sample rate, the per-expert power shift and the co-tenant noise are assumptions chosen as round numbers. Which
  channel is best on real hardware, and whether an expert's power shift exceeds a real sensor's noise, is not
  established.
- The model is one 12 x 24 int8 layer with known weights and a 64-token vocabulary. Attention, KV caches, batching and
  prefill parallelism, tensor-core and mixed-precision datapaths, clock gating, voltage and frequency scaling and a
  trained router are not modelled, so the token and topic accuracies describe the toy and not a transformer.
- The slow-sensor figures depend on a modelling choice. The toy leaves the rest of each sensor window idle, so the
  token-dependent part is diluted by the window length (finding 2). A real decode step performs token-dependent
  arithmetic throughout the window, and what a 50 Hz or 10 Hz sensor makes of that sum is not modelled. The toy cannot
  say in which direction its slow-sensor rows are wrong.
- The attacker is a template attacker with perfect profiling: it knows the weights, the embeddings, the router table and
  each channel's gain and noise exactly. Error in any of them would lower every accuracy printed here. Output logits,
  which the RFD also names as a target, are not modelled.
- The cloud result is a sensor model, a shared sensor with extra noise and an exposed sample rate. GPU partitioning,
  power capping, hypervisor access control and the telemetry that a tenant can really read are not modelled, so whether
  attacks work through real virtualization layers is not established.
- Every figure is one seeded Monte Carlo run on one random layer and one random router (seed 0). Token and expert-pair
  accuracies use 2000 trials, the hidden-response figures 20 responses of 32 tokens, topic accuracies and token searches
  600 trials, flag rates 400 and recovery counts 100. A standard error is therefore about 0.01 for the 2000-trial
  accuracies, up to 0.02 for the 600-trial ones and up to 0.05 for the 100-trial recovery rates, and token searches
  resolve only powers of two. Other seeds move the details, for example which tokens share a sum, and the spread over
  layers and routers is not reported.
- The refusal result holds by construction (finding 9). It shows that decoding a response before replacing it leaves the
  decoding visible, not how large the leak is on a real serving stack.
- Defences (RFD 004), covert use of the channels by a trained model (RFD 003) and fingerprinting of a particular
  mixture-of-experts model (RFD 002) are outside the package. The only defence-like variables are the sensor's window,
  its noise, the co-tenants' noise and the order of a refusal and decoding.

## Attacks tested

Each row is an attack run against the model, the property of the sensor or pipeline that is under test as the defence,
and the figures the program prints for it.

| Attack | Defence under test | Outcome |
| --- | --- | --- |
| Fixed-versus-random t test on the oscilloscope trace | None: the first layer leaks at the full sample rate | Flags `2 / 94 / 217 of 288` samples after 100, 1,000 and 10,000 traces |
| The same test on the EM probe | None: lower gain, lower per-operation noise | Flags `1 / 114 / 238 of 288` samples |
| The same test on the 50 kHz microphone | A window of 20,000 operations holds the token's 288 | One sample per trace, `0 / 0 / 0 of 1` flagged; `5.7e5` traces per group would be needed |
| The same test on the 50 Hz software sensor | A window of 20,000,000 operations | `0 / 0 / 0 of 1` flagged; `1.8e9` traces per group would be needed |
| The same test with no leak planted | The test's false-alarm rate | `0 of 288 samples flagged` after 10,000 traces |
| Correlation attack on activation 77 at noise 2 LU | Measurement noise | Recovered after `40` samples; a random activation needs `308` |
| Correlation attack on an activation whose double leaks the same | Ghost hypotheses | `5 ~ 10 (0.998)`: the pair is nearly indistinguishable; activation 0 is never recovered |
| Naming a token from one oscilloscope run | Measurement noise | Accuracy `0.838`; `0.986` after averaging 2 runs |
| Naming a token through the 50 Hz software sensor | A slow sensor reports one number per token | `0.021` against chance `0.016`; ceiling `0.859` even with no noise |
| Topic of a prompt from expert routing, software sensor at 50 Hz | Slow sampling | `0.677` for a 16-token prompt; 64 tokens give 90% |
| The same through a sensor shared with 16 tenants | Co-tenant noise | 256 tokens give 90% |
| The same, sensor at 1,000 Hz, 16 tenants | A faster exposed rate against co-tenant noise | 32 tokens give 90% |
| The same with sensor noise 1.6 times an expert's power shift | Noise above the signal | 90% is not reached within 1024 tokens |
| Topic of a prompt by naming tokens (dense model), microphone | Window averaging | `0.248` against chance `0.250` |
| Reading a response behind a post-hoc refusal filter (oscilloscope) | A refusal applied after decoding | `0.841` of the tokens recovered, the same as with no refusal |
| Reading a response behind a pre-check refusal | A refusal applied before decoding | `0.000` recovered |

## Related RFDs

- RFD 002, MoE Expert Fingerprinting: the routing leak here is the physical mechanism that RFD 002 would exploit, but
  only at the level of a topic inferred from a router table. Recovering token sequences from expert signatures, how many
  experts must be distinguishable, and transfer across model versions are not modelled. No code is shared.
- RFD 003, Trained Covert Communication via Side-Channels: the same channels used by a transmitter that cooperates with
  the receiver, where this package has a passive victim. Not modelled; the refusal result of §7 is the passive
  counterpart of the refusal that RFD 003 has the model hide a covert answer behind.
- RFD 004, Functional Invariants for Side-Channel Defense: the defence side of these leaks. Finding 7 shows how the tokens
  an attacker needs move with noise, which is the quantity such a defence would raise, but no invariant is implemented.
- RFD 012, Analog Sensors for Compute Verification: the defender's use of the same power, EM, acoustic and thermal
  signals. Its open question on processing signals without leaking model information concerns the same quantity that §1
  and §3 measure, what a signal reveals about the model, and nothing here implements a verifier.
