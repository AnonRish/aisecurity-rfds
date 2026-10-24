"""RFD 002, MoE Expert Fingerprinting: a simulation of what a physical sensor could tell an attacker about which
experts a mixture-of-experts model ran, and how much of the input token that reveals.

It is a model, not a measurement: no GPU, no real mixture-of-experts model and no real sensor was used. Three layers:

* ``moe``: a small mixture-of-experts language model (a learned top-k router in every block, hand-written gradients, an
  Adam trainer) fitted here to a synthetic first-order language, so that routing, the loss in quality and the effect of
  quantization, fine-tuning and routing noise are measured on a model that exists and not read off a formula;
* ``signatures``: the assumed sensor, in which every expert has a signature vector and a block's reading is the sum of
  its active experts' signatures plus Gaussian noise, with closed-form bounds on decoding which experts ran;
* ``attack``: a naive-Bayes decoder that uses a table of how often each token made each set of experts run, built with
  white-box access to a model, and the measurements behind the RFD's three open questions.

``python -m aisecurity_rfds.rfd_002_moe_expert_fingerprinting`` prints every number the README quotes. The README says
what is not established: the toy model, its language and the sensor's signatures are assumptions, each figure is one
trained model, and the decoder is not claimed to be the best attacker.
"""

from .attack import (
    Profile,
    build_profile,
    deterministic_ceiling,
    route_change,
    set_likelihood,
    token_scores,
)
from .moe import (
    Chain,
    Config,
    evaluate,
    expert_load,
    forward,
    init_params,
    quantize,
    routes_of,
    routing_noise,
    train,
)
from .signatures import (
    SetSpace,
    Signatures,
    bits_per_slot,
    distinct_signatures,
    noise_free_ceiling,
    pair_errors,
    set_accuracy,
    set_error_bounds,
)

__all__ = [
    "Chain",
    "Config",
    "Profile",
    "SetSpace",
    "Signatures",
    "bits_per_slot",
    "build_profile",
    "deterministic_ceiling",
    "distinct_signatures",
    "evaluate",
    "expert_load",
    "forward",
    "init_params",
    "noise_free_ceiling",
    "pair_errors",
    "quantize",
    "route_change",
    "routes_of",
    "routing_noise",
    "set_accuracy",
    "set_error_bounds",
    "set_likelihood",
    "token_scores",
    "train",
]
