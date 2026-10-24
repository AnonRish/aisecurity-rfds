"""RFD 001, Side-Channel Leakage from LLM Inference: a simulation of what power, EM, acoustic, thermal and
software-sensor measurements could reveal about a model's inputs, and which of those leaks survive a slow or shared
sensor.

It is a model, not a measurement. The leakage law (Hamming weight of each multiplication) and every channel
parameter are assumptions; nothing here was measured on a GPU. What the package contributes is the analysis
code (leakage detection, correlation attack, identification bounds, routing leakage) validated against closed-form
results on synthetic traces, and a calculator for how the answer moves when the assumptions are replaced.
"""

from .analysis import (
    TVLA_THRESHOLD,
    cpa_recover,
    identification_error_bounds,
    identify,
    repeats_needed,
    tvla_traces_needed,
    welch_t,
)
from .leakage import ACOUSTIC, CHANNELS, EM_PROBE, SCOPE, SOFTWARE, THERMAL, Channel, ToyLayer, hw16
from .routing import MoeRouter, category_accuracy, mutual_information

__all__ = [
    "ACOUSTIC",
    "CHANNELS",
    "EM_PROBE",
    "SCOPE",
    "SOFTWARE",
    "THERMAL",
    "TVLA_THRESHOLD",
    "Channel",
    "MoeRouter",
    "ToyLayer",
    "category_accuracy",
    "cpa_recover",
    "hw16",
    "identification_error_bounds",
    "identify",
    "mutual_information",
    "repeats_needed",
    "tvla_traces_needed",
    "welch_t",
]
