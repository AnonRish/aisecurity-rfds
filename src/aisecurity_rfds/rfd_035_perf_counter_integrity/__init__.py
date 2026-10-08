"""RFD 035, Performance Counter Integrity Protection: a simulation of counter-tampering attacks against a
synthetic GPU telemetry model, the verifier-side sanity checks that catch them, how much an adversary who knows
the checks can still hide, and a signed, hash-chained readout protocol with authorised resets (explored
exhaustively at small bounds).

Everything runs on synthetic telemetry with assumed constants (see ``model.py``). Results characterise that model
and the protocol code; they are not measurements of any hardware.
"""

from .attacks import ATTACKS, AttackStats, detection_matrix, evaluate_attack
from .checks import CHECK_NAMES, Thresholds, calibrate, inspect
from .hiding import closed_form_max_hidden_fraction, hiding_table, margin_tradeoff, max_hidden_fraction
from .model import CLASSES, NOVEL_CLASS, Reading, WorkloadClass, generate_reading
from .monotonic import explore
from .readout import ReadoutBlock, ReadoutVerifier, ResetAuthority, ResetTicket

__all__ = [
    "ATTACKS",
    "CHECK_NAMES",
    "CLASSES",
    "NOVEL_CLASS",
    "AttackStats",
    "ReadoutBlock",
    "ReadoutVerifier",
    "Reading",
    "ResetAuthority",
    "ResetTicket",
    "Thresholds",
    "WorkloadClass",
    "calibrate",
    "closed_form_max_hidden_fraction",
    "detection_matrix",
    "evaluate_attack",
    "explore",
    "generate_reading",
    "hiding_table",
    "inspect",
    "margin_tradeoff",
    "max_hidden_fraction",
]
