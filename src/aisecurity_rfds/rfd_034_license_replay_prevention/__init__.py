"""RFD 034, License Replay Prevention: executable models of the hash-log, sequence-number, challenge and hybrid
approaches on simulated non-volatile memory, with the attacks the RFD names (rollback, high-voltage
disruption, torn writes) implemented against them.

This is a software model. It characterises which design survives which attack under stated assumptions; it
does not show that any real chip's fuses, flash or random-number generator behave like the model.
"""

from .guards import (
    ChallengeGuard,
    Decision,
    HashLogGuard,
    HybridGuard,
    ReplayGuard,
    SequenceGuard,
    StrideFloor,
)
from .hardware import ChipNvm, FaultPlan, PowerLoss
from .ledger import Commitment, CommitmentEmitter, ExternalLedger
from .licensing import License, LicenseAuthority

__all__ = [
    "ChallengeGuard",
    "ChipNvm",
    "Commitment",
    "CommitmentEmitter",
    "Decision",
    "ExternalLedger",
    "FaultPlan",
    "HashLogGuard",
    "HybridGuard",
    "License",
    "LicenseAuthority",
    "PowerLoss",
    "ReplayGuard",
    "SequenceGuard",
    "StrideFloor",
]
