"""RFD 032, Offline Licensing for Compute Rationing: an on-chip license verifier and meter block built on real
Ed25519 signatures, with a throttle-action model and small simulations of mid-job exhaustion, license size
and stockpiling.

The protocol code is a software reference model. The throttle and policy simulations use assumed
performance and job-size parameters (labelled in the code) and characterise those models, not any hardware.
"""

from .chip import Decision, OfflineLicenseChip
from .licensing import ComputeLicense, LicenseAuthority
from .meters import Meter
from .policy_sim import SimParams, SimResult, drawer_stockpile, revocation_outcome, simulate
from .throttle import HardwareProfile, Job, JobResult, ThrottleAction, run_job, throttle_rate

__all__ = [
    "ComputeLicense",
    "Decision",
    "HardwareProfile",
    "Job",
    "JobResult",
    "LicenseAuthority",
    "Meter",
    "OfflineLicenseChip",
    "SimParams",
    "SimResult",
    "ThrottleAction",
    "drawer_stockpile",
    "revocation_outcome",
    "run_job",
    "simulate",
    "throttle_rate",
]
