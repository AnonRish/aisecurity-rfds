"""RFD 037, Secure Chip Binning and Feature Lockout: executable models of what binning exposes and what lockouts cost.

Four layers, each a model under stated assumptions and not a hardware design:

* ``binning``: a clustered-defect yield model with planners for one binned die, for a die per tier, and for a hybrid
  that sells defective top dies lower; it measures the share of lower-tier chips whose silicon could run at the top
  tier (the RFD's concern) and the silicon cost of avoiding that (open question 1);
* ``mechanisms``, ``lockout``: the RFD's four lockout mechanisms, plus firmware flags, fuses with redundancy and signed
  tokens in three strengths, as small chip models run against attacker budgets, so that what defeats what is computed
  rather than asserted, and the field-failure cost of a PUF-bound key is simulated;
* ``rma``: an exhaustive search of tier changes after sale, covering upgrade, downgrade, die replacement and token
  replay (open questions 3 and 5);
* ``audit``: how many chips an authority must inspect and what a cheating manufacturer can hide (open question 2).

``python -m aisecurity_rfds.rfd_037_secure_chip_binning`` prints every number the README quotes. The README says what is
not established: no real die, fuse, fab, price or inspection technique was measured, and open question 4 is not
addressed.
"""

from .audit import (
    Lot,
    design_files_match,
    destroyed_fraction,
    estimate_detection,
    evasion_budget,
    inspections_needed,
    miss_probability,
    repeated_miss_probability,
    sample_audit,
)
from .binning import (
    ILLUSTRATIVE_DIE,
    ILLUSTRATIVE_TIERS,
    Die,
    InfeasibleError,
    Process,
    Tier,
    best_safe_spares,
    breakeven_defect_density,
    breakeven_volume,
    dies_per_wafer,
    good_block_pmf,
    max_safe_spares,
    plan_binning,
    plan_hybrid,
    plan_separate,
    secure_cost,
    silicon_premium,
    total_cost_per_unit,
)
from .lockout import MECHANISMS, PROFILES, budgets, glitch_bypass_probability, matrix, minimum_budget, puf_field_failures
from .mechanisms import Attacker, Outcome, evaluate
from .rma import DESIGNS, Design, Exploration, explore

__all__ = [
    "DESIGNS",
    "ILLUSTRATIVE_DIE",
    "ILLUSTRATIVE_TIERS",
    "MECHANISMS",
    "PROFILES",
    "Attacker",
    "Design",
    "Die",
    "Exploration",
    "InfeasibleError",
    "Lot",
    "Outcome",
    "Process",
    "Tier",
    "best_safe_spares",
    "breakeven_defect_density",
    "breakeven_volume",
    "budgets",
    "design_files_match",
    "destroyed_fraction",
    "dies_per_wafer",
    "estimate_detection",
    "evaluate",
    "evasion_budget",
    "explore",
    "glitch_bypass_probability",
    "good_block_pmf",
    "inspections_needed",
    "matrix",
    "max_safe_spares",
    "minimum_budget",
    "miss_probability",
    "plan_binning",
    "plan_hybrid",
    "plan_separate",
    "puf_field_failures",
    "repeated_miss_probability",
    "sample_audit",
    "secure_cost",
    "silicon_premium",
    "total_cost_per_unit",
]
