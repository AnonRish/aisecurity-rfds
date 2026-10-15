"""RFD 031, Chip Registry for AI Governance: a reference implementation of the registry's protocol and its checks.

The RFD asks for a registry of unique chip identifiers with ownership and possibly location, an update protocol, a query
interface and an audit trail. Five modules, each a model under stated assumptions and not a deployment:

* ``identity``: the three identifiers the RFD names. A device key attests to a challenge and a counter, and the counter
  is a probabilistic tripwire for copied keys; a PUF-derived key can fail to reproduce; a glitter pattern identifies a
  package and has analytic and simulated error rates (open question 1, as far as a registry can say anything about it);
* ``smt``, ``registry``: a verifiable database (a sparse Merkle map with proofs of presence and absence), signed
  requests applied by one deterministic function, an append-only Merkle log, and the checks a client or auditor runs
  (technical requirements 2 to 5, open questions 5 and 6: sale, transfer, retirement, destruction);
* ``witness``: cosigning witnesses and quorums, and how many must collude for an operator to show two registries (open
  question 4, the technical part of governance);
* ``location``: location commitments that hide a facility until opened, and the sampling arithmetic of audits for
  declared chips that are not where they are declared (open questions 2 and 3);
* ``attacks``, ``scenario``: every attack in the README, executed against the real code.

``python -m aisecurity_rfds.rfd_031_chip_registry`` prints every number the README quotes. The README says what is not
established: no real chip key, PUF or tag was measured, open question 1 (robustness of on-chip keys against a state
actor) is not addressed, and the registry's clock, its operator's willingness to include requests, and the physical
location of a chip are outside what the cryptography can establish.
"""

from .identity import (
    Attestation,
    CounterTracker,
    DeviceKeyChip,
    PufChip,
    chip_id_of,
    clone_detection_probability,
    glitter_error_rates,
    party_key,
    puf_false_accept_probability,
    verify_attestation,
)
from .location import (
    Facility,
    Opening,
    chips_to_inspect,
    commit_location,
    detection_probability,
    verify_opening,
)
from .registry import (
    Policy,
    Record,
    Registry,
    Rejected,
    apply_request,
    audit_log,
    make_request,
    sign_request,
    verify_history,
    verify_query,
)
from .smt import SparseMerkleMap, verify_proof
from .witness import (
    CompromisedWitness,
    Quorum,
    Witness,
    WitnessRefused,
    liveness_margin,
    min_colluders_to_fork,
)

__all__ = [
    "Attestation",
    "CompromisedWitness",
    "CounterTracker",
    "DeviceKeyChip",
    "Facility",
    "Opening",
    "Policy",
    "PufChip",
    "Quorum",
    "Record",
    "Registry",
    "Rejected",
    "SparseMerkleMap",
    "Witness",
    "WitnessRefused",
    "apply_request",
    "audit_log",
    "chip_id_of",
    "chips_to_inspect",
    "clone_detection_probability",
    "commit_location",
    "detection_probability",
    "glitter_error_rates",
    "liveness_margin",
    "make_request",
    "min_colluders_to_fork",
    "party_key",
    "puf_false_accept_probability",
    "sign_request",
    "verify_attestation",
    "verify_history",
    "verify_opening",
    "verify_proof",
    "verify_query",
]
