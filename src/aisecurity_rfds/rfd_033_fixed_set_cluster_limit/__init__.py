"""RFD 033, Fixed Set Cluster Size Limiting: an executable model of pod membership, link gating and what a cap costs.

Three layers, each a model under stated assumptions and not a hardware design:

* ``pod``, ``cluster``, ``gate``, ``scenarios``: manufacturer-signed pod manifests, a one-way fuse generation
  counter, a signed-key-exchange handshake, and fleet experiments (cluster size, key extraction, where the
  high-bandwidth decision lives, the update window after a member is removed);
* ``limiter``: an exact integer token-bucket egress limiter with a pod class and one shared external class;
* ``training``, ``podsize``, ``inference``, ``primitives``: a closed-form model of what an external bandwidth cap
  costs an attacker who trains, or an operator who serves, a large model across pods, with its constants checked by
  executing small versions of the computations.

``python -m aisecurity_rfds.rfd_033_fixed_set_cluster_limit`` prints every number the README quotes. The README says
what is not established: manufacturing cost and retrofit (open questions 5 and 6), the physical transceiver row,
and the behaviour of real hardware.
"""

from .cluster import Fleet, bridge_cluster_size, build_fleet, components, establish, largest_hbw_cluster
from .gate import Capability, Design, attempt, bypass_matrix
from .limiter import ChipPort, TokenBucket
from .pod import (
    Chip,
    HandshakeError,
    LinkDownError,
    LinkMode,
    Manufacturer,
    PodManifest,
    ProvisionError,
    Release,
)
from .training import External, Hardware, Layout, Model, Plan, Run, cost_multiplier, defeat_bandwidth_bps, plan

__all__ = [
    "Capability",
    "Chip",
    "ChipPort",
    "Design",
    "External",
    "Fleet",
    "HandshakeError",
    "Hardware",
    "Layout",
    "LinkDownError",
    "LinkMode",
    "Manufacturer",
    "Model",
    "Plan",
    "PodManifest",
    "ProvisionError",
    "Release",
    "Run",
    "TokenBucket",
    "attempt",
    "bridge_cluster_size",
    "build_fleet",
    "bypass_matrix",
    "components",
    "cost_multiplier",
    "defeat_bandwidth_bps",
    "establish",
    "largest_hbw_cluster",
    "plan",
]
