# AI Security Forum RFDs: reference implementations, simulations and analyses

Executable, tested responses to the "Requests for Discussion" at <https://projects.aisecurity.forum/>:
42 open problems in AI infrastructure security and verification.

## Read this first: what "answering an RFD" means here

The RFDs are idea-stage documents. They have no acceptance criteria, and many describe physical
hardware (tamper-proof enclosures, SCIFs, chips fabricated without dormant features) or institutions
(a hardware verification lab) that no software repository can build or verify. This repository therefore
does **not** claim that any RFD is solved. For each RFD it provides the most useful artifact that can
honestly be built and tested in software, and states exactly what that artifact does not show.

| Kind | Meaning |
| --- | --- |
| `reference-implementation` | Working code for a protocol, algorithm or data structure, built on real cryptography and exercised by adversarial tests. It is still software: no hardware claim follows. |
| `simulation` | An executable model of a physical or system mechanism on synthetic data. Results characterise the model, not hardware. |
| `analysis` | Quantitative design analysis (calculators, parametric models, decision procedures) where the RFD is mostly hardware or institutional. |

Every package contains a README (status, findings, each open question with an evidence-backed status,
"Not established", attacks tested), a machine-readable `rfd.toml`, a deterministic
`python -m aisecurity_rfds.rfd_NNN_<slug>` entry point that prints the numbers quoted in its README, and
at least 15 tests including adversarial ones, and a list of deliberately seeded bugs
(`docs/mutations/rfd_NNN.toml`) that `scripts/mutation_check.py` applies to a copy of the code to confirm
the tests actually notice breakage. `tests/test_repo_integrity.py` enforces all of this, plus a lint for
overclaiming language and a rule that packages import only the shared library.

## Status

<!-- STATUS:BEGIN -->
**3 of 42 RFD packages present** (2 reference-implementation, 1 simulation); 212 RFD-specific test functions (parametrised cases are not expanded, so pytest runs more).

| # | RFD | Author | Kind | Open questions addressed | Test functions | What the package provides |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | Side-Channel Leakage from LLM Inference | Gabriel Kulp | not started | - | - | |
| 002 | MoE Expert Fingerprinting | Gabriel Kulp | not started | - | - | |
| 003 | Trained Covert Communication via Side-Channels | Gabriel Kulp | not started | - | - | |
| 004 | Functional Invariants for Side-Channel Defense | Gabriel Kulp | not started | - | - | |
| 005 | Network Tap for Compute Verification | James Petrie | not started | - | - | |
| 006 | Secure Enclosure (SCIF) for AI Clusters | James Petrie | not started | - | - | |
| 007 | Mutually Trusted Cluster for Log Verification | James Petrie | not started | - | - | |
| 008 | Attested Logging with Existing Hardware | James Petrie | not started | - | - | |
| 009 | Zero-Maintenance Compute for Tamper Evidence | James Petrie | not started | - | - | |
| 010 | Trusted Kernel-Logging IP Block | James Petrie | not started | - | - | |
| 011 | Nonconfidential Verification (Red Team/Blue Team) | James Petrie | not started | - | - | |
| 012 | Analog Sensors for Compute Verification | James Petrie | not started | - | - | |
| 013 | Network Exhaustion Protocol | James Petrie | not started | - | - | |
| 014 | Memory Exhaustion Protocol | James Petrie | not started | - | - | |
| 015 | Hardware Verification Lab | James Petrie | not started | - | - | |
| 016 | Zero-Knowledge Proofs for LLM Inference | James Petrie | not started | - | - | |
| 017 | Inference-Only Verification Package | Romeo Dean | not started | - | - | |
| 018 | Granular AI Workload Verification | Romeo Dean | not started | - | - | |
| 019 | Guarantee Processor Design | FlexHEG | not started | - | - | |
| 020 | Tamper-Evident Secure Enclosure for Accelerators | FlexHEG | not started | - | - | |
| 021 | Interlock-Based Verification Architecture | FlexHEG | not started | - | - | |
| 022 | NIC Repurposing for Guarantee Processor | FlexHEG | not started | - | - | |
| 023 | Compute Graph Declaration Protocol | FlexHEG | not started | - | - | |
| 024 | Distributed FLOP Counting | FlexHEG | not started | - | - | |
| 025 | k-of-n Guarantee Update Mechanism | FlexHEG | not started | - | - | |
| 026 | Location Verification for AI Compute | Amodo / Tim Fist | not started | - | - | |
| 027 | Training vs Inference Workload Discrimination | Amodo | not started | - | - | |
| 028 | Retrofittable Tamper Detection System | Amodo | not started | - | - | |
| 029 | NIC-Based Bandwidth Limiting & Interconnect Verification | Amodo | not started | - | - | |
| 030 | Verification Research Testbed & Dataset | Amodo | not started | - | - | |
| 031 | Chip Registry for AI Governance | Oxford Martin AI Governance Initiative | not started | - | - | |
| 032 | Offline Licensing for Compute Rationing | RAND Corporation | reference-implementation | 5/7 | 42 | An on-chip license verifier and meter block on real Ed25519 (device binding, replay, overflow, caps, fail-safe checkpointing), plus assumption-labelled models of throttle actions, mid-job exhaustion policies and stockpiling. |
| 033 | Fixed Set Cluster Size Limiting | RAND Corporation | not started | - | - | |
| 034 | License Replay Prevention | RAND Corporation | reference-implementation | 4/5 | 56 | Executable hash-log, sequence-number, challenge and hybrid replay guards on simulated fuses and flash, run against rollback, high-voltage disruption, torn-write and stockpiling attacks. |
| 035 | Performance Counter Integrity Protection | RAND Corporation | simulation | 3/5 | 114 | Synthetic per-SM telemetry with the RFD's six counter attacks, eight verifier checks, an attacker who hides work behind them, and a signed hash-chained readout protocol with authorised resets, explored exhaustively at small bounds. |
| 036 | Tamper-Respondent PUF Enclosures | RAND Corporation | not started | - | - | |
| 037 | Secure Chip Binning and Feature Lockout | RAND Corporation | not started | - | - | |
| 038 | Containerized Data Centers for Mobile/Military AI | Oxford Martin AI Governance Initiative | not started | - | - | |
| 039 | Model Fingerprint Attestation | Oxford Martin AI Governance Initiative | not started | - | - | |
| 040 | Model Tenancy Ledger Attestation | Oxford Martin AI Governance Initiative | not started | - | - | |
| 041 | Device-Model Mating for AI-Enabled Weapons | Oxford Martin AI Governance Initiative | not started | - | - | |
| 042 | On-Device Logging for AI Weapons Verification | Oxford Martin AI Governance Initiative | not started | - | - | |
<!-- STATUS:END -->

RFDs marked "not started" have no package yet; their integrity tests are reported as skipped, not passed.
Run `AISEC_REQUIRE_ALL=1 pytest` to make every missing package a failure.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # whole suite
pytest tests/rfd_035_perf_counter_integrity
python -m aisecurity_rfds.rfd_035_perf_counter_integrity   # prints that package's headline findings
python scripts/mutation_check.py 35      # seeded-bug check: every listed bug must make a test fail
python scripts/build_status.py           # regenerate this status table and docs/STATUS.md
```

Python 3.11+; dependencies are numpy, scipy and `cryptography` only. All randomness is seeded and no
test touches the network.

## Layout

```
src/aisecurity_rfds/common/     shared building blocks (below)
src/aisecurity_rfds/rfd_NNN_*/  one package per RFD: code, README.md, rfd.toml, __main__.py
tests/                          tests/common, tests/rfd_NNN_*, tests/test_repo_integrity.py
docs/                           CONVENTIONS.md (the contract), rfd_index.json, STATUS.md (generated)
docs/mutations/rfd_NNN.toml     seeded bugs each package's tests must catch
scripts/build_status.py         regenerates the status table from the packages themselves
scripts/mutation_check.py       applies each seeded bug to a temporary copy of src/ and runs the tests
```

## Shared library (`aisecurity_rfds.common`)

Small models for research and testing, not hardened production primitives: domain-separated hashing,
Ed25519 signing, an RFC 6962 Merkle log with inclusion and consistency proofs (its tree heads match
the Certificate Transparency reference vectors, asserted in the tests), a hash-chained log with
external-anchor verification, X25519 + HKDF + AES-GCM sessions with counter nonces and replay
rejection, a simulated PUF with a code-offset fuzzy extractor, simulated non-volatile memories
(OTP fuses, flash with torn writes and rollback, battery-backed SRAM), and detection-probability
statistics. Known limits are documented in the module docstrings; for example, a Merkle root does not
commit to tree size, so `(size, root)` must be authenticated together.

## Provenance

This repository was produced with AI assistance (Claude) and is meant to be checked, not trusted:
run the tests and the per-package entry points instead of relying on the prose. Edit this section to
state authorship however you prefer before publishing.

## Not included

No `LICENSE` file is included; add one before publishing. Nothing here is endorsed by, or affiliated
with, the AI Security Forum or the RFD authors; the RFD texts remain theirs.
