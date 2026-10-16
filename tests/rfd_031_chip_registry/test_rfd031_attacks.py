"""The attack catalogue: every row names what the code actually did, and the README is built from these rows."""

from __future__ import annotations

from aisecurity_rfds.rfd_031_chip_registry import attacks

PROTOCOL = {
    "Enrolment signed by a key that is not an approved manufacturer": ("manufacturer allow-list", "rejected: unauthorised"),
    "Enrolment under an id that is not the hash of the chip's key": ("chip id binding", "rejected: bad-chip-id"),
    "A second enrolment of a registered chip": ("unique chip id", "rejected: duplicate"),
    "A transfer offered by someone who does not own the chip": ("owner's signature", "rejected: unauthorised"),
    "An old offer replayed after the record has moved on": ("expected version", "rejected: stale-version"),
    "A second offer while the first is pending": ("transfer state machine", "rejected: bad-state"),
    "Completion by a party other than the named buyer": ("buyer's signature", "rejected: unauthorised"),
    "Completion after the offer has lapsed": ("offer expiry", "rejected: expired"),
    "Completion that reuses the chip's attestation from an earlier offer": ("challenge bound to the offer", "rejected: bad-attestation"),
    "The genuine chip completes a sale after a copy of it attested": ("attestation counter", "rejected: counter-regression"),
    "The seller withdraws an offer before it lapses": ("offer window", "rejected: not-expired"),
    "A sale into an embargoed jurisdiction": ("policy list", "rejected: embargoed"),
    "Any change to a retired chip": ("retirement is final", "rejected: retired"),
    "Destruction recorded without an accredited witness": ("witness signature", "rejected: unwitnessed"),
    "A relocation while a transfer is pending": ("transfer state machine", "rejected: bad-state"),
    "A request that names the same field twice": ("canonical fields", "rejected: bad-field"),
    "A chip recorded as lost attests to an inspector": ("attestation monitoring", "alert: zombie"),
}

OPERATOR = {
    "Rewrite an old entry and re-sign the head": ("audit replay", "detected: entry 2 records a state its request does not produce"),
    "The same rewrite shown to a witness that cosigned the earlier head": ("consistency proof", "refused: inconsistent"),
    "Append a transfer the owner never signed": ("audit replay", "detected: entry 4 is not a valid request (unauthorised)"),
    "Operator lets a copy complete a sale: the whole log is replayed": (
        "audit replay",
        "detected: entry 4 completes a transfer with a counter that did not rise",
    ),
    "The same: the chip's history is replayed": (
        "history replay",
        "rejected: counter-regression",
    ),
    "Change a record in the map but not in the log: a single query": ("map proof", "accepted"),
    "The same change: the chip's history is replayed": ("history replay", "rejected: history-disagrees-with-map"),
    "The same change: the whole log is replayed": ("audit replay", "detected: map root differs from the signed head"),
    "Swap two unrelated entries and re-sign the head": ("audit replay", "accepted: audit passes"),
    "The same, to an auditor that kept the earlier head": ("audit replay with earlier heads", "detected: log root at size 3 differs from an earlier head"),
    "Date an entry before a head that was published without it": ("audit replay with earlier heads", "detected: entry 3 is dated before a head that was published without it"),
    "Date an entry earlier, but after the head published without it": ("audit replay with earlier heads", "accepted: audit passes"),
    "Serve a head from before a recent transfer": ("head age limit", "rejected: stale-head"),
    "Serve a head dated far in the future so it stays fresh": ("clock skew limit", "rejected: future-head"),
    "Show two clients different histories: 3 witnesses, 2 must cosign, 0 colluding": ("witness quorum", "no way found to get both accepted"),
    "Show two clients different histories: 3 witnesses, 2 must cosign, 1 colluding": ("witness quorum", "both heads accepted"),
    "Show two clients different histories: 5 witnesses, 4 must cosign, 1 colluding": ("witness quorum", "no way found to get both accepted"),
    "Show two clients different histories: 5 witnesses, 4 must cosign, 3 colluding": ("witness quorum", "both heads accepted"),
    "Drop the newest entries and sign a shorter head": ("witness", "refused: shrinking"),
    "The same, to an auditor holding the shorter log and the longer head": ("audit replay", "detected: log has 3 entries but the head covers 4"),
    "Never log a request that was submitted": ("none", "accepted: audit passes"),
    "Claim that a registered chip is not registered": ("non-membership proof", "rejected: bad-proof"),
    "Answer a query with a different chip's record": ("record and chip id check", "rejected: wrong-record"),
}


def as_dict(results):
    return {r.attack: (r.defence, r.outcome) for r in results}


def test_every_protocol_attack_has_the_recorded_defence_and_outcome():
    results = attacks.protocol_attacks()
    assert len(results) == len(PROTOCOL) == len({r.attack for r in results})
    assert as_dict(results) == PROTOCOL


def test_every_operator_attack_has_the_recorded_defence_and_outcome():
    results = attacks.operator_attacks()
    assert len(results) == len(OPERATOR) == len({r.attack for r in results})
    assert as_dict(results) == OPERATOR


def test_the_catalogue_is_deterministic():
    assert attacks.protocol_attacks() == attacks.protocol_attacks()
    assert attacks.operator_attacks() == attacks.operator_attacks()


def test_the_rules_refuse_every_protocol_attack_that_is_a_request():
    outcomes = [r.outcome for r in attacks.protocol_attacks()]
    assert sum(o.startswith("rejected: ") for o in outcomes) == 16
    assert "accepted" not in outcomes


def test_the_attacks_that_pass_unnoticed_are_the_four_the_readme_admits_to():
    undetected = [r.attack for r in attacks.operator_attacks() if r.outcome.startswith("accepted")]
    assert undetected == [
        "Change a record in the map but not in the log: a single query",
        "Swap two unrelated entries and re-sign the head",
        "Date an entry earlier, but after the head published without it",
        "Never log a request that was submitted",
    ]


def test_forked_heads_are_two_valid_heads_of_the_same_size_that_disagree():
    operator, head_a, head_b, now = attacks.forked_heads()
    assert head_a.verify(operator) and head_b.verify(operator)
    assert head_a.size == head_b.size == 4
    assert head_a.log_root != head_b.log_root
    assert now == head_a.timestamp == head_b.timestamp
