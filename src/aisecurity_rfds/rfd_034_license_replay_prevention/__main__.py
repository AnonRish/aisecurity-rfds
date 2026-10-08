"""Print the headline findings quoted in README.md. Deterministic; a few seconds of CPU."""

from __future__ import annotations

from collections.abc import Sequence

from . import scenarios as sc


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in cells]
    return "\n".join([line, rule, *body])


def main() -> None:
    print("RFD 034 - License replay prevention. Simulated hardware: every number is a property of the model.\n")

    labels = [c.label for c in sc.CONFIGS]
    short = {label: f"C{i + 1}" for i, label in enumerate(labels)}
    print("1. Attack matrix (blocked = attacker gains no budget the authority did not grant)")
    print("   Columns: " + "; ".join(f"{short[label]} = {label}" for label in labels))
    rows = [[name, *(row[label].cell() for label in labels)] for name, row in sc.attack_matrix()]
    print(_table(["attack", *(short[label] for label in labels)], rows))

    print("\n2. Liveness: licenses usable out of 3 in flight when delivered in the order 3, 1, 2")
    live = [[c.label, "n/a (one challenge outstanding)" if (v := sc.liveness_out_of_order(c)) is None else f"{v}/3"]
            for c in sc.CONFIGS]
    print(_table(["guard", "usable"], live))

    print("\n3. Exhaustive power loss while applying a license (every flash/fuse operation x before/torn/after)")
    reports = [sc.crash_consistency(c) for c in (*sc.CONFIGS, sc.NAIVE_SEQUENCE)]
    print(_table(
        ["guard", "crash points", "double-credit", "license lost", "retry works"],
        [[r.label, r.points, r.violations, r.lost, r.recoverable] for r in reports],
    ))

    print("\n4. Storage after n licenses: logical payload bytes / physical flash bytes (journaled) / fuses burned")
    storage_cfgs = [sc.CONFIG_BY_LABEL[k] for k in ("hash-log", "hash-log/ring4", "sequence", "hybrid(W=4)", "challenge+otp-epoch")]
    rows = []
    for cfg in storage_cfgs:
        rows.append([cfg.label, *("{}/{}/{}".format(*sc.storage_profile(cfg, n)) for n in (1, 10, 100))])
    print(_table(["guard", "n=1", "n=10", "n=100"], rows))

    print("\n5. OTP rollback floor: fuses used vs licenses an attacker can replay after a flash rollback (n=32)")
    rows = []
    for stride in (1, 2, 4, 8, 16, 32):
        worst, fuses = sc.worst_rollback_replay(sc.sequence_config(stride))
        rows.append([f"sequence+otp(S={stride})", fuses, worst])
    for stride in (1, 2, 4, 8):
        worst, fuses = sc.worst_rollback_replay(sc.hybrid_config(4, stride))
        rows.append([f"hybrid(W=4)+otp(S={stride})", fuses, worst])
    print(_table(["guard", "fuses burned after 32", "worst-case replayable"], rows))

    print("\n6. Mean % of legitimate licenses stranded when delivery is shuffled by up to D places (n=60, 10 trials)")
    windows: list[int | None] = [None, 2, 4, 8]
    rows = []
    for window in windows:
        name = "sequence" if window is None else f"hybrid(W={window})"
        rows.append([name, *(f"{100 * sc.mean_stranded_fraction(window, d):.1f}" for d in (0, 2, 4, 8))])
    print(_table(["guard", "D=0", "D=2", "D=4", "D=8"], rows))

    print("\n7. Challenge guard with a random-number generator that repeats after every reset:")
    print(f"   replay accepted without OTP epoch binding: {sc.weak_rng_replay(False)}")
    print(f"   replay accepted with OTP epoch binding:    {sc.weak_rng_replay(True)}")

    print("\n8. Hash log keyed on the full license (signature included) vs the signed content; padded-signature replay:")
    rows = []
    for scope in ("content", "full"):
        rows.append([
            scope,
            sc.padded_signature_replay(scope, sc.ed25519_verify),
            sc.padded_signature_replay(scope, sc.lenient_verifier),
        ])
    print(_table(["hash scope", "replay works, strict Ed25519", "replay works, lenient verifier"], rows))

    print("\n9. External commitments (every 5 licenses) vs a rollback attacker who replays 10 licenses:")
    for careful in (False, True):
        anomalies, credited = sc.ledger_detection(careful)
        who = "careful (withholds replay commitments)" if careful else "careless (chip reports everything)"
        print(f"   {who}: ledger anomalies = {anomalies or 'none'}; licenses credited = {credited} (authority granted 15)")


if __name__ == "__main__":
    main()
