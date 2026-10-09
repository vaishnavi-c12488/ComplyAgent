"""
ComplyAgent - AML Rules Engine

Hard-rule layer for deterministic AML risk indicators.

Rules are intentionally transparent and auditable.
They do not represent legal advice or a production regulatory
threshold.
"""

from typing import Any


def evaluate_rules(
    transaction_pattern: dict,
    sanctions_match: dict,
) -> dict:
    """
    Evaluate deterministic risk rules.

    Returns structured evidence only.
    """

    triggered_rules: list[str] = []
    rule_scores: list[float] = []

    # --------------------------------------------------------
    # Sanctions rule
    # --------------------------------------------------------

    if sanctions_match.get("matched", False):
        triggered_rules.append(
            "SANCTIONS_MATCH"
        )
        rule_scores.append(1.0)

    # --------------------------------------------------------
    # Structuring
    # --------------------------------------------------------

    if transaction_pattern.get(
        "structuring_count",
        0,
    ) >= 2:
        triggered_rules.append(
            "STRUCTURING_PATTERN"
        )
        rule_scores.append(0.80)

    # --------------------------------------------------------
    # Rapid movement
    # --------------------------------------------------------

    if transaction_pattern.get(
        "rapid_movement_count",
        0,
    ) >= 3:
        triggered_rules.append(
            "RAPID_MOVEMENT_PATTERN"
        )
        rule_scores.append(0.70)

    # --------------------------------------------------------
    # High-risk jurisdiction exposure
    # --------------------------------------------------------

    if transaction_pattern.get(
        "high_risk_jurisdiction_ratio",
        0.0,
    ) >= 0.55:
        triggered_rules.append(
            "HIGH_RISK_JURISDICTION_EXPOSURE"
        )
        rule_scores.append(0.75)

    # --------------------------------------------------------
    # Cross-border activity
    # --------------------------------------------------------

    if transaction_pattern.get(
        "international_ratio",
        0.0,
    ) >= 0.65:
        triggered_rules.append(
            "HIGH_CROSS_BORDER_ACTIVITY"
        )
        rule_scores.append(0.55)

    # --------------------------------------------------------
    # Multiple beneficiaries
    # --------------------------------------------------------

    if transaction_pattern.get(
        "new_beneficiary_count",
        0,
    ) >= 5:
        triggered_rules.append(
            "MULTIPLE_NEW_BENEFICIARIES"
        )
        rule_scores.append(0.50)

    # --------------------------------------------------------
    # Aggregate rule score
    # --------------------------------------------------------

    if rule_scores:
        rule_score = min(
            1.0,
            max(rule_scores),
        )
    else:
        rule_score = 0.0

    return {
        "rules_triggered": triggered_rules,
        "rule_score": round(
            rule_score,
            4,
        ),
        "rule_count": len(
            triggered_rules
        ),
        "requires_rule_review": bool(
            triggered_rules
        ),
    }


def main() -> None:
    sample_transaction = {
        "structuring_count": 3,
        "rapid_movement_count": 4,
        "high_risk_jurisdiction_ratio": 0.2,
        "international_ratio": 0.3,
        "new_beneficiary_count": 2,
    }

    sample_sanctions = {
        "matched": False,
    }

    result = evaluate_rules(
        sample_transaction,
        sample_sanctions,
    )

    print("=" * 70)
    print("COMPLYAGENT - RULES ENGINE")
    print("=" * 70)
    print(
        f"Triggered rules: "
        f"{result['rules_triggered']}"
    )
    print(
        f"Rule score: "
        f"{result['rule_score']:.4f}"
    )
    print("\nRules engine: PASSED")


if __name__ == "__main__":
    main()