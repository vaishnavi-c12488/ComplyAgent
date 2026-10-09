
"""
ComplyAgent - Risk Agent

Combines:
1. Screening Agent evidence
2. Deterministic AML rules
3. Gradient-boosted ML risk score
4. Transaction-pattern tool evidence

Output is a strict Pydantic RiskEvidence contract.
"""

from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from agents.screening_agent import (
    ScreeningEvidence,
)
from mcp_server.tools.transaction_pattern_check import (
    check_transaction_pattern,
)
from risk_model.rules_engine import (
    evaluate_rules,
)
from risk_model.ml_risk_scorer import (
    predict_risk,
)


# ============================================================
# PYDANTIC CONTRACTS
# ============================================================

class TransactionPatternEvidence(BaseModel):
    case_id: str
    customer_id: str
    high_risk: bool
    scenario: str
    indicators: list[str]

    transaction_count_24h: int
    total_amount_24h: float
    avg_transaction_amount: float
    max_transaction_amount: float
    cash_ratio: float
    international_ratio: float
    high_risk_jurisdiction_ratio: float
    rapid_movement_count: int
    structuring_count: int
    new_beneficiary_count: int
    round_amount_ratio: float


class RuleEvidence(BaseModel):
    rules_triggered: list[str]
    rule_score: float = Field(ge=0.0, le=1.0)
    rule_count: int = Field(ge=0)
    requires_rule_review: bool


class MLRiskEvidence(BaseModel):
    prediction: int
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=100.0)
    model: str


class RiskEvidence(BaseModel):
    customer_id: str
    document_id: str

    screening: ScreeningEvidence
    transaction_pattern: TransactionPatternEvidence
    rules: RuleEvidence
    ml_risk: MLRiskEvidence

    overall_risk_score: float = Field(ge=0.0, le=100.0)

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    requires_review: bool
    evidence: list[str] = Field(default_factory=list)


# ============================================================
# RISK LEVEL
# ============================================================

def determine_risk_level(
    overall_score: float,
    sanctions_matched: bool,
) -> Literal["low", "medium", "high", "critical"]:
    """Determine the risk level using the configured thresholds."""

    if sanctions_matched:
        return "critical"

    if overall_score >= 75:
        return "high"

    if overall_score >= 45:
        return "medium"

    return "low"


# ============================================================
# RISK AGENT
# ============================================================

def run_risk_agent(
    screening: ScreeningEvidence,
) -> RiskEvidence:
    """Run the Risk Agent using structured upstream evidence."""

    # 1. Transaction-pattern tool
    transaction_result = check_transaction_pattern(
        screening.customer_id
    )

    transaction_pattern = (
        TransactionPatternEvidence.model_validate(
            transaction_result
        )
    )

    # 2. Deterministic rules engine
    sanctions_dict = {
        "matched": screening.sanctions_match.matched
    }

    rules_result = evaluate_rules(
        transaction_pattern.model_dump(),
        sanctions_dict,
    )

    rules = RuleEvidence.model_validate(rules_result)

    # 3. ML risk scorer.
    # The validated transaction evidence now includes all 11
    # model features, so available features are not replaced by
    # the scorer's missing-feature defaults.
    ml_result = predict_risk(
        transaction_pattern.model_dump()
    )

    ml_risk = MLRiskEvidence.model_validate(ml_result)

    # 4. Combine evidence
    ml_score = ml_risk.risk_score
    rule_score = rules.rule_score * 100.0

    transaction_score = (
        100.0 if transaction_pattern.high_risk else 0.0
    )

    screening_score = (
        100.0 if screening.sanctions_match.matched else 0.0
    )

    overall_score = (
        0.50 * ml_score
        + 0.25 * rule_score
        + 0.15 * transaction_score
        + 0.10 * screening_score
    )

    overall_score = min(100.0, max(0.0, overall_score))

    risk_level = determine_risk_level(
        overall_score,
        screening.sanctions_match.matched,
    )

    requires_review = (
        screening.requires_review
        or rules.requires_rule_review
        or risk_level in {"high", "critical"}
    )

    # 5. Evidence trace
    evidence = [
        f"ML risk score: {ml_risk.risk_score:.2f}/100.",
        f"Deterministic rule score: {rule_score:.2f}/100.",
        f"Transaction pattern high risk: {transaction_pattern.high_risk}.",
        f"Sanctions match: {screening.sanctions_match.matched}.",
        f"Overall risk score: {overall_score:.2f}/100.",
        f"Risk level: {risk_level}.",
    ]

    if rules.rules_triggered:
        evidence.append(
            "Rules triggered: "
            + ", ".join(rules.rules_triggered)
        )

    if transaction_pattern.indicators:
        evidence.append(
            "Transaction indicators: "
            + ", ".join(transaction_pattern.indicators)
        )

    return RiskEvidence(
        customer_id=screening.customer_id,
        document_id=screening.document_id,
        screening=screening,
        transaction_pattern=transaction_pattern,
        rules=rules,
        ml_risk=ml_risk,
        overall_risk_score=round(overall_score, 2),
        risk_level=risk_level,
        requires_review=requires_review,
        evidence=evidence,
    )


# ============================================================
# DEMO / VALIDATION
# ============================================================

def main() -> None:
    print("=" * 70)
    print("COMPLYAGENT - RISK AGENT")
    print("=" * 70)

    from agents.screening_agent import (
        load_identity_data,
        IdentityInput,
        run_screening_agent,
    )

    df = load_identity_data()
    row = df.iloc[0]

    identity = IdentityInput(
        customer_id=str(row["customer_id"]),
        full_name=str(row["full_name"]),
        date_of_birth=str(row["date_of_birth"]),
        document_number=str(row["document_number"]),
        nationality=str(row["nationality"]),
        country=str(row["country"]),
    )

    print(f"\nTesting customer: {identity.customer_id}")

    screening = run_screening_agent(identity)

    print(f"Screening status: {screening.screening_status}")
    print("\nRunning transaction-pattern analysis...")

    result = run_risk_agent(screening)

    # Contract and feature completeness checks
    assert isinstance(result, RiskEvidence)
    assert result.customer_id == screening.customer_id
    assert result.document_id == screening.document_id
    assert 0 <= result.overall_risk_score <= 100
    assert result.risk_level in {"low", "medium", "high", "critical"}
    assert isinstance(result.rules, RuleEvidence)
    assert isinstance(result.ml_risk, MLRiskEvidence)
    assert isinstance(result.transaction_pattern, TransactionPatternEvidence)

    feature_values = result.transaction_pattern.model_dump()

    required_ml_features = {
        "transaction_count_24h",
        "total_amount_24h",
        "avg_transaction_amount",
        "max_transaction_amount",
        "cash_ratio",
        "international_ratio",
        "high_risk_jurisdiction_ratio",
        "rapid_movement_count",
        "structuring_count",
        "new_beneficiary_count",
        "round_amount_ratio",
    }

    assert required_ml_features.issubset(feature_values.keys())

    print("\n" + "-" * 70)
    print("RISK RESULT")
    print("-" * 70)
    print(f"Customer ID       : {result.customer_id}")
    print(f"Transaction case  : {result.transaction_pattern.case_id}")
    print(f"ML risk score     : {result.ml_risk.risk_score:.2f}/100")
    print(f"Rule score        : {result.rules.rule_score * 100:.2f}/100")
    print(f"Overall risk      : {result.overall_risk_score:.2f}/100")
    print(f"Risk level        : {result.risk_level}")
    print(f"Requires review   : {result.requires_review}")
    print(f"Rules triggered   : {result.rules.rules_triggered}")
    print(f"AML indicators    : {result.transaction_pattern.indicators}")

    print("\nEvidence:")
    for item in result.evidence:
        print(f"  - {item}")

    print("\nAll 11 ML transaction features: PRESENT")
    print("Pydantic contract validation: PASSED")
    print("Risk Agent execution: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
