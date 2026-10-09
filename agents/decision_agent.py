
"""
ComplyAgent - Decision Agent

Consumes structured ScreeningEvidence and RiskEvidence and
produces the final onboarding decision.

Decision outcomes:
    APPROVE
    REVIEW
    REJECT

The Decision Agent synthesizes evidence from upstream agents.
It does not create new risk evidence.
"""

from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT UPSTREAM CONTRACTS
# ============================================================

from agents.screening_agent import (
    IdentityInput,
    load_identity_data,
    run_screening_agent,
    ScreeningEvidence,
)

from agents.risk_agent import (
    run_risk_agent,
    RiskEvidence,
)


# ============================================================
# DECISION CONTRACT
# ============================================================

class DecisionEvidence(BaseModel):
    """Final structured output of the Decision Agent."""

    customer_id: str
    document_id: str

    decision: Literal[
        "APPROVE",
        "REVIEW",
        "REJECT",
    ]

    decision_reason: str

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    overall_risk_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    screening_status: Literal[
        "clear",
        "potential_match",
        "confirmed_match",
        "manual_review",
    ]

    requires_review: bool

    screening_evidence: ScreeningEvidence
    risk_evidence: RiskEvidence

    evidence: list[str] = Field(default_factory=list)

    trace_complete: bool


# ============================================================
# DECISION LOGIC
# ============================================================

def determine_decision(
    screening: ScreeningEvidence,
    risk: RiskEvidence,
) -> tuple[Literal["APPROVE", "REVIEW", "REJECT"], str]:
    """
    Determine the final onboarding decision.

    Priority:
        1. High-confidence sanctions match -> REJECT
        2. Other positive sanctions match -> REVIEW
        3. Critical risk -> REJECT
        4. High risk -> REVIEW
        5. Screening review required -> REVIEW
        6. Risk review required -> REVIEW
        7. Medium risk -> REVIEW
        8. Otherwise -> APPROVE

    The sanctions threshold is a demonstration policy, not a
    substitute for an organization's approved compliance policy.
    """

    sanctions_match = screening.sanctions_match

    # A positive, high-confidence match is escalated to rejection
    # under this demonstration policy.
    if (
        sanctions_match.matched
        and sanctions_match.match_score >= 90
    ):
        return (
            "REJECT",
            "High-confidence sanctions match detected; "
            "escalation must follow the applicable compliance policy.",
        )

    # Any other positive sanctions match must not fall through
    # to approval.
    if sanctions_match.matched:
        return (
            "REVIEW",
            "Potential sanctions match requires manual compliance review.",
        )

    if risk.risk_level == "critical":
        return (
            "REJECT",
            "Critical compliance risk detected.",
        )

    if risk.risk_level == "high":
        return (
            "REVIEW",
            "High compliance risk requires manual review.",
        )

    if screening.requires_review:
        return (
            "REVIEW",
            "Screening evidence requires manual review.",
        )

    if risk.requires_review:
        return (
            "REVIEW",
            "Risk evidence requires manual review.",
        )

    if risk.risk_level == "medium":
        return (
            "REVIEW",
            "Medium compliance risk requires manual review.",
        )

    return (
        "APPROVE",
        "No blocking compliance condition was identified "
        "and the synthetic risk level is low.",
    )


# ============================================================
# AUDIT TRACE VALIDATION
# ============================================================

def validate_complete_trace(
    screening: ScreeningEvidence,
    risk: RiskEvidence,
) -> bool:
    """
    Verify that required upstream evidence exists and refers
    to the same customer and document.
    """

    if not screening.customer_id or not screening.document_id:
        return False

    if not risk.customer_id or not risk.document_id:
        return False

    if screening.customer_id != risk.customer_id:
        return False

    if screening.document_id != risk.document_id:
        return False

    if not screening.evidence or not risk.evidence:
        return False

    if screening.sanctions_match is None:
        return False

    if risk.rules is None:
        return False

    if risk.ml_risk is None:
        return False

    if risk.transaction_pattern is None:
        return False

    return True


# ============================================================
# DECISION AGENT
# ============================================================

def run_decision_agent(
    screening: ScreeningEvidence,
    risk: RiskEvidence,
) -> DecisionEvidence:
    """Produce the final structured onboarding decision."""

    # 1. Verify customer/document consistency.
    if screening.customer_id != risk.customer_id:
        raise ValueError(
            "Customer ID mismatch between Screening and Risk evidence."
        )

    if screening.document_id != risk.document_id:
        raise ValueError(
            "Document ID mismatch between Screening and Risk evidence."
        )

    # 2. Require a complete upstream trace before deciding.
    trace_complete = validate_complete_trace(screening, risk)

    if not trace_complete:
        raise ValueError(
            "Decision cannot be produced because the upstream "
            "audit trace is incomplete."
        )

    # 3. Determine the decision.
    decision, reason = determine_decision(screening, risk)

    # 4. Combine upstream evidence.
    evidence = [
        *screening.evidence,
        *risk.evidence,
        f"Final decision: {decision}.",
        f"Decision reason: {reason}",
    ]

    # 5. Return the validated contract.
    return DecisionEvidence(
        customer_id=screening.customer_id,
        document_id=screening.document_id,
        decision=decision,
        decision_reason=reason,
        risk_level=risk.risk_level,
        overall_risk_score=risk.overall_risk_score,
        screening_status=screening.screening_status,
        requires_review=(
            screening.requires_review
            or risk.requires_review
            or decision == "REVIEW"
        ),
        screening_evidence=screening,
        risk_evidence=risk,
        evidence=evidence,
        trace_complete=True,
    )


# ============================================================
# END-TO-END TEST
# ============================================================

def main() -> None:
    print("=" * 70)
    print("COMPLYAGENT - DECISION AGENT")
    print("=" * 70)

    # 1. Load synthetic identity.
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

    # 2. Screening Agent.
    print("\nRunning Screening Agent...")
    screening = run_screening_agent(identity)

    print(f"Screening status: {screening.screening_status}")

    # 3. Risk Agent.
    print("\nRunning Risk Agent...")
    risk = run_risk_agent(screening)

    print(f"Risk level: {risk.risk_level}")
    print(f"Risk score: {risk.overall_risk_score:.2f}/100")

    # 4. Decision Agent.
    print("\nRunning Decision Agent...")
    result = run_decision_agent(screening, risk)

    # 5. Contract and trace checks.
    assert isinstance(result, DecisionEvidence)

    assert (
        result.customer_id
        == screening.customer_id
        == risk.customer_id
    )

    assert (
        result.document_id
        == screening.document_id
        == risk.document_id
    )

    assert result.decision in {"APPROVE", "REVIEW", "REJECT"}

    assert 0 <= result.overall_risk_score <= 100
    assert result.trace_complete is True
    assert len(result.evidence) > 0

    if result.decision == "REJECT":
        assert (
            screening.sanctions_match.matched
            and screening.sanctions_match.match_score >= 90
        ) or risk.risk_level == "critical"

    # 6. Display final result.
    print("\n" + "-" * 70)
    print("FINAL DECISION")
    print("-" * 70)

    print(f"Customer ID       : {result.customer_id}")
    print(f"Document ID       : {result.document_id}")
    print(f"Decision          : {result.decision}")
    print(f"Reason            : {result.decision_reason}")
    print(f"Risk level        : {result.risk_level}")
    print(f"Risk score        : {result.overall_risk_score:.2f}/100")
    print(f"Screening status  : {result.screening_status}")
    print(f"Requires review   : {result.requires_review}")
    print(f"Trace complete    : {result.trace_complete}")

    print("\nEvidence:")
    for item in result.evidence:
        print(f"  - {item}")

    print("\n" + "=" * 70)
    print("Pydantic contract validation: PASSED")
    print("Complete evidence trace: PASSED")
    print("Decision Agent execution: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
