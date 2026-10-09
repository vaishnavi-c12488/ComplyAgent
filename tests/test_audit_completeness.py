"""
ComplyAgent - Audit Completeness Tests

The key requirement tested here:

    No final decision without a complete audit trace.
"""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from audit.trace_logger import (
    create_trace,
    validate_trace,
)


# ============================================================
# COMPLETE TRACE
# ============================================================

def complete_evidence():
    document = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "status": "processed",
    }

    screening = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "screening_status": "clear",
        "requires_review": False,
        "evidence": [
            "No sanctions match."
        ],
    }

    risk = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "risk_level": "low",
        "overall_risk_score": 1.91,
        "evidence": [
            "Low synthetic risk."
        ],
    }

    decision = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "decision": "APPROVE",
        "decision_reason": "Low risk.",
    }

    return (
        document,
        screening,
        risk,
        decision,
    )


# ============================================================
# TEST 1 - COMPLETE TRACE IS ACCEPTED
# ============================================================

def test_complete_trace_is_valid():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    trace = create_trace(
        document,
        screening,
        risk,
        decision,
    )

    assert validate_trace(trace) is True


# ============================================================
# TEST 2 - MISSING DOCUMENT EVIDENCE
# ============================================================

def test_missing_document_evidence_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    with pytest.raises(ValueError):
        create_trace(
            {},
            screening,
            risk,
            decision,
        )


# ============================================================
# TEST 3 - MISSING SCREENING EVIDENCE
# ============================================================

def test_missing_screening_evidence_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    with pytest.raises(ValueError):
        create_trace(
            document,
            {},
            risk,
            decision,
        )


# ============================================================
# TEST 4 - MISSING RISK EVIDENCE
# ============================================================

def test_missing_risk_evidence_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    with pytest.raises(ValueError):
        create_trace(
            document,
            screening,
            {},
            decision,
        )


# ============================================================
# TEST 5 - MISSING DECISION
# ============================================================

def test_missing_decision_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    incomplete_decision = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
    }

    with pytest.raises(ValueError):
        create_trace(
            document,
            screening,
            risk,
            incomplete_decision,
        )


# ============================================================
# TEST 6 - CUSTOMER MISMATCH
# ============================================================

def test_customer_mismatch_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    risk["customer_id"] = "CUST99999"

    with pytest.raises(ValueError):
        create_trace(
            document,
            screening,
            risk,
            decision,
        )


# ============================================================
# TEST 7 - DOCUMENT MISMATCH
# ============================================================

def test_document_mismatch_is_rejected():
    (
        document,
        screening,
        risk,
        decision,
    ) = complete_evidence()

    risk["document_id"] = "DOC99999999"

    with pytest.raises(ValueError):
        create_trace(
            document,
            screening,
            risk,
            decision,
        )


# ============================================================
# TEST 8 - RAW INCOMPLETE TRACE IS INVALID
# ============================================================

def test_incomplete_raw_trace_is_invalid():

    incomplete_trace = {
        "trace_id": "CUST00001-DOC00000001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "final_decision": "APPROVE",
    }

    assert (
        validate_trace(
            incomplete_trace
        )
        is False
    )