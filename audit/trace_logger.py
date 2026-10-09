"""
ComplyAgent - Audit Trace Logger

Persists complete compliance traces to:
1. JSONL file for local audit/debugging.
2. PostgreSQL for persistent structured audit storage.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import psycopg


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIT_DIR = PROJECT_ROOT / "audit" / "logs"
AUDIT_LOG_PATH = AUDIT_DIR / "decision_traces.jsonl"


# ============================================================
# POSTGRESQL CONFIGURATION
# ============================================================

DB_HOST = os.getenv("COMPLYAGENT_DB_HOST", "localhost")
DB_PORT = int(os.getenv("COMPLYAGENT_DB_PORT", "5432"))
DB_NAME = os.getenv("COMPLYAGENT_DB_NAME", "complyagent")
DB_USER = os.getenv("COMPLYAGENT_DB_USER", "postgres")
DB_PASSWORD = os.getenv("COMPLYAGENT_DB_PASSWORD")


# ============================================================
# TRACE VALIDATION
# ============================================================

REQUIRED_TRACE_FIELDS = {
    "trace_id",
    "timestamp",
    "customer_id",
    "document_id",
    "document_evidence",
    "screening_evidence",
    "risk_evidence",
    "decision_evidence",
    "final_decision",
}


def validate_trace(trace: dict[str, Any]) -> bool:
    """Validate that a trace contains all required pipeline stages."""

    if not isinstance(trace, dict):
        return False

    if not REQUIRED_TRACE_FIELDS.issubset(trace.keys()):
        return False

    required_values = [
        trace.get("trace_id"),
        trace.get("timestamp"),
        trace.get("customer_id"),
        trace.get("document_id"),
        trace.get("document_evidence"),
        trace.get("screening_evidence"),
        trace.get("risk_evidence"),
        trace.get("decision_evidence"),
        trace.get("final_decision"),
    ]

    if any(
        value is None or value == ""
        for value in required_values
    ):
        return False

    for field in [
        "document_evidence",
        "screening_evidence",
        "risk_evidence",
        "decision_evidence",
    ]:
        if not isinstance(trace[field], dict):
            return False

    return True


# ============================================================
# TRACE CREATION
# ============================================================

def create_trace(
    document_evidence: dict[str, Any],
    screening_evidence: dict[str, Any],
    risk_evidence: dict[str, Any],
    decision_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build a complete audit trace from all four pipeline stages."""

    customer_ids = {
        document_evidence.get("customer_id"),
        screening_evidence.get("customer_id"),
        risk_evidence.get("customer_id"),
        decision_evidence.get("customer_id"),
    }

    document_ids = {
        document_evidence.get("document_id"),
        screening_evidence.get("document_id"),
        risk_evidence.get("document_id"),
        decision_evidence.get("document_id"),
    }

    if len(customer_ids) != 1:
        raise ValueError(
            "Customer ID mismatch across agent evidence."
        )

    if len(document_ids) != 1:
        raise ValueError(
            "Document ID mismatch across agent evidence."
        )

    customer_id = next(iter(customer_ids))
    document_id = next(iter(document_ids))

    decision = decision_evidence.get("decision")

    if not decision:
        raise ValueError("Final decision is missing.")

    trace_id = f"{customer_id}-{document_id}"

    timestamp = datetime.now(timezone.utc).isoformat()

    trace = {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "customer_id": customer_id,
        "document_id": document_id,
        "document_evidence": document_evidence,
        "screening_evidence": screening_evidence,
        "risk_evidence": risk_evidence,
        "decision_evidence": decision_evidence,
        "final_decision": decision,
    }

    if not validate_trace(trace):
        raise ValueError(
            "Generated audit trace is incomplete."
        )

    return trace


# ============================================================
# JSONL PERSISTENCE
# ============================================================

def save_trace(
    trace: dict[str, Any],
) -> Path:
    """
    Persist one complete trace to JSONL.

    This preserves the original audit behaviour.
    """

    if not validate_trace(trace):
        raise ValueError(
            "Incomplete trace cannot be persisted."
        )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                trace,
                ensure_ascii=False,
            )
            + "\n"
        )

    return AUDIT_LOG_PATH


# ============================================================
# POSTGRESQL CONNECTION
# ============================================================

def get_db_connection():
    """Create a PostgreSQL connection."""

    if not DB_PASSWORD:
        raise RuntimeError(
            "COMPLYAGENT_DB_PASSWORD environment variable is not set."
        )

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# POSTGRESQL PERSISTENCE
# ============================================================

def save_trace_postgres(
    trace: dict[str, Any],
) -> None:
    """
    Persist one complete trace to PostgreSQL.
    """

    if not validate_trace(trace):
        raise ValueError(
            "Incomplete trace cannot be persisted."
        )

    risk_score = float(
        trace["risk_evidence"].get(
            "overall_risk_score",
            0.0,
        )
    )

    risk_level = str(
        trace["risk_evidence"].get(
            "risk_level",
            "low",
        )
    )

    screening_status = str(
        trace["screening_evidence"].get(
            "screening_status",
            "unknown",
        )
    )

    trace_complete = bool(
        trace["decision_evidence"].get(
            "trace_complete",
            True,
        )
    )

    with get_db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO decision_audit (
                    trace_id,
                    timestamp,
                    customer_id,
                    document_id,
                    final_decision,
                    risk_level,
                    risk_score,
                    screening_status,
                    trace_complete,
                    document_evidence,
                    screening_evidence,
                    risk_evidence,
                    decision_evidence
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s::jsonb,
                    %s::jsonb,
                    %s::jsonb
                )
                ON CONFLICT (trace_id)
                DO UPDATE SET
                    timestamp = EXCLUDED.timestamp,
                    final_decision = EXCLUDED.final_decision,
                    risk_level = EXCLUDED.risk_level,
                    risk_score = EXCLUDED.risk_score,
                    screening_status = EXCLUDED.screening_status,
                    trace_complete = EXCLUDED.trace_complete,
                    document_evidence = EXCLUDED.document_evidence,
                    screening_evidence = EXCLUDED.screening_evidence,
                    risk_evidence = EXCLUDED.risk_evidence,
                    decision_evidence = EXCLUDED.decision_evidence
                """,
                (
                    trace["trace_id"],
                    trace["timestamp"],
                    trace["customer_id"],
                    trace["document_id"],
                    trace["final_decision"],
                    risk_level,
                    risk_score,
                    screening_status,
                    trace_complete,
                    json.dumps(trace["document_evidence"]),
                    json.dumps(trace["screening_evidence"]),
                    json.dumps(trace["risk_evidence"]),
                    json.dumps(trace["decision_evidence"]),
                ),
            )


# ============================================================
# TRACE READBACK
# ============================================================

def load_traces() -> list[dict[str, Any]]:
    """Load all persisted JSONL audit traces."""

    if not AUDIT_LOG_PATH.exists():
        return []

    traces = []

    with AUDIT_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            traces.append(
                json.loads(line)
            )

    return traces


# ============================================================
# END-TO-END AUDIT TEST
# ============================================================

def main() -> None:
    print("=" * 70)
    print("COMPLYAGENT - AUDIT TRACE LOGGER")
    print("=" * 70)

    document = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "status": "processed",
        "quality": "medium",
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
            "Low synthetic AML risk."
        ],
    }

    decision = {
        "customer_id": "CUST00001",
        "document_id": "DOC00000001",
        "decision": "APPROVE",
        "decision_reason": (
            "No blocking compliance condition."
        ),
        "trace_complete": True,
    }

    print("\nCreating complete trace...")

    trace = create_trace(
        document_evidence=document,
        screening_evidence=screening,
        risk_evidence=risk,
        decision_evidence=decision,
    )

    assert validate_trace(trace)

    print("Complete trace validation: PASSED")

    path = save_trace(trace)

    print(f"JSONL trace saved to: {path}")

    traces = load_traces()

    assert len(traces) >= 1
    assert validate_trace(traces[-1])

    print("JSONL readback validation: PASSED")

    print("\nSaving trace to PostgreSQL...")

    save_trace_postgres(trace)

    print("PostgreSQL persistence: PASSED")

    print(f"Trace ID: {trace['trace_id']}")
    print(f"Final decision: {trace['final_decision']}")

    print("\n" + "=" * 70)
    print("Audit Trace Logger: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()