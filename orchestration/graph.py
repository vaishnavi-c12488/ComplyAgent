"""
ComplyAgent - LangGraph Orchestration

Pipeline:

    START
      ↓
    Document Agent
      ↓
    Screening Agent
      ↓
    Risk Agent
      ↓
    Decision Agent
      ↓
    Audit Trace
      ↓
    END

All agent communication uses structured Pydantic contracts.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# AGENT IMPORTS
# ============================================================

from agents.screening_agent import (
    IdentityInput,
    ScreeningEvidence,
    run_screening_agent,
)

from agents.risk_agent import (
    RiskEvidence,
    run_risk_agent,
)

from agents.decision_agent import (
    DecisionEvidence,
    run_decision_agent,
)

from agents.document_agent import (
    DocumentEvidence,
)

from audit.trace_logger import save_trace


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ComplyAgentState(TypedDict, total=False):
    """
    Structured state passed through LangGraph.
    """

    identity: IdentityInput

    document: DocumentEvidence

    screening: ScreeningEvidence

    risk: RiskEvidence

    decision: DecisionEvidence

    audit_trace: dict[str, Any]


# ============================================================
# DOCUMENT NODE
# ============================================================

def document_node(
    state: ComplyAgentState,
) -> dict:
    """
    Validate document evidence supplied to the workflow.

    The actual image processing is performed by
    vision.document_intake before LangGraph invocation.
    """

    document = state.get("document")

    if document is None:
        raise ValueError(
            "Document Agent requires DocumentEvidence."
        )

    if not isinstance(
        document,
        DocumentEvidence,
    ):
        raise TypeError(
            "Document Agent output must be DocumentEvidence."
        )

    return {
        "document": document
    }


# ============================================================
# SCREENING NODE
# ============================================================

def screening_node(
    state: ComplyAgentState,
) -> dict:

    identity = state.get("identity")

    if identity is None:
        raise ValueError(
            "LangGraph state is missing IdentityInput."
        )

    screening = run_screening_agent(
        identity
    )

    if not isinstance(
        screening,
        ScreeningEvidence,
    ):
        raise TypeError(
            "Screening Agent did not return ScreeningEvidence."
        )

    return {
        "screening": screening
    }


# ============================================================
# RISK NODE
# ============================================================

def risk_node(
    state: ComplyAgentState,
) -> dict:

    screening = state.get("screening")

    if screening is None:
        raise ValueError(
            "Risk Agent requires ScreeningEvidence."
        )

    risk = run_risk_agent(
        screening
    )

    if not isinstance(
        risk,
        RiskEvidence,
    ):
        raise TypeError(
            "Risk Agent did not return RiskEvidence."
        )

    return {
        "risk": risk
    }


# ============================================================
# DECISION NODE
# ============================================================

def decision_node(
    state: ComplyAgentState,
) -> dict:

    screening = state.get("screening")

    risk = state.get("risk")

    if screening is None:
        raise ValueError(
            "Decision Agent requires ScreeningEvidence."
        )

    if risk is None:
        raise ValueError(
            "Decision Agent requires RiskEvidence."
        )

    decision = run_decision_agent(
        screening,
        risk,
    )

    if not isinstance(
        decision,
        DecisionEvidence,
    ):
        raise TypeError(
            "Decision Agent did not return DecisionEvidence."
        )

    return {
        "decision": decision
    }


# ============================================================
# AUDIT NODE
# ============================================================

def audit_node(
    state: ComplyAgentState,
) -> dict:
    """
    Create and persist the complete evidence trace.
    """

    document = state.get("document")

    screening = state.get("screening")

    risk = state.get("risk")

    decision = state.get("decision")

    # --------------------------------------------------------
    # Required evidence
    # --------------------------------------------------------

    if document is None:
        raise ValueError(
            "Audit requires DocumentEvidence."
        )

    if screening is None:
        raise ValueError(
            "Audit requires ScreeningEvidence."
        )

    if risk is None:
        raise ValueError(
            "Audit requires RiskEvidence."
        )

    if decision is None:
        raise ValueError(
            "Audit requires DecisionEvidence."
        )

    # --------------------------------------------------------
    # Type validation
    # --------------------------------------------------------

    if not isinstance(
        document,
        DocumentEvidence,
    ):
        raise TypeError(
            "Invalid DocumentEvidence."
        )

    if not isinstance(
        screening,
        ScreeningEvidence,
    ):
        raise TypeError(
            "Invalid ScreeningEvidence."
        )

    if not isinstance(
        risk,
        RiskEvidence,
    ):
        raise TypeError(
            "Invalid RiskEvidence."
        )

    if not isinstance(
        decision,
        DecisionEvidence,
    ):
        raise TypeError(
            "Invalid DecisionEvidence."
        )

    # --------------------------------------------------------
    # Cross-agent consistency
    # --------------------------------------------------------

    if not (
        screening.customer_id
        == risk.customer_id
        == decision.customer_id
    ):
        raise ValueError(
            "Audit rejected: customer ID mismatch."
        )

    if not (
        screening.document_id
        == risk.document_id
        == decision.document_id
    ):
        raise ValueError(
            "Audit rejected: document ID mismatch."
        )

    # --------------------------------------------------------
    # Complete decision trace required
    # --------------------------------------------------------

    if decision.trace_complete is not True:
        raise ValueError(
            "Audit rejected: decision trace is incomplete."
        )

    # --------------------------------------------------------
    # Convert Pydantic objects to JSON
    # --------------------------------------------------------

    document_data = document.model_dump(
        mode="json"
    )

    screening_data = screening.model_dump(
        mode="json"
    )

    risk_data = risk.model_dump(
        mode="json"
    )

    decision_data = decision.model_dump(
        mode="json"
    )

    # --------------------------------------------------------
    # Trace ID
    # --------------------------------------------------------

    trace_id = (
        f"{decision.customer_id}-"
        f"{decision.document_id}"
    )

    # --------------------------------------------------------
    # Complete audit trace
    # --------------------------------------------------------

    trace = {
        "trace_id": trace_id,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "customer_id":
            decision.customer_id,

        "document_id":
            decision.document_id,

        "document_evidence":
            document_data,

        "screening_evidence":
            screening_data,

        "risk_evidence":
            risk_data,

        "decision_evidence":
            decision_data,

        "final_decision":
            decision.decision,
    }

    # --------------------------------------------------------
    # Persist JSONL + PostgreSQL
    # --------------------------------------------------------

    save_trace(
        trace
    )

    print(
        f"\nAudit trace persisted: {trace_id}"
    )

    return {
        "audit_trace": trace
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():
    """
    Build and compile the complete ComplyAgent graph.
    """

    graph = StateGraph(
        ComplyAgentState
    )

    graph.add_node(
        "document",
        document_node,
    )

    graph.add_node(
        "screening",
        screening_node,
    )

    graph.add_node(
        "risk",
        risk_node,
    )

    graph.add_node(
        "decision",
        decision_node,
    )

    graph.add_node(
        "audit",
        audit_node,
    )

    # --------------------------------------------------------
    # Sequential workflow
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "document",
    )

    graph.add_edge(
        "document",
        "screening",
    )

    graph.add_edge(
        "screening",
        "risk",
    )

    graph.add_edge(
        "risk",
        "decision",
    )

    graph.add_edge(
        "decision",
        "audit",
    )

    graph.add_edge(
        "audit",
        END,
    )

    return graph.compile()


# ============================================================
# END-TO-END TEST
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "COMPLYAGENT - LANGGRAPH ORCHESTRATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load synthetic identity
    # --------------------------------------------------------

    from agents.screening_agent import (
        load_identity_data,
    )

    from vision.document_intake import (
        analyze_uploaded_document,
    )

    df = load_identity_data()

    row = df.iloc[0]

    identity = IdentityInput(
        customer_id=str(
            row["customer_id"]
        ),
        full_name=str(
            row["full_name"]
        ),
        date_of_birth=str(
            row["date_of_birth"]
        ),
        document_number=str(
            row["document_number"]
        ),
        nationality=str(
            row["nationality"]
        ),
        country=str(
            row["country"]
        ),
    )

    print(
        f"\nCustomer: "
        f"{identity.customer_id}"
    )

    # --------------------------------------------------------
    # Use one existing MIDV image for integration test
    # --------------------------------------------------------

    metadata_path = (
        PROJECT_ROOT
        / "data_generation"
        / "synthetic_documents"
        / "document_metadata.csv"
    )

    import pandas as pd

    metadata = pd.read_csv(
        metadata_path
    )

    sample = metadata.iloc[0]

    image_path = (
        PROJECT_ROOT
        / str(sample["image_path"])
    )

    image_bytes = image_path.read_bytes()

    _, document = analyze_uploaded_document(
        file_bytes=image_bytes,
        document_id=identity.document_number,
        document_type=str(
            sample["document_type"]
        ),
    )

    print(
        f"Document quality: "
        f"{document.document_quality}"
    )

    print(
        f"OCR confidence: "
        f"{document.ocr.confidence:.2f}"
    )

    print(
        f"ELA score: "
        f"{document.tamper.ela_score:.4f}"
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    app = build_graph()

    print(
        "\nLangGraph compiled: PASSED"
    )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    result = app.invoke(
        {
            "identity": identity,
            "document": document,
        }
    )

    screening = result.get(
        "screening"
    )

    risk = result.get(
        "risk"
    )

    decision = result.get(
        "decision"
    )

    audit_trace = result.get(
        "audit_trace"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert isinstance(
        document,
        DocumentEvidence,
    )

    assert isinstance(
        screening,
        ScreeningEvidence,
    )

    assert isinstance(
        risk,
        RiskEvidence,
    )

    assert isinstance(
        decision,
        DecisionEvidence,
    )

    assert isinstance(
        audit_trace,
        dict,
    )

    assert decision.trace_complete is True

    assert (
        screening.customer_id
        == risk.customer_id
        == decision.customer_id
    )

    assert (
        screening.document_id
        == risk.document_id
        == decision.document_id
    )

    assert (
        audit_trace["document_evidence"]
        == document.model_dump(
            mode="json"
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "DOCUMENT ANALYSIS"
    )

    print("-" * 70)

    print(
        f"Document ID       : "
        f"{document.document_id}"
    )

    print(
        f"Document Type     : "
        f"{document.document_type}"
    )

    print(
        f"Document Quality  : "
        f"{document.document_quality}"
    )

    print(
        f"OCR Confidence    : "
        f"{document.ocr.confidence:.2f}"
    )

    print(
        f"OCR Word Count    : "
        f"{document.ocr.word_count}"
    )

    print(
        f"ELA Score         : "
        f"{document.tamper.ela_score:.4f}"
    )

    print(
        f"ELA Suspicious    : "
        f"{document.tamper.suspicious}"
    )

    print("\n" + "-" * 70)

    print(
        "LANGGRAPH RESULT"
    )

    print("-" * 70)

    print(
        f"Customer ID       : "
        f"{decision.customer_id}"
    )

    print(
        f"Screening status  : "
        f"{screening.screening_status}"
    )

    print(
        f"Risk level        : "
        f"{risk.risk_level}"
    )

    print(
        f"Risk score        : "
        f"{risk.overall_risk_score:.2f}/100"
    )

    print(
        f"Final decision    : "
        f"{decision.decision}"
    )

    print(
        f"Trace complete    : "
        f"{decision.trace_complete}"
    )

    print("\n" + "=" * 70)

    print(
        "Document Agent: PASSED"
    )

    print(
        "Pydantic contracts: PASSED"
    )

    print(
        "LangGraph execution: PASSED"
    )

    print(
        "Audit trace creation: PASSED"
    )

    print(
        "JSONL + PostgreSQL persistence: PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()