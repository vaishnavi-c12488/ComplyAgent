"""
ComplyAgent - FastAPI Application

Provides an API endpoint for running the complete KYC/AML
compliance workflow:

Upload document
    ↓
Document Intake
    ↓
Document Agent
    ↓
Screening Agent
    ↓
Risk Agent
    ↓
Decision Agent
    ↓
PostgreSQL Audit Trace
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.screening_agent import (
    IdentityInput,
    load_identity_data,
)
from orchestration.graph import build_graph
from vision.document_intake import analyze_uploaded_document


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CASE_MAPPING_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "case_mapping.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ComplyAgent API",
    description=(
        "KYC/AML compliance decision API powered by "
        "Document Intake, LangGraph agents, MCP tools, "
        "and PostgreSQL audit logging."
    ),
    version="1.0.0",
)


# ============================================================
# RESPONSE CONTRACT
# ============================================================


class ComplianceResponse(BaseModel):
    """Validated API response returned after assessment."""

    customer_id: str
    document_id: str

    document_type: str
    document_quality: str
    ocr_confidence: float
    ocr_word_count: int
    ela_score: float
    ela_suspicious: bool

    screening_status: str

    risk_level: str
    risk_score: float

    final_decision: str
    trace_complete: bool
    audit_trace_id: str


# ============================================================
# IDENTITY LOADING
# ============================================================


def get_identity(customer_id: str) -> IdentityInput:
    """
    Load one synthetic identity record by customer ID.
    """

    df = load_identity_data()

    matches = df[
        df["customer_id"].astype(str) == str(customer_id)
    ]

    if matches.empty:
        raise ValueError(
            f"Customer ID not found: {customer_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Duplicate customer ID found: {customer_id}"
        )

    row = matches.iloc[0]

    return IdentityInput(
        customer_id=str(row["customer_id"]),
        full_name=str(row["full_name"]),
        date_of_birth=str(row["date_of_birth"]),
        document_number=str(row["document_number"]),
        nationality=str(row["nationality"]),
        country=str(row["country"]),
    )


# ============================================================
# DOCUMENT TYPE LOOKUP
# ============================================================


def get_document_type(customer_id: str) -> str:
    """
    Resolve the synthetic customer's mapped MIDV document type.

    The case mapping is a deterministic synthetic linkage between
    a Faker-generated identity and one MIDV benchmark document.
    """

    if not CASE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Case mapping not found: {CASE_MAPPING_PATH}"
        )

    import pandas as pd

    mapping = pd.read_csv(CASE_MAPPING_PATH)

    required_columns = {
        "customer_id",
        "midv_document_type",
    }

    missing = required_columns - set(mapping.columns)

    if missing:
        raise ValueError(
            "Case mapping is missing required columns: "
            f"{sorted(missing)}"
        )

    matches = mapping[
        mapping["customer_id"].astype(str)
        == str(customer_id)
    ]

    if matches.empty:
        raise ValueError(
            f"No document mapping found for customer: "
            f"{customer_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple document mappings found for customer: "
            f"{customer_id}"
        )

    return str(matches.iloc[0]["midv_document_type"])


# ============================================================
# ROOT
# ============================================================


@app.get("/")
def root() -> dict[str, str]:
    """Basic API information."""

    return {
        "service": "ComplyAgent API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {
        "status": "healthy",
        "service": "complyagent-api",
    }


# ============================================================
# COMPLIANCE ASSESSMENT
# ============================================================


@app.post(
    "/compliance/check",
    response_model=ComplianceResponse,
)
async def compliance_check(
    customer_id: str = Form(
        ...,
        min_length=1,
        description=(
            "Synthetic customer ID from "
            "identity_data.csv."
        ),
    ),
    document: UploadFile = File(
        ...,
        description=(
            "Identity document image. "
            "Supported formats: JPG, JPEG, PNG, TIF, TIFF."
        ),
    ),
) -> ComplianceResponse:
    """
    Run the complete ComplyAgent compliance workflow.

    Request:
        multipart/form-data

        customer_id:
            Synthetic customer ID.

        document:
            Uploaded identity document.

    Workflow:

        uploaded document
            ↓
        Document Intake
            ↓
        LangGraph
            ↓
        Screening
            ↓
        Risk
            ↓
        Decision
            ↓
        Audit
    """

    try:
        # ----------------------------------------------------
        # 1. Validate customer
        # ----------------------------------------------------

        identity = get_identity(customer_id)

        # ----------------------------------------------------
        # 2. Validate uploaded document
        # ----------------------------------------------------

        if not document.filename:
            raise ValueError(
                "Uploaded document must have a filename."
            )

        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        }

        extension = Path(
            document.filename
        ).suffix.lower()

        if extension not in allowed_extensions:
            raise ValueError(
                "Unsupported document format. "
                "Use JPG, JPEG, PNG, TIF, or TIFF."
            )

        # ----------------------------------------------------
        # 3. Read uploaded image
        # ----------------------------------------------------

        file_bytes = await document.read()

        if not file_bytes:
            raise ValueError(
                "Uploaded document is empty."
            )

        # ----------------------------------------------------
        # 4. Resolve synthetic document type
        # ----------------------------------------------------

        document_type = get_document_type(
            customer_id
        )

        # ----------------------------------------------------
        # 5. Run Document Intake
        # ----------------------------------------------------

        _, document_evidence = (
            analyze_uploaded_document(
                file_bytes=file_bytes,
                document_id=identity.document_number,
                document_type=document_type,
            )
        )

        # ----------------------------------------------------
        # 6. Build LangGraph
        # ----------------------------------------------------

        workflow = build_graph()

        # ----------------------------------------------------
        # 7. Execute complete workflow
        # ----------------------------------------------------

        initial_state = {
            "identity": identity,
            "document": document_evidence,
        }

        result: dict[str, Any] = workflow.invoke(
            initial_state
        )

        # ----------------------------------------------------
        # 8. Extract agent results
        # ----------------------------------------------------

        screening = result.get("screening")
        risk = result.get("risk")
        decision = result.get("decision")
        audit_trace = result.get("audit_trace")

        if screening is None:
            raise RuntimeError(
                "Compliance workflow did not produce "
                "ScreeningEvidence."
            )

        if risk is None:
            raise RuntimeError(
                "Compliance workflow did not produce "
                "RiskEvidence."
            )

        if decision is None:
            raise RuntimeError(
                "Compliance workflow did not produce "
                "DecisionEvidence."
            )

        if audit_trace is None:
            raise RuntimeError(
                "Compliance workflow did not produce "
                "an audit trace."
            )

        # ----------------------------------------------------
        # 9. Final audit validation
        # ----------------------------------------------------

        if decision.trace_complete is not True:
            raise RuntimeError(
                "Compliance decision rejected because "
                "the audit trace is incomplete."
            )

        # ----------------------------------------------------
        # 10. Return structured response
        # ----------------------------------------------------

        return ComplianceResponse(
            customer_id=decision.customer_id,
            document_id=decision.document_id,

            document_type=document_evidence.document_type,
            document_quality=(
                document_evidence.document_quality
            ),
            ocr_confidence=round(
                document_evidence.ocr.confidence,
                2,
            ),
            ocr_word_count=(
                document_evidence.ocr.word_count
            ),
            ela_score=round(
                document_evidence.tamper.ela_score,
                4,
            ),
            ela_suspicious=(
                document_evidence.tamper.suspicious
            ),

            screening_status=(
                screening.screening_status
            ),

            risk_level=risk.risk_level,
            risk_score=round(
                risk.overall_risk_score,
                2,
            ),

            final_decision=decision.decision,
            trace_complete=decision.trace_complete,

            audit_trace_id=str(
                audit_trace["trace_id"]
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Compliance workflow failed: "
                f"{exc}"
            ),
        ) from exc