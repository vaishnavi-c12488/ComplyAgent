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
# PYDANTIC CONTRACTS
# ============================================================

class IdentityInput(BaseModel):
    """
    Validated identity information passed from the onboarding
    layer to the Screening Agent.
    """

    customer_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    date_of_birth: str = Field(min_length=1)
    document_number: str = Field(min_length=1)
    nationality: str = Field(min_length=1)
    country: str = Field(min_length=1)


class SanctionsMatch(BaseModel):
    """
    A single sanctions/PEP screening match.

    This structure prevents free-text information from being
    passed between agents.
    """

    matched: bool
    source: Literal[
        "OFAC",
        "UN",
        "PEP",
        "NONE",
    ]
    matched_name: str = ""
    match_score: float = Field(
        ge=0.0,
        le=100.0,
    )
    reason: str = ""


class ScreeningEvidence(BaseModel):
    """
    Strict output contract of the Screening Agent.

    This object is designed to be consumed by the Risk Agent.
    """

    customer_id: str
    document_id: str

    sanctions_match: SanctionsMatch

    screening_status: Literal[
        "clear",
        "potential_match",
        "confirmed_match",
        "manual_review",
    ]

    requires_review: bool

    evidence: list[str] = Field(
        default_factory=list
    )


# ============================================================
# SANCTIONS TOOL INTERFACE
# ============================================================

def sanctions_lookup(
    full_name: str,
    date_of_birth: str,
    nationality: str,
    country: str,
) -> SanctionsMatch:
    """
    Interface for the custom sanctions_lookup MCP tool.

    The production implementation must query the real OFAC/UN
    sanctions data. We intentionally do not fabricate a
    sanctions database here.

    This function currently provides a clear integration
    boundary for the MCP tool.
    """

    try:
        from mcp_server.tools.sanctions_lookup import (
            lookup_sanctions,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The sanctions_lookup MCP tool is not available. "
            "Create/configure "
            "mcp_server/tools/sanctions_lookup.py "
            "with the real OFAC/UN sanctions data before "
            "running the Screening Agent."
        ) from exc

    result = lookup_sanctions(
        full_name=full_name,
        date_of_birth=date_of_birth,
        nationality=nationality,
        country=country,
    )

    if isinstance(result, SanctionsMatch):
        return result

    if not isinstance(result, dict):
        raise TypeError(
            "sanctions_lookup must return either a "
            "SanctionsMatch object or a dictionary."
        )

    return SanctionsMatch.model_validate(result)


# ============================================================
# SCREENING LOGIC
# ============================================================

def determine_screening_status(
    sanctions_match: SanctionsMatch,
) -> tuple[
    Literal[
        "clear",
        "potential_match",
        "confirmed_match",
        "manual_review",
    ],
    bool,
]:
    """
    Convert sanctions evidence into a structured screening
    status.

    The Screening Agent does not make the final onboarding
    decision. It only produces screening evidence for the
    downstream Risk and Decision Agents.
    """

    if sanctions_match.matched:

        if sanctions_match.match_score >= 90:
            return "confirmed_match", True

        return "potential_match", True

    return "clear", False


# ============================================================
# SCREENING AGENT
# ============================================================

def run_screening_agent(
    identity: IdentityInput,
) -> ScreeningEvidence:
    """
    Run sanctions/PEP screening for one synthetic applicant.
    """

    sanctions_match = sanctions_lookup(
        full_name=identity.full_name,
        date_of_birth=identity.date_of_birth,
        nationality=identity.nationality,
        country=identity.country,
    )

    status, requires_review = determine_screening_status(
        sanctions_match
    )

    evidence = [
        f"Sanctions/PEP match: "
        f"{sanctions_match.matched}",
        f"Screening source: "
        f"{sanctions_match.source}",
        f"Match score: "
        f"{sanctions_match.match_score:.2f}",
    ]

    if sanctions_match.matched:
        evidence.append(
            "Potential or confirmed sanctions/PEP match "
            "requires downstream review."
        )
    else:
        evidence.append(
            "No sanctions/PEP match was returned by the "
            "screening tool."
        )

    if sanctions_match.reason:
        evidence.append(
            f"Tool reason: {sanctions_match.reason}"
        )

    return ScreeningEvidence(
        customer_id=identity.customer_id,
        document_id=identity.document_number,
        sanctions_match=sanctions_match,
        screening_status=status,
        requires_review=requires_review,
        evidence=evidence,
    )


# ============================================================
# SYNTHETIC IDENTITY LOADING
# ============================================================

IDENTITY_DATA_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "identity_data.csv"
)


def load_identity_data():
    """
    Load the synthetic identity records generated for
    ComplyAgent.
    """

    import pandas as pd

    if not IDENTITY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic identity data not found: "
            f"{IDENTITY_DATA_PATH}"
        )

    df = pd.read_csv(
        IDENTITY_DATA_PATH
    )

    required_columns = {
        "customer_id",
        "full_name",
        "date_of_birth",
        "document_number",
        "nationality",
        "country",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Identity data is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Synthetic identity dataset is empty."
        )

    return df


# ============================================================
# VALIDATION / DEMO
# ============================================================

def main():
    print("=" * 70)
    print("COMPLYAGENT - SCREENING AGENT")
    print("=" * 70)

    df = load_identity_data()

    print()
    print(
        f"Synthetic identities available: {len(df)}"
    )

    # --------------------------------------------------------
    # Validate the first synthetic identity before calling the
    # sanctions MCP tool.
    # --------------------------------------------------------

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

    print()
    print(
        f"Testing customer: "
        f"{identity.customer_id}"
    )

    print(
        f"Name: {identity.full_name}"
    )

    print()
    print(
        "Identity input contract validation: PASSED"
    )

    # --------------------------------------------------------
    # Run screening.
    # --------------------------------------------------------

    try:

        result = run_screening_agent(
            identity
        )

    except RuntimeError as exc:

        print()
        print(
            "Screening Agent execution is waiting for "
            "the sanctions MCP tool."
        )

        print()
        print(str(exc))

        return

    # --------------------------------------------------------
    # Validate final contract.
    # --------------------------------------------------------

    if not isinstance(
        result,
        ScreeningEvidence,
    ):
        raise TypeError(
            "Screening Agent did not return "
            "ScreeningEvidence."
        )

    print()
    print(
        f"Screening status: "
        f"{result.screening_status}"
    )

    print(
        f"Sanctions match: "
        f"{result.sanctions_match.matched}"
    )

    print(
        f"Requires review: "
        f"{result.requires_review}"
    )

    print()
    print(
        "Pydantic contract validation: PASSED"
    )

    print(
        "Screening Agent execution: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()