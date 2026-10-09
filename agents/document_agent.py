from pathlib import Path
from typing import Literal
import sys

import cv2
import pandas as pd
from pydantic import BaseModel, Field


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vision.tamper_detection_ela import calculate_ela


# ============================================================
# PROJECT PATHS
# ============================================================

SYNTHETIC_DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
)

OCR_RESULTS_PATH = (
    SYNTHETIC_DOCUMENTS_DIR
    / "ocr_results.csv"
)

DOCUMENT_METADATA_PATH = (
    SYNTHETIC_DOCUMENTS_DIR
    / "document_metadata.csv"
)

ELA_RESULTS_PATH = (
    SYNTHETIC_DOCUMENTS_DIR
    / "ela_evaluation.csv"
)


# ============================================================
# PYDANTIC CONTRACTS
# ============================================================

class OCRResult(BaseModel):
    """Validated OCR evidence produced by the document pipeline."""

    text: str = ""
    confidence: float = Field(ge=0.0, le=100.0)
    word_count: int = Field(ge=0)


class TamperEvidence(BaseModel):
    """Validated tamper-detection evidence."""

    ela_score: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    suspicious: bool
    method: Literal["ELA"] = "ELA"


class DocumentEvidence(BaseModel):
    """
    Strict output contract of the Document Agent.

    This object is designed to be consumed by the
    Screening Agent later in the LangGraph pipeline.
    """

    document_id: str
    document_type: str

    ocr: OCRResult
    tamper: TamperEvidence

    document_quality: Literal[
        "high",
        "medium",
        "low",
    ]

    requires_review: bool

    evidence: list[str] = Field(default_factory=list)


# ============================================================
# DATA LOADING
# ============================================================

def load_ocr_results() -> pd.DataFrame:
    """Load and validate OCR results."""

    if not OCR_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"OCR results not found: {OCR_RESULTS_PATH}"
        )

    df = pd.read_csv(OCR_RESULTS_PATH)

    required_columns = {
        "document_id",
        "document_type",
        "ocr_text",
        "ocr_confidence",
        "word_count",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"OCR results are missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("OCR results are empty.")

    if df["document_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["document_id"].duplicated(keep=False),
                "document_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate document IDs found in OCR results: "
            f"{duplicates[:10]}"
        )

    return df


def load_document_metadata() -> pd.DataFrame:
    """Load and validate source-document metadata."""

    if not DOCUMENT_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Document metadata not found: "
            f"{DOCUMENT_METADATA_PATH}"
        )

    df = pd.read_csv(DOCUMENT_METADATA_PATH)

    required_columns = {
        "document_id",
        "document_type",
        "image_path",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Document metadata is missing columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Document metadata is empty."
        )

    if df["document_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["document_id"].duplicated(keep=False),
                "document_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate document IDs found in document metadata: "
            f"{duplicates[:10]}"
        )

    return df


def load_ela_threshold() -> float:
    """
    Load the calibrated ELA threshold from the evaluation file.

    The evaluation file contains normal and tampered images.
    It is used here only to obtain the previously calibrated
    threshold.
    """

    if not ELA_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"ELA evaluation results not found: "
            f"{ELA_RESULTS_PATH}"
        )

    df = pd.read_csv(ELA_RESULTS_PATH)

    if df.empty:
        raise ValueError(
            "ELA evaluation results are empty."
        )

    if "threshold" not in df.columns:
        raise ValueError(
            "ELA evaluation results must contain "
            "'threshold'."
        )

    thresholds = (
        pd.to_numeric(
            df["threshold"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No valid ELA threshold found."
        )

    if len(thresholds) > 1:
        raise ValueError(
            "Multiple ELA thresholds found. "
            "Expected one calibrated threshold."
        )

    threshold = float(thresholds[0])

    if threshold < 0:
        raise ValueError(
            "ELA threshold cannot be negative."
        )

    return threshold


# ============================================================
# IMAGE LOADING
# ============================================================

def load_source_image(image_path: str):
    """Load an original source document image."""

    path = Path(image_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Source document image not found: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read source document image: {path}"
        )

    return image


# ============================================================
# DOCUMENT QUALITY
# ============================================================

def determine_document_quality(
    ocr_confidence: float,
    word_count: int,
) -> Literal["high", "medium", "low"]:
    """
    Assign document-quality evidence from OCR signals.

    This is an evidence-quality classification, not a
    document authenticity decision.
    """

    if ocr_confidence >= 60 and word_count >= 20:
        return "high"

    if ocr_confidence >= 30 and word_count >= 5:
        return "medium"

    return "low"


# ============================================================
# DOCUMENT EVIDENCE
# ============================================================

def build_document_evidence(
    document_id: str,
    document_type: str,
    ocr_text: str,
    ocr_confidence: float,
    word_count: int,
    ela_score: float,
    ela_threshold: float,
) -> DocumentEvidence:
    """Build the validated Document Agent output."""

    ocr = OCRResult(
        text="" if pd.isna(ocr_text) else str(ocr_text),
        confidence=float(ocr_confidence),
        word_count=int(word_count),
    )

    tamper_suspicious = (
        float(ela_score) > float(ela_threshold)
    )

    tamper = TamperEvidence(
        ela_score=float(ela_score),
        threshold=float(ela_threshold),
        suspicious=tamper_suspicious,
    )

    quality = determine_document_quality(
        ocr_confidence=float(ocr_confidence),
        word_count=int(word_count),
    )

    evidence = [
        f"OCR confidence: {ocr.confidence:.2f}",
        f"OCR word count: {ocr.word_count}",
        f"ELA score: {tamper.ela_score:.4f}",
        f"ELA threshold: {tamper.threshold:.4f}",
    ]

    if tamper_suspicious:
        evidence.append(
            "ELA signal exceeds the configured threshold."
        )

    if quality == "low":
        evidence.append(
            "Low OCR evidence quality requires additional review."
        )

    requires_review = (
        tamper_suspicious
        or quality == "low"
    )

    return DocumentEvidence(
        document_id=document_id,
        document_type=document_type,
        ocr=ocr,
        tamper=tamper,
        document_quality=quality,
        requires_review=requires_review,
        evidence=evidence,
    )


# ============================================================
# BATCH PROCESSING
# ============================================================

def run_document_agent(
    limit: int | None = None,
) -> list[DocumentEvidence]:
    """
    Process OCR and direct ELA evidence into validated
    DocumentEvidence objects.

    ELA is calculated directly on each original source
    document.

    The ELA evaluation CSV is used only for the calibrated
    threshold and is not merged with OCR results.
    """

    # --------------------------------------------------------
    # Load OCR results.
    # --------------------------------------------------------

    ocr_df = load_ocr_results()

    # --------------------------------------------------------
    # Load document metadata.
    # --------------------------------------------------------

    metadata_df = load_document_metadata()

    # --------------------------------------------------------
    # Load calibrated ELA threshold.
    # --------------------------------------------------------

    ela_threshold = load_ela_threshold()

    # --------------------------------------------------------
    # Merge OCR results with document metadata.
    #
    # Both files contain image_path, so Pandas creates:
    #
    #   image_path_ocr
    #   image_path_metadata
    #
    # We intentionally use image_path_metadata because the
    # document metadata is the authoritative source for the
    # original MIDV document image.
    # --------------------------------------------------------

    merged = ocr_df.merge(
        metadata_df,
        on="document_id",
        how="inner",
        suffixes=("_ocr", "_metadata"),
    )

    if merged.empty:
        raise ValueError(
            "No documents could be matched between OCR "
            "results and document metadata."
        )

    # --------------------------------------------------------
    # Verify required merged columns.
    # --------------------------------------------------------

    required_merged_columns = {
        "document_id",
        "document_type_ocr",
        "ocr_text",
        "ocr_confidence",
        "word_count",
        "image_path_metadata",
    }

    missing = (
        required_merged_columns
        - set(merged.columns)
    )

    if missing:
        raise ValueError(
            "Merged OCR/metadata data is missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Verify one record per document.
    # --------------------------------------------------------

    if merged["document_id"].duplicated().any():
        duplicates = (
            merged.loc[
                merged["document_id"].duplicated(keep=False),
                "document_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate document IDs detected after "
            "OCR/metadata merge: "
            f"{duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Apply processing limit.
    # --------------------------------------------------------

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        merged = merged.head(limit)

    # --------------------------------------------------------
    # Process documents.
    # --------------------------------------------------------

    results: list[DocumentEvidence] = []

    for _, row in merged.iterrows():

        document_id = str(
            row["document_id"]
        )

        document_type = str(
            row["document_type_ocr"]
        )

        # IMPORTANT:
        # Use the metadata image path, not the OCR image path.
        image_path = str(
            row["image_path_metadata"]
        )

        # ----------------------------------------------------
        # Load original document image.
        # ----------------------------------------------------

        image = load_source_image(
            image_path
        )

        # ----------------------------------------------------
        # Calculate ELA directly on this document.
        # ----------------------------------------------------

        ela_score, _ = calculate_ela(
            image
        )

        # ----------------------------------------------------
        # Build validated DocumentEvidence.
        # ----------------------------------------------------

        result = build_document_evidence(
            document_id=document_id,
            document_type=document_type,
            ocr_text=str(row["ocr_text"]),
            ocr_confidence=float(
                row["ocr_confidence"]
            ),
            word_count=int(
                row["word_count"]
            ),
            ela_score=float(ela_score),
            ela_threshold=ela_threshold,
        )

        results.append(result)

    return results


# ============================================================
# VALIDATION / DEMO
# ============================================================

def main():
    print("=" * 70)
    print("COMPLYAGENT - DOCUMENT AGENT")
    print("=" * 70)

    results = run_document_agent(
        limit=10
    )

    print()
    print(
        f"Documents processed: {len(results)}"
    )
    print()

    # --------------------------------------------------------
    # Check document ID uniqueness.
    # --------------------------------------------------------

    document_ids = [
        result.document_id
        for result in results
    ]

    if len(document_ids) != len(set(document_ids)):
        raise RuntimeError(
            "Document Agent produced duplicate document IDs."
        )

    # --------------------------------------------------------
    # Display first five results.
    # --------------------------------------------------------

    for result in results[:5]:

        print(
            f"{result.document_id} | "
            f"quality={result.document_quality} | "
            f"OCR={result.ocr.confidence:.2f} | "
            f"ELA={result.tamper.ela_score:.4f} | "
            f"ELA suspicious={result.tamper.suspicious} | "
            f"review={result.requires_review}"
        )

    print()
    print(
        "Pydantic contract validation: PASSED"
    )
    print(
        "Document ID uniqueness check: PASSED"
    )
    print(
        "Document Agent execution: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()