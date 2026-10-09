"""
ComplyAgent - Document Intake

Processes one uploaded identity document using the existing
OpenCV, Tesseract OCR, ELA, and Document Agent components.

Pipeline:

    Uploaded Document
            |
            v
    OpenCV Image Decode
            |
            v
    Tesseract OCR
            |
            v
    ELA Tamper Analysis
            |
            v
    DocumentEvidence
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING COMPLYAGENT COMPONENTS
# ============================================================

from vision.ocr_extractor import (
    configure_tesseract,
    run_ocr,
)

from vision.tamper_detection_ela import (
    calculate_ela,
)

from agents.document_agent import (
    DocumentEvidence,
    build_document_evidence,
    load_ela_threshold,
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

SUPPORTED_IMAGE_TYPES = {
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
}


# ============================================================
# IMAGE DECODING
# ============================================================

def decode_uploaded_image(
    file_bytes: bytes,
) -> np.ndarray:
    """
    Decode uploaded image bytes into an OpenCV BGR image.
    """

    if not file_bytes:
        raise ValueError(
            "Uploaded document is empty."
        )

    image_array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Unable to decode the uploaded document image. "
            "Use a valid JPG, JPEG, PNG, TIF, or TIFF image."
        )

    if image.size == 0:
        raise ValueError(
            "Uploaded document image contains no image data."
        )

    return image


# ============================================================
# DOCUMENT ANALYSIS
# ============================================================

def analyze_uploaded_document(
    file_bytes: bytes,
    document_id: str,
    document_type: str,
) -> tuple[np.ndarray, DocumentEvidence]:
    """
    Analyze one uploaded identity document.

    Steps:

        1. Validate document identifiers.
        2. Decode the uploaded image.
        3. Configure Tesseract.
        4. Run OCR.
        5. Calculate ELA.
        6. Load the calibrated ELA threshold.
        7. Build the DocumentEvidence contract.

    Returns:

        image:
            Decoded OpenCV BGR image.

        document_evidence:
            Validated DocumentEvidence object.
    """

    # --------------------------------------------------------
    # Validate identifiers
    # --------------------------------------------------------

    document_id = str(
        document_id
    ).strip()

    document_type = str(
        document_type
    ).strip()

    if not document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    if not document_type:
        raise ValueError(
            "document_type cannot be empty."
        )

    # --------------------------------------------------------
    # Decode uploaded image
    # --------------------------------------------------------

    image = decode_uploaded_image(
        file_bytes
    )

    # --------------------------------------------------------
    # Configure Tesseract
    #
    # This uses the existing project configuration:
    #
    # C:\Program Files\Tesseract-OCR\tesseract.exe
    # --------------------------------------------------------

    configure_tesseract()

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    (
        ocr_text,
        ocr_confidence,
        word_count,
    ) = run_ocr(
        image
    )

    # --------------------------------------------------------
    # Normalize OCR outputs
    # --------------------------------------------------------

    ocr_text = str(
        ocr_text
    )

    ocr_confidence = float(
        ocr_confidence
    )

    word_count = int(
        word_count
    )

    ocr_confidence = max(
        0.0,
        min(
            100.0,
            ocr_confidence,
        ),
    )

    word_count = max(
        0,
        word_count,
    )

    # --------------------------------------------------------
    # ELA TAMPER ANALYSIS
    #
    # calculate_ela() returns:
    #
    #     (score, ela_visual)
    #
    # We only need the numeric score for the
    # DocumentEvidence contract.
    # --------------------------------------------------------

    (
        ela_score,
        _ela_visual,
    ) = calculate_ela(
        image
    )

    ela_score = float(
        ela_score
    )

    # --------------------------------------------------------
    # LOAD CALIBRATED ELA THRESHOLD
    # --------------------------------------------------------

    ela_threshold = float(
        load_ela_threshold()
    )

    # --------------------------------------------------------
    # BUILD DOCUMENT EVIDENCE
    # --------------------------------------------------------

    document_evidence = build_document_evidence(
        document_id=document_id,
        document_type=document_type,
        ocr_text=ocr_text,
        ocr_confidence=ocr_confidence,
        word_count=word_count,
        ela_score=ela_score,
        ela_threshold=ela_threshold,
    )

    # --------------------------------------------------------
    # CONTRACT VALIDATION
    # --------------------------------------------------------

    if not isinstance(
        document_evidence,
        DocumentEvidence,
    ):
        raise TypeError(
            "Document processing did not return "
            "a valid DocumentEvidence contract."
        )

    if (
        document_evidence.document_id
        != document_id
    ):
        raise ValueError(
            "DocumentEvidence document_id mismatch."
        )

    return (
        image,
        document_evidence,
    )


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:
    """
    Test Document Intake using one existing MIDV-500 image.
    """

    print("=" * 70)
    print("COMPLYAGENT - DOCUMENT INTAKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Metadata path
    # --------------------------------------------------------

    metadata_path = (
        PROJECT_ROOT
        / "data_generation"
        / "synthetic_documents"
        / "document_metadata.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Document metadata not found: {metadata_path}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        metadata_path
    )

    if metadata.empty:
        raise ValueError(
            "Document metadata is empty."
        )

    required_columns = {
        "document_id",
        "document_type",
        "image_path",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Document metadata is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Select first benchmark document
    # --------------------------------------------------------

    sample = metadata.iloc[0]

    document_id = str(
        sample["document_id"]
    )

    document_type = str(
        sample["document_type"]
    )

    image_path = Path(
        str(sample["image_path"])
    )

    if not image_path.is_absolute():
        image_path = (
            PROJECT_ROOT
            / image_path
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Document image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Read image bytes
    # --------------------------------------------------------

    file_bytes = image_path.read_bytes()

    # --------------------------------------------------------
    # Analyze document
    # --------------------------------------------------------

    _, evidence = analyze_uploaded_document(
        file_bytes=file_bytes,
        document_id=document_id,
        document_type=document_type,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print(
        f"Document ID       : "
        f"{evidence.document_id}"
    )

    print(
        f"Document Type     : "
        f"{evidence.document_type}"
    )

    print(
        f"Document Quality  : "
        f"{evidence.document_quality}"
    )

    print(
        f"OCR Confidence    : "
        f"{evidence.ocr.confidence:.2f}"
    )

    print(
        f"OCR Word Count    : "
        f"{evidence.ocr.word_count}"
    )

    print(
        f"ELA Score         : "
        f"{evidence.tamper.ela_score:.4f}"
    )

    print(
        f"ELA Threshold     : "
        f"{evidence.tamper.threshold:.4f}"
    )

    print(
        f"ELA Suspicious    : "
        f"{evidence.tamper.suspicious}"
    )

    print(
        f"Requires Review   : "
        f"{evidence.requires_review}"
    )

    print()
    print("Extracted OCR Text:")
    print("-" * 70)

    if evidence.ocr.text:
        print(
            evidence.ocr.text
        )
    else:
        print(
            "[No OCR text extracted]"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OpenCV image decoding       : PASSED")
    print("Tesseract configuration     : PASSED")
    print("OCR extraction              : PASSED")
    print("ELA analysis                : PASSED")
    print("DocumentEvidence contract   : PASSED")
    print("Document Intake             : PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()