
from pathlib import Path
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import pytesseract


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "document_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "ocr_results.csv"
)

NUM_DOCUMENTS = 100


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Priority:
# 1. TESSERACT_CMD environment variable
# 2. Tesseract executable found on system PATH
# 3. Existing Windows installation path

TESSERACT_PATH = Path(
    os.getenv(
        "TESSERACT_CMD",
        shutil.which("tesseract")
        or r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    )
)


def configure_tesseract():
    """Configure and verify the installed Tesseract executable."""

    executable = str(TESSERACT_PATH)

    # Verify the executable exists when an explicit path is used.
    if not Path(executable).is_file():
        raise FileNotFoundError(
            "Tesseract executable not found. "
            "Install Tesseract OCR or set the TESSERACT_CMD "
            "environment variable to its executable path. "
            f"Configured path: {executable}"
        )

    pytesseract.pytesseract.tesseract_cmd = executable

    version = pytesseract.get_tesseract_version()

    print(f"Tesseract version: {version}")

    return version


# ============================================================
# FILE HELPERS
# ============================================================

def load_metadata() -> pd.DataFrame:
    """Load and validate document metadata."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = {
        "document_id",
        "document_type",
        "image_path",
        "ground_truth_path",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Document metadata is empty.")

    return df


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_image(path: Path) -> np.ndarray:
    """Read a document image."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    return image


# ============================================================
# DOCUMENT QUAD
# ============================================================

def get_document_quad(ground_truth: dict):
    """Extract the document-level quadrilateral."""

    quad = ground_truth.get("quad")

    if quad is None:
        raise ValueError(
            "Sample ground-truth JSON does not contain 'quad'."
        )

    if len(quad) != 4:
        raise ValueError(
            "Document quad must contain exactly four points."
        )

    points = np.asarray(
        quad,
        dtype=np.float32,
    )

    if points.shape != (4, 2):
        raise ValueError(
            f"Unexpected quad shape: {points.shape}"
        )

    return points


def order_points(points: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]

    return ordered


def rectify_document(
    image: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """Perspective-correct the document region."""

    points = order_points(quad)

    width_top = np.linalg.norm(points[1] - points[0])
    width_bottom = np.linalg.norm(points[2] - points[3])
    height_left = np.linalg.norm(points[3] - points[0])
    height_right = np.linalg.norm(points[2] - points[1])

    width = max(
        int(round(max(width_top, width_bottom))),
        100,
    )

    height = max(
        int(round(max(height_left, height_right))),
        100,
    )

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        points,
        destination,
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return corrected


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Prepare the rectified document for OCR."""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    height, width = gray.shape

    # Upscale the document to improve recognition.
    scale = 2

    gray = cv2.resize(
        gray,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    # Mild denoising.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Adaptive threshold handles uneven document lighting.
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return processed


# ============================================================
# OCR
# ============================================================

def run_ocr(image: np.ndarray):
    """Run Tesseract OCR and return text, confidence, and word count."""

    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []

    for index, text in enumerate(data["text"]):
        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            continue

        if confidence < 0:
            continue

        words.append(text)
        confidences.append(confidence)

    extracted_text = " ".join(words)

    average_confidence = (
        float(np.mean(confidences))
        if confidences
        else 0.0
    )

    return (
        extracted_text,
        average_confidence,
        len(words),
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(row):
    """Rectify one document and run OCR."""

    image_path = PROJECT_ROOT / row["image_path"]
    ground_truth_path = PROJECT_ROOT / row["ground_truth_path"]

    image = read_image(image_path)
    ground_truth = load_json(ground_truth_path)

    quad = get_document_quad(ground_truth)

    corrected = rectify_document(
        image,
        quad,
    )

    processed = preprocess_for_ocr(corrected)

    text, confidence, word_count = run_ocr(processed)

    return {
        "document_id": row["document_id"],
        "document_type": row["document_type"],
        "image_path": row["image_path"],
        "ground_truth_path": row["ground_truth_path"],
        "ocr_text": text,
        "ocr_confidence": round(confidence, 2),
        "word_count": word_count,
        "document_quad_available": True,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the document-level OCR pipeline."""

    print("=" * 70)
    print("COMPLYAGENT - DOCUMENT-LEVEL OCR")
    print("=" * 70)

    configure_tesseract()

    df = load_metadata()

    sample_size = min(
        NUM_DOCUMENTS,
        len(df),
    )

    selected = df.sample(
        n=sample_size,
        random_state=42,
    ).reset_index(drop=True)

    print(f"Processing {sample_size} documents...")
    print()

    results = []

    for index, row in selected.iterrows():
        result = process_document(row)
        results.append(result)

        print(
            f"[{index + 1:03d}/{sample_size:03d}] "
            f"{row['document_id']} | "
            f"confidence={result['ocr_confidence']:.2f} | "
            f"words={result['word_count']}"
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("DOCUMENT-LEVEL OCR COMPLETE")
    print("=" * 70)

    print(f"Documents processed       : {len(results_df)}")
    print(
        "Average OCR confidence    : "
        f"{results_df['ocr_confidence'].mean():.2f}"
    )
    print(
        "Documents with OCR words  : "
        f"{(results_df['word_count'] > 0).sum()}"
    )

    print()
    print(f"Results saved to:\n{OUTPUT_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
