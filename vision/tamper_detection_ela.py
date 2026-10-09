from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_PREVIEWS = 20
JPEG_QUALITY = 90
THRESHOLD_PERCENTILE = 99

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "tamper_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
)

ELA_DIR = OUTPUT_ROOT / "ela_previews"

RESULTS_PATH = (
    OUTPUT_ROOT / "ela_evaluation.csv"
)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def read_image(path: Path) -> np.ndarray:
    """Read an image using OpenCV."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: {path}"
        )

    return image


def calculate_ela(
    image: np.ndarray,
    jpeg_quality: int = JPEG_QUALITY,
):
    """
    Calculate Error Level Analysis (ELA).

    The image is temporarily JPEG-compressed and compared
    with the original. Regions with larger differences may
    indicate localized editing or recompression artifacts.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            jpeg_quality,
        ],
    )

    if not success:
        raise ValueError(
            "JPEG encoding failed during ELA."
        )

    recompressed = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if recompressed is None:
        raise ValueError(
            "JPEG decoding failed during ELA."
        )

    difference = cv2.absdiff(
        image,
        recompressed,
    )

    gray_difference = cv2.cvtColor(
        difference,
        cv2.COLOR_BGR2GRAY,
    )

    score = float(
        np.mean(gray_difference)
    )

    # Normalize ELA image for visualization.
    max_value = float(
        gray_difference.max()
    )

    if max_value > 0:
        ela_visual = cv2.normalize(
            gray_difference,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
    else:
        ela_visual = np.zeros_like(
            gray_difference
        )

    return score, ela_visual


def save_ela_preview(
    path: Path,
    ela_image: np.ndarray,
):
    """Save an ELA visualization."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        ela_image,
    )

    if not success:
        raise IOError(
            f"Failed to save ELA preview: {path}"
        )


# ============================================================
# DATA VALIDATION
# ============================================================

def load_manifest() -> pd.DataFrame:
    """Load and validate the tamper metadata manifest."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "document_id",
        "capture_group",
        "document_type",
        "normal_image",
        "tampered_image",
        "tamper_type",
        "label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Tamper metadata manifest is empty."
        )

    return df


# ============================================================
# DATASET PREPARATION
# ============================================================

def build_evaluation_dataframe(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the normal/tampered manifest into a single
    evaluation table containing both classes.
    """

    normal_records = []

    tampered_records = []

    for _, row in manifest.iterrows():

        normal_records.append(
            {
                "document_id": row["document_id"],
                "capture_group": row["capture_group"],
                "document_type": row["document_type"],
                "image_path": row["normal_image"],
                "tamper_type": "normal",
                "true_label": 0,
            }
        )

        tampered_records.append(
            {
                "document_id": row["document_id"],
                "capture_group": row["capture_group"],
                "document_type": row["document_type"],
                "image_path": row["tampered_image"],
                "tamper_type": row["tamper_type"],
                "true_label": 1,
            }
        )

    evaluation_df = pd.DataFrame(
        normal_records + tampered_records
    )

    return evaluation_df


# ============================================================
# ELA PROCESSING
# ============================================================

def calculate_scores(
    evaluation_df: pd.DataFrame,
):
    """Calculate ELA scores for every document."""

    scores = []

    for index, row in evaluation_df.iterrows():

        image_path = (
            PROJECT_ROOT
            / row["image_path"]
        )

        image = read_image(
            image_path
        )

        score, ela_image = calculate_ela(
            image
        )

        scores.append(
            {
                "document_id": row["document_id"],
                "capture_group": row["capture_group"],
                "document_type": row["document_type"],
                "image_path": row["image_path"],
                "tamper_type": row["tamper_type"],
                "true_label": int(row["true_label"]),
                "ela_score": score,
            }
        )

        # Save only a manageable number of previews.
        if index < NUM_PREVIEWS:

            preview_name = (
                f"{index + 1:03d}_"
                f"{row['document_id']}_"
                f"{row['tamper_type']}.png"
            )

            preview_path = (
                ELA_DIR / preview_name
            )

            save_ela_preview(
                preview_path,
                ela_image,
            )

        print(
            f"[{index + 1:03d}/{len(evaluation_df):03d}] "
            f"{row['document_id']} | "
            f"{row['tamper_type']} | "
            f"ELA={score:.4f}"
        )

    return pd.DataFrame(scores)


# ============================================================
# THRESHOLD AND EVALUATION
# ============================================================

def evaluate_scores(
    results_df: pd.DataFrame,
):
    """
    Calculate a threshold using only normal documents.

    The threshold is the 99th percentile of normal ELA scores.
    """

    normal_scores = results_df.loc[
        results_df["true_label"] == 0,
        "ela_score",
    ]

    if normal_scores.empty:
        raise ValueError(
            "No normal documents found for threshold calibration."
        )

    threshold = float(
        np.percentile(
            normal_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    results_df = results_df.copy()

    results_df["predicted_label"] = (
        results_df["ela_score"] > threshold
    ).astype(int)

    y_true = results_df["true_label"]
    y_pred = results_df["predicted_label"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    tn, fp, fn, tp = matrix.ravel()

    normal_count = len(
        normal_scores
    )

    false_positive_rate = (
        fp / normal_count
        if normal_count > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("ELA TAMPER DETECTION RESULTS")
    print("=" * 60)

    print(
        f"Normal documents       : "
        f"{normal_count}"
    )

    print(
        f"Tampered documents     : "
        f"{int(results_df['true_label'].sum())}"
    )

    print(
        f"Threshold              : "
        f"{threshold:.4f}"
    )

    print(
        f"Accuracy               : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision              : "
        f"{precision:.4f}"
    )

    print(
        f"Recall                 : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score               : "
        f"{f1:.4f}"
    )

    print(
        f"False Positive Rate    : "
        f"{false_positive_rate:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Results by tamper type:")

    tampered_only = results_df[
        results_df["true_label"] == 1
    ]

    for tamper_type, group in (
        tampered_only.groupby("tamper_type")
    ):
        detection_rate = (
            group["predicted_label"].mean()
        )

        print(
            f"  {tamper_type:20s} : "
            f"{detection_rate:.4f}"
        )

    print("=" * 60)

    return results_df, threshold


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Run the complete ELA detection pipeline."""

    print("=" * 60)
    print("COMPLYAGENT - ELA TAMPER DETECTION")
    print("=" * 60)

    manifest = load_manifest()

    print(
        f"Loaded tamper manifest: "
        f"{len(manifest)} records"
    )

    evaluation_df = build_evaluation_dataframe(
        manifest
    )

    print(
        f"Evaluation images: "
        f"{len(evaluation_df)}"
    )

    results_df = calculate_scores(
        evaluation_df
    )

    results_df, threshold = evaluate_scores(
        results_df
    )

    results_df["threshold"] = threshold

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print()
    print(
        f"Evaluation CSV saved to:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"ELA previews saved to:\n"
        f"{ELA_DIR}"
    )

    print()
    print("ELA processing complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()