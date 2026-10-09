
"""
ComplyAgent — Held-out AML Risk and Decision Evaluation.

Evaluates the existing trained model on a reproducible test split.
Results are based on synthetic data and are not real-world AML validation.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data_generation" / "aml_transaction_patterns.csv"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "decision_evaluation_results.csv"


# ------------------------------------------------------------
# Imports from the existing project
# ------------------------------------------------------------

from risk_model.ml_risk_scorer import predict_risk

FEATURE_COLUMNS = [
    "transaction_count_24h",
    "total_amount_24h",
    "avg_transaction_amount",
    "max_transaction_amount",
    "cash_ratio",
    "international_ratio",
    "high_risk_jurisdiction_ratio",
    "rapid_movement_count",
    "structuring_count",
    "new_beneficiary_count",
    "round_amount_ratio",
]

TARGET_COLUMN = "high_risk"


def load_data() -> pd.DataFrame:
    """Load and validate the synthetic AML dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Generate the AML dataset before running this evaluation."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("The AML dataset is empty.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    df = df.copy()
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("high_risk must contain only 0 and 1.")

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    return df


def main() -> None:
    print("=" * 68)
    print("COMPLYAGENT — HELD-OUT AML RISK EVALUATION")
    print("=" * 68)

    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Both target classes need at least two records for a "
            "stratified train/test split."
        )

    # Reproduce the project's 80/20 split and random seed.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Evaluate the already-trained model; do not retrain it here.
    predictions = []
    risk_scores = []
    predicted_decisions = []

    for _, row in X_test.iterrows():
        result = predict_risk(row.to_dict())

        prediction = int(result["prediction"])
        score = float(result["risk_score"])

        if prediction not in (0, 1):
            raise ValueError(
                f"Model returned an invalid prediction: {prediction}"
            )

        if not 0 <= score <= 100:
            raise ValueError(
                f"Model returned an invalid risk score: {score}"
            )

        predictions.append(prediction)
        risk_scores.append(score)

        # Simplified evaluation policy:
        # a predicted high-risk case is sent for review.
        predicted_decisions.append(
            "REVIEW" if score >= 45 else "APPROVE"
        )

    y_true = y_test.to_numpy()
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Build an auditable row-level evaluation file.
    results = df.loc[X_test.index].copy()
    results["actual_high_risk"] = y_true
    results["predicted_high_risk"] = y_pred
    results["risk_score"] = risk_scores
    results["predicted_decision"] = predicted_decisions
    results["prediction_correct"] = (
        results["actual_high_risk"] == results["predicted_high_risk"]
    )

    # This compares the simplified policy with the synthetic label.
    results["expected_decision"] = [
        "REVIEW" if label == 1 else "APPROVE"
        for label in y_true
    ]
    results["decision_agreement"] = (
        results["expected_decision"] == results["predicted_decision"]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDataset records       : {len(df)}")
    print(f"Test records          : {len(X_test)}")
    print(f"Actual low-risk cases : {int((y_true == 0).sum())}")
    print(f"Actual high-risk cases: {int((y_true == 1).sum())}")

    print("\n--- Held-out model metrics ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 score  : {f1:.4f}")

    print("\n--- Classification report ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Low risk", "High risk"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    print(
        "\nSimplified decision-policy agreement: "
        f"{results['decision_agreement'].mean():.4f}"
    )
    print(f"\nResults saved to:\n{OUTPUT_PATH}")
    print("\nNote: These metrics evaluate synthetic data only.")
    print("The simplified policy is not a substitute for the full Decision Agent.")
    print("=" * 68)


if __name__ == "__main__":
    main()
