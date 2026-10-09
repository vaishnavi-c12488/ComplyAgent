"""
ComplyAgent - Gradient Boosted AML Risk Scorer

Trains a lightweight GradientBoostingClassifier on the
synthetic AML transaction-pattern dataset.

The model is a portfolio-project demonstration and is NOT
validated for real-world regulatory or production use.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "aml_transaction_patterns.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "risk_model"
    / "artifacts"
)

MODEL_PATH = (
    MODEL_DIR
    / "aml_gradient_boosting.joblib"
)


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


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"AML dataset not found:\n{DATA_PATH}\n\n"
            "Run:\n"
            "python data_generation/aml_typology_generator.py"
        )

    df = pd.read_csv(DATA_PATH)

    required = set(FEATURE_COLUMNS)
    required.add("high_risk")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing ML columns: {sorted(missing)}"
        )

    return df


def train_model() -> GradientBoostingClassifier:
    df = load_dataset()

    X = df[FEATURE_COLUMNS]
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    print("=" * 70)
    print("COMPLYAGENT - ML RISK SCORER")
    print("=" * 70)

    print(
        f"\nTraining rows : {len(X_train)}"
    )
    print(
        f"Testing rows  : {len(X_test)}"
    )

    print("\nHeld-out synthetic evaluation:")
    print(
        f"Accuracy  : {accuracy:.4f}"
    )
    print(
        f"Precision : {precision:.4f}"
    )
    print(
        f"Recall    : {recall:.4f}"
    )
    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )

    print(
        "\nML risk scorer training: PASSED"
    )

    return model


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train_model()

    return joblib.load(
        MODEL_PATH
    )


def predict_risk(
    transaction_pattern: dict,
) -> dict:
    bundle = load_model()

    model = bundle["model"]

    row = {
        feature: transaction_pattern.get(
            feature,
            0.0,
        )
        for feature in FEATURE_COLUMNS
    }

    X = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )

    probability = float(
        model.predict_proba(X)[0][1]
    )

    prediction = int(
        model.predict(X)[0]
    )

    return {
        "prediction": prediction,
        "risk_probability": round(
            probability,
            4,
        ),
        "risk_score": round(
            probability * 100,
            2,
        ),
        "model": "GradientBoostingClassifier",
    }


if __name__ == "__main__":
    train_model()