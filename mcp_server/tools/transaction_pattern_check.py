
"""
ComplyAgent - Transaction Pattern MCP Tool

Reads synthetic AML transaction-pattern data and returns
structured risk evidence for a customer.

This is a deterministic local MCP-tool boundary for the
Risk Agent.
"""

from pathlib import Path
import hashlib

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "aml_transaction_patterns.csv"
)


REQUIRED_COLUMNS = {
    "case_id",
    "customer_id",
    "scenario",
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
    "high_risk",
}


def load_transaction_data() -> pd.DataFrame:
    """Load and validate the synthetic AML transaction dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"AML transaction dataset not found:\n{DATA_PATH}\n\n"
            "Run:\n"
            "python data_generation/aml_typology_generator.py"
        )

    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "AML transaction dataset is missing columns: "
            f"{sorted(missing)}"
        )

    if df.empty:
        raise ValueError("AML transaction dataset is empty.")

    if df["customer_id"].isna().any():
        raise ValueError(
            "AML transaction dataset contains missing customer IDs."
        )

    return df


def select_customer_case(
    df: pd.DataFrame,
    customer_id: str,
) -> pd.Series:
    """Select a stable synthetic transaction case for a customer."""

    customer_rows = df[
        df["customer_id"].astype(str) == str(customer_id)
    ]

    if customer_rows.empty:
        raise ValueError(
            f"No transaction pattern found for customer {customer_id}."
        )

    # Deterministic selection: the same customer receives the
    # same case when the underlying dataset remains unchanged.
    digest = hashlib.sha256(
        customer_id.encode("utf-8")
    ).hexdigest()

    index = int(digest[:8], 16) % len(customer_rows)

    return customer_rows.iloc[index]


def check_transaction_pattern(
    customer_id: str,
) -> dict:
    """Return structured AML transaction-pattern evidence."""

    df = load_transaction_data()

    row = select_customer_case(df, customer_id)

    high_risk = bool(int(row["high_risk"]))

    indicators = []

    if row["structuring_count"] >= 2:
        indicators.append("structuring_pattern")

    if row["rapid_movement_count"] >= 3:
        indicators.append("rapid_movement_pattern")

    if row["international_ratio"] >= 0.65:
        indicators.append("high_cross_border_activity")

    if row["high_risk_jurisdiction_ratio"] >= 0.55:
        indicators.append("high_risk_jurisdiction_exposure")

    if row["new_beneficiary_count"] >= 5:
        indicators.append("multiple_new_beneficiaries")

    if row["round_amount_ratio"] >= 0.60:
        indicators.append("high_round_amount_ratio")

    if high_risk and not indicators:
        indicators.append("elevated_synthetic_risk_score")

    return {
        "case_id": str(row["case_id"]),
        "customer_id": str(row["customer_id"]),
        "high_risk": high_risk,
        "scenario": str(row["scenario"]),
        "indicators": indicators,
        "transaction_count_24h": int(
            row["transaction_count_24h"]
        ),
        "total_amount_24h": float(
            row["total_amount_24h"]
        ),
        "avg_transaction_amount": float(
            row["avg_transaction_amount"]
        ),
        "max_transaction_amount": float(
            row["max_transaction_amount"]
        ),
        "cash_ratio": float(
            row["cash_ratio"]
        ),
        "international_ratio": float(
            row["international_ratio"]
        ),
        "high_risk_jurisdiction_ratio": float(
            row["high_risk_jurisdiction_ratio"]
        ),
        "rapid_movement_count": int(
            row["rapid_movement_count"]
        ),
        "structuring_count": int(
            row["structuring_count"]
        ),
        "new_beneficiary_count": int(
            row["new_beneficiary_count"]
        ),
        "round_amount_ratio": float(
            row["round_amount_ratio"]
        ),
    }


def main() -> None:
    print("=" * 70)
    print("COMPLYAGENT - TRANSACTION PATTERN TOOL")
    print("=" * 70)

    df = load_transaction_data()

    print(f"\nTransaction cases loaded: {len(df)}")

    result = check_transaction_pattern("CUST00001")

    print("\nTest customer: CUST00001")
    print(f"Scenario: {result['scenario']}")
    print(f"High risk: {result['high_risk']}")
    print(f"Indicators: {result['indicators']}")

    expected_features = {
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
    }

    missing_features = expected_features - set(result)

    if missing_features:
        raise AssertionError(
            f"Missing transaction features: {sorted(missing_features)}"
        )

    print("\nAll 11 ML transaction features: PRESENT")
    print("Transaction pattern tool: PASSED")


if __name__ == "__main__":
    main()
