"""
ComplyAgent - Synthetic AML Transaction Pattern Generator

Generates synthetic transaction-pattern data only.
No real customer or financial data is used.

The scenarios are inspired by AML risk concepts and typology
categories described by FATF. They are synthetic stress-test
patterns, not real-world fraud labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data_generation"
OUTPUT_PATH = OUTPUT_DIR / "aml_transaction_patterns.csv"


RANDOM_SEED = 42
N_ROWS = 5000


def generate_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    scenarios = [
        "normal",
        "structuring",
        "rapid_movement",
        "cross_border",
        "high_risk_jurisdiction",
        "multiple_typologies",
    ]

    for i in range(N_ROWS):
        scenario = rng.choice(
            scenarios,
            p=[
                0.50,
                0.12,
                0.12,
                0.10,
                0.08,
                0.08,
            ],
        )

        transaction_count_24h = int(
            rng.integers(1, 15)
        )

        avg_transaction_amount = float(
            rng.uniform(500, 25000)
        )

        total_amount_24h = float(
            transaction_count_24h
            * avg_transaction_amount
            * rng.uniform(0.7, 1.3)
        )

        max_transaction_amount = float(
            avg_transaction_amount
            * rng.uniform(1.0, 2.5)
        )

        cash_ratio = float(
            rng.uniform(0.0, 0.35)
        )

        international_ratio = float(
            rng.uniform(0.0, 0.40)
        )

        high_risk_jurisdiction_ratio = float(
            rng.uniform(0.0, 0.20)
        )

        rapid_movement_count = int(
            rng.integers(0, 3)
        )

        structuring_count = int(
            rng.integers(0, 2)
        )

        new_beneficiary_count = int(
            rng.integers(0, 5)
        )

        round_amount_ratio = float(
            rng.uniform(0.0, 0.35)
        )

        # ----------------------------------------------------
        # Scenario-specific synthetic patterns
        # ----------------------------------------------------

        if scenario == "structuring":
            transaction_count_24h = int(
                rng.integers(8, 25)
            )
            avg_transaction_amount = float(
                rng.uniform(8000, 9500)
            )
            total_amount_24h = float(
                transaction_count_24h
                * avg_transaction_amount
            )
            structuring_count = int(
                rng.integers(2, 7)
            )
            round_amount_ratio = float(
                rng.uniform(0.45, 0.90)
            )

        elif scenario == "rapid_movement":
            transaction_count_24h = int(
                rng.integers(8, 25)
            )
            rapid_movement_count = int(
                rng.integers(3, 10)
            )
            new_beneficiary_count = int(
                rng.integers(3, 10)
            )

        elif scenario == "cross_border":
            international_ratio = float(
                rng.uniform(0.65, 1.0)
            )
            new_beneficiary_count = int(
                rng.integers(2, 8)
            )

        elif scenario == "high_risk_jurisdiction":
            high_risk_jurisdiction_ratio = float(
                rng.uniform(0.55, 1.0)
            )
            international_ratio = float(
                rng.uniform(0.50, 1.0)
            )

        elif scenario == "multiple_typologies":
            transaction_count_24h = int(
                rng.integers(12, 30)
            )
            rapid_movement_count = int(
                rng.integers(4, 12)
            )
            structuring_count = int(
                rng.integers(2, 8)
            )
            international_ratio = float(
                rng.uniform(0.55, 1.0)
            )
            high_risk_jurisdiction_ratio = float(
                rng.uniform(0.45, 1.0)
            )
            new_beneficiary_count = int(
                rng.integers(5, 12)
            )
            round_amount_ratio = float(
                rng.uniform(0.50, 0.95)
            )

        # ----------------------------------------------------
        # Synthetic risk label
        # ----------------------------------------------------

        risk_signal = (
            0.10 * min(transaction_count_24h / 20, 1.0)
            + 0.12 * min(total_amount_24h / 250000, 1.0)
            + 0.08 * min(max_transaction_amount / 50000, 1.0)
            + 0.10 * cash_ratio
            + 0.15 * international_ratio
            + 0.18 * high_risk_jurisdiction_ratio
            + 0.12 * min(rapid_movement_count / 8, 1.0)
            + 0.15 * min(structuring_count / 5, 1.0)
            + 0.08 * min(new_beneficiary_count / 8, 1.0)
            + 0.07 * round_amount_ratio
        )

        risk_signal += float(
            rng.normal(0, 0.025)
        )

        high_risk = int(
            risk_signal >= 0.42
        )

        rows.append(
            {
                "case_id": f"AML{i + 1:06d}",
                "customer_id": f"CUST{(i % 100) + 1:05d}",
                "scenario": scenario,
                "transaction_count_24h": transaction_count_24h,
                "total_amount_24h": round(
                    total_amount_24h,
                    2,
                ),
                "avg_transaction_amount": round(
                    avg_transaction_amount,
                    2,
                ),
                "max_transaction_amount": round(
                    max_transaction_amount,
                    2,
                ),
                "cash_ratio": round(
                    cash_ratio,
                    4,
                ),
                "international_ratio": round(
                    international_ratio,
                    4,
                ),
                "high_risk_jurisdiction_ratio": round(
                    high_risk_jurisdiction_ratio,
                    4,
                ),
                "rapid_movement_count": rapid_movement_count,
                "structuring_count": structuring_count,
                "new_beneficiary_count": new_beneficiary_count,
                "round_amount_ratio": round(
                    round_amount_ratio,
                    4,
                ),
                "high_risk": high_risk,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("COMPLYAGENT - AML TYPOLOGY DATA GENERATOR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_data()

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nRows generated : {len(df)}")
    print(f"Output         : {OUTPUT_PATH}")
    print(
        f"High-risk cases: "
        f"{int(df['high_risk'].sum())}"
    )
    print(
        f"Low-risk cases : "
        f"{int((df['high_risk'] == 0).sum())}"
    )

    print("\nScenario distribution:")
    print(
        df["scenario"]
        .value_counts()
        .to_string()
    )

    print("\nAML dataset generation: PASSED")


if __name__ == "__main__":
    main()