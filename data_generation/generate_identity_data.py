"""
ComplyAgent - Synthetic KYC Identity Data Generator

Generates completely synthetic customer/KYC records for development
and evaluation. No real personally identifiable information (PII)
is used.

Output:
    data_generation/synthetic_documents/identity_data.csv
"""

from pathlib import Path

import pandas as pd
from faker import Faker


# Reproducibility: the same seed produces the same synthetic dataset.
SEED = 42
NUM_RECORDS = 100

OUTPUT_DIR = Path("data_generation/synthetic_documents")
OUTPUT_FILE = OUTPUT_DIR / "identity_data.csv"

fake = Faker("en_IN")
Faker.seed(SEED)


def generate_identity_data(num_records: int = NUM_RECORDS) -> pd.DataFrame:
    """Generate synthetic KYC identity records."""

    if num_records <= 0:
        raise ValueError("num_records must be greater than 0.")

    records = []

    for index in range(1, num_records + 1):
        records.append(
            {
                "customer_id": f"CUST{index:05d}",
                "full_name": fake.name(),
                "date_of_birth": fake.date_of_birth(
                    minimum_age=18,
                    maximum_age=75,
                ).isoformat(),
                "document_number": (
                    f"DOC{index:08d}"
                ),
                "document_type": fake.random_element(
                    elements=(
                        "passport",
                        "national_id",
                        "driving_license",
                    )
                ),
                "nationality": "Indian",
                "address": fake.address().replace("\n", ", "),
                "city": fake.city(),
                "country": "India",
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Generate and save the synthetic identity dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = generate_identity_data()

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("=" * 60)
    print("ComplyAgent Synthetic Identity Data")
    print("=" * 60)
    print(f"Records generated : {len(dataframe)}")
    print(f"Columns           : {len(dataframe.columns)}")
    print(f"Output            : {OUTPUT_FILE}")
    print()
    print("Column names:")
    print(", ".join(dataframe.columns))
    print()
    print("First 5 records:")
    print(dataframe.head().to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()