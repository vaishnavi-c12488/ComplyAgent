from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IDENTITY_FILE = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "identity_data.csv"
)

DOCUMENT_METADATA_FILE = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "document_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "case_mapping.csv"
)


def main() -> None:
    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"Identity data not found: {IDENTITY_FILE}"
        )

    if not DOCUMENT_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Document metadata not found: {DOCUMENT_METADATA_FILE}"
        )

    identities = pd.read_csv(IDENTITY_FILE)
    documents = pd.read_csv(DOCUMENT_METADATA_FILE)

    required_identity_columns = {
        "customer_id",
        "full_name",
        "document_number",
        "document_type",
    }

    required_document_columns = {
        "document_id",
        "document_type",
        "source_dataset",
        "image_path",
        "ground_truth_path",
    }

    missing_identity = required_identity_columns - set(identities.columns)
    missing_documents = required_document_columns - set(documents.columns)

    if missing_identity:
        raise ValueError(
            f"Missing identity columns: {sorted(missing_identity)}"
        )

    if missing_documents:
        raise ValueError(
            f"Missing document metadata columns: {sorted(missing_documents)}"
        )

    identities = identities.sort_values("customer_id").reset_index(drop=True)
    documents = documents.sort_values("document_id").reset_index(drop=True)

    mapping_count = min(len(identities), len(documents))

    if mapping_count == 0:
        raise ValueError("No identities or document records available.")

    identities = identities.iloc[:mapping_count].copy()
    documents = documents.iloc[:mapping_count].copy()

    mapping = pd.DataFrame(
        {
            "customer_id": identities["customer_id"].values,
            "synthetic_document_number": identities["document_number"].values,
            "full_name": identities["full_name"].values,
            "identity_document_type": identities["document_type"].values,
            "midv_document_id": documents["document_id"].values,
            "midv_document_type": documents["document_type"].values,
            "midv_image_path": documents["image_path"].values,
            "midv_ground_truth_path": documents["ground_truth_path"].values,
            "source_dataset": documents["source_dataset"].values,
        }
    )

    if mapping["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id values found.")

    if mapping["midv_document_id"].duplicated().any():
        raise ValueError("Duplicate MIDV document IDs found.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_FILE, index=False)

    print("CASE MAPPING CREATED")
    print(f"Identity records used : {len(identities)}")
    print(f"MIDV documents used   : {len(documents)}")
    print(f"Mapping records       : {len(mapping)}")
    print(f"Output file           : {OUTPUT_FILE}")
    print()
    print("Validation:")
    print(
        f"Unique customer IDs   : "
        f"{mapping['customer_id'].nunique() == len(mapping)}"
    )
    print(
        f"Unique MIDV IDs       : "
        f"{mapping['midv_document_id'].nunique() == len(mapping)}"
    )


if __name__ == "__main__":
    main()