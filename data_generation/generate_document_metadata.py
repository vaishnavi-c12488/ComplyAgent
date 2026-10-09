"""
ComplyAgent - MIDV-500 Document Metadata Generator

Creates a metadata index for all usable MIDV-500 document samples.

Rules:
- Original MIDV source files are never modified.
- Images are searched recursively.
- Each image must have a matching ground-truth JSON.
- Matching is based on the image's relative path under `images/`
  and the corresponding relative path under `ground_truth/`.
- Top-level document-type template files are excluded.
- Ground-truth values are NOT copied into the metadata CSV.

Output:
    data_generation/synthetic_documents/document_metadata.csv
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "source"
    / "midv500"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "document_metadata.csv"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def project_relative_path(path: Path) -> str:
    """Return a project-relative POSIX path."""

    return path.relative_to(PROJECT_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_document_metadata() -> tuple[pd.DataFrame, list[str]]:
    """Build metadata for every valid MIDV image/JSON pair."""

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"MIDV-500 source directory not found:\n{SOURCE_DIR}"
        )

    records = []
    unmatched_images = []

    document_type_dirs = sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_dir()
    )

    if not document_type_dirs:
        raise RuntimeError(
            f"No MIDV document-type directories found in:\n{SOURCE_DIR}"
        )

    for document_type_dir in document_type_dirs:

        images_dir = document_type_dir / "images"
        ground_truth_dir = document_type_dir / "ground_truth"

        if not images_dir.is_dir():
            continue

        if not ground_truth_dir.is_dir():
            continue

        # Recursively find all actual document images.
        image_files = sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_file in image_files:

            # Relative path inside the images directory.
            relative_image = image_file.relative_to(images_dir)

            # The corresponding JSON has the same relative path
            # under ground_truth, with .json extension.
            ground_truth_file = (
                ground_truth_dir
                / relative_image.with_suffix(".json")
            )

            # Exclude the top-level document template:
            #
            # images/01_alb_id.tif
            # ground_truth/01_alb_id.json
            #
            # We want the actual document samples such as:
            #
            # images/CA/CA01_01.tif
            # ground_truth/CA/CA01_01.json
            if len(relative_image.parts) == 1:
                continue

            if not ground_truth_file.is_file():
                unmatched_images.append(
                    project_relative_path(image_file)
                )
                continue

            # Stable ID based on document type + relative sample path.
            relative_sample = relative_image.with_suffix("")

            document_id = (
                f"{document_type_dir.name}__"
                + "__".join(relative_sample.parts)
            )

            records.append(
                {
                    "document_id": document_id,
                    "document_type": document_type_dir.name,
                    "source_dataset": "MIDV-500",
                    "image_path": project_relative_path(image_file),
                    "ground_truth_path": project_relative_path(
                        ground_truth_file
                    ),
                }
            )

    if not records:
        raise RuntimeError(
            "No valid MIDV-500 document samples were found."
        )

    dataframe = pd.DataFrame(
        records,
        columns=[
            "document_id",
            "document_type",
            "source_dataset",
            "image_path",
            "ground_truth_path",
        ],
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    if dataframe["document_id"].duplicated().any():
        duplicates = (
            dataframe.loc[
                dataframe["document_id"].duplicated(keep=False),
                "document_id",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate document IDs detected:\n"
            + "\n".join(duplicates[:20])
        )

    if dataframe["image_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate image paths detected in metadata."
        )

    if dataframe["ground_truth_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate ground-truth paths detected in metadata."
        )

    return dataframe, unmatched_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate and validate the document metadata CSV."""

    dataframe, unmatched_images = build_document_metadata()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("=" * 70)
    print("ComplyAgent - MIDV-500 Document Metadata")
    print("=" * 70)

    print(f"Valid document pairs : {len(dataframe)}")
    print(
        f"Document types       : "
        f"{dataframe['document_type'].nunique()}"
    )
    print(f"Unmatched images      : {len(unmatched_images)}")
    print(f"Metadata rows         : {len(dataframe)}")
    print(f"Metadata columns      : {len(dataframe.columns)}")
    print(f"Output file           : {OUTPUT_FILE}")

    print()
    print("Documents by type:")
    print(
        dataframe["document_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("First 5 metadata records:")
    print(
        dataframe.head(5).to_string(index=False)
    )

    if unmatched_images:
        print()
        print("WARNING - images without matching JSON:")

        for image_path in unmatched_images[:20]:
            print(f"  - {image_path}")

        if len(unmatched_images) > 20:
            print(
                f"  ... and "
                f"{len(unmatched_images) - 20} more."
            )

    print("=" * 70)


if __name__ == "__main__":
    main()