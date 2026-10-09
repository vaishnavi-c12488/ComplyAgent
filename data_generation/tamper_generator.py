from pathlib import Path
import json
import random
import re

import cv2
import numpy as np
import pandas as pd
from faker import Faker


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_DOCUMENTS = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
    / "document_metadata.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data_generation"
    / "synthetic_documents"
)

NORMAL_DIR = OUTPUT_ROOT / "normal"
TAMPERED_DIR = OUTPUT_ROOT / "tampered"

TAMPER_TYPES = [
    "text_substitution",
    "region_copy",
    "region_inpaint",
    "compression_edit",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

fake = Faker("en_IN")
Faker.seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_metadata() -> pd.DataFrame:
    """Load and validate the existing MIDV metadata."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    df = pd.read_csv(METADATA_PATH)

    required_columns = {
        "document_id",
        "document_type",
        "source_dataset",
        "image_path",
        "ground_truth_path",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Metadata file contains no records.")

    return df


def load_ground_truth(path: Path) -> dict:
    """Load a MIDV ground-truth JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_image(path: Path) -> np.ndarray:
    """Read a TIFF image safely using OpenCV."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image and verify that OpenCV succeeded."""

    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Failed to save image: {path}")


def get_quad(field_data: dict):
    """Return a field quadrilateral as an integer OpenCV array."""

    quad = field_data.get("quad")

    if not quad or len(quad) != 4:
        return None

    points = np.array(quad, dtype=np.float32)

    if points.shape != (4, 2):
        return None

    return np.round(points).astype(np.int32)


def clamp_points(
    points: np.ndarray,
    width: int,
    height: int,
):
    """Keep polygon points inside image boundaries."""

    points = points.copy()

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    return points


def generate_synthetic_value(original_value: str) -> str:
    """
    Generate a synthetic replacement value using Faker.

    The replacement preserves a rough category based on the
    original text so that we do not insert arbitrary sensitive
    information.
    """

    value = str(original_value).strip()

    if not value:
        return fake.word().upper()

    if re.fullmatch(r"[A-Za-z .'-]+", value):
        generated = fake.name()
        return generated[: max(len(value) + 5, 12)]

    if re.fullmatch(r"\d+", value):
        return "".join(
            random.choices(
                "0123456789",
                k=max(len(value), 6),
            )
        )

    if re.fullmatch(r"[A-Za-z0-9]+", value):
        generated = fake.bothify(
            text="".join(
                "?" if c.isalpha() else "#"
                for c in value
            )
        )
        return generated.upper()

    return fake.word().upper()


def find_text_fields(ground_truth: dict):
    """Return valid text fields with quadrilateral coordinates."""

    fields = []

    for field_name, field_data in ground_truth.items():

        if not isinstance(field_data, dict):
            continue

        if "value" not in field_data or "quad" not in field_data:
            continue

        quad = get_quad(field_data)

        if quad is None:
            continue

        value = str(field_data["value"]).strip()

        if not value:
            continue

        fields.append(
            {
                "name": field_name,
                "value": value,
                "quad": quad,
            }
        )

    return fields


# ============================================================
# TAMPER METHODS
# ============================================================

def text_substitution(
    image: np.ndarray,
    field: dict,
) -> np.ndarray:
    """
    Replace a ground-truth text region with synthetic Faker text.

    The original field is first removed using OpenCV inpainting,
    then synthetic text is written into the same document region.
    """

    result = image.copy()

    height, width = result.shape[:2]

    quad = clamp_points(
        field["quad"],
        width,
        height,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [quad],
        255,
    )

    # Slightly expand the mask to cover the original characters.
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    result = cv2.inpaint(
        result,
        mask,
        3,
        cv2.INPAINT_TELEA,
    )

    x, y, w, h = cv2.boundingRect(quad)

    if w < 10 or h < 8:
        return result

    replacement = generate_synthetic_value(
        field["value"]
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(
        0.35,
        min(1.2, h / 35),
    )

    thickness = max(
        1,
        int(round(h / 30)),
    )

    text_size, _ = cv2.getTextSize(
        replacement,
        font,
        font_scale,
        thickness,
    )

    # Reduce font size until text fits the target region.
    while (
        text_size[0] > max(w - 4, 1)
        and font_scale > 0.2
    ):
        font_scale -= 0.05

        text_size, _ = cv2.getTextSize(
            replacement,
            font,
            font_scale,
            thickness,
        )

    text_x = x + 2

    text_y = y + max(
        text_size[1] + 2,
        h // 2,
    )

    text_y = min(
        text_y,
        y + h - 2,
    )

    cv2.putText(
        result,
        replacement,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return result


def region_copy(image: np.ndarray) -> np.ndarray:
    """Copy a small image region to another location."""

    result = image.copy()

    height, width = result.shape[:2]

    if width < 100 or height < 100:
        return result

    crop_w = max(
        20,
        width // 10,
    )

    crop_h = max(
        20,
        height // 10,
    )

    x1 = random.randint(
        0,
        width - crop_w,
    )

    y1 = random.randint(
        0,
        height - crop_h,
    )

    source = result[
        y1:y1 + crop_h,
        x1:x1 + crop_w,
    ].copy()

    x2 = random.randint(
        0,
        width - crop_w,
    )

    y2 = random.randint(
        0,
        height - crop_h,
    )

    result[
        y2:y2 + crop_h,
        x2:x2 + crop_w,
    ] = source

    return result


def region_inpaint(image: np.ndarray) -> np.ndarray:
    """Remove a small image region using OpenCV inpainting."""

    result = image.copy()

    height, width = result.shape[:2]

    if width < 100 or height < 100:
        return result

    region_w = max(
        15,
        width // 12,
    )

    region_h = max(
        15,
        height // 12,
    )

    x = random.randint(
        0,
        width - region_w,
    )

    y = random.randint(
        0,
        height - region_h,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.rectangle(
        mask,
        (x, y),
        (x + region_w, y + region_h),
        255,
        -1,
    )

    result = cv2.inpaint(
        result,
        mask,
        5,
        cv2.INPAINT_TELEA,
    )

    return result


def compression_edit(image: np.ndarray) -> np.ndarray:
    """
    Apply localized JPEG compression and restore it into the image.

    This creates a controlled editing artifact without changing
    the original source dataset.
    """

    result = image.copy()

    height, width = result.shape[:2]

    if width < 100 or height < 100:
        return result

    region_w = max(
        50,
        width // 3,
    )

    region_h = max(
        50,
        height // 3,
    )

    x = random.randint(
        0,
        width - region_w,
    )

    y = random.randint(
        0,
        height - region_h,
    )

    region = result[
        y:y + region_h,
        x:x + region_w,
    ].copy()

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        35,
    ]

    success, encoded = cv2.imencode(
        ".jpg",
        region,
        encode_params,
    )

    if not success:
        return result

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        return result

    decoded = cv2.resize(
        decoded,
        (region_w, region_h),
        interpolation=cv2.INTER_LINEAR,
    )

    result[
        y:y + region_h,
        x:x + region_w,
    ] = decoded

    return result


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():
    """Generate a reproducible normal/tampered document subset."""

    df = load_metadata()

    # Sample from different document groups rather than taking
    # consecutive frames from the same capture sequence.
    df = df.copy()

    df["capture_group"] = (
        df["document_id"]
        .str.rsplit("_", n=1)
        .str[0]
    )

    # IMPORTANT:
    # Keep capture_group as a column because it is required later
    # when creating the tamper metadata manifest.
    grouped = (
        df.groupby(
            "capture_group",
            group_keys=False,
        )
        .first()
        .reset_index()
    )

    sample_size = min(
        NUM_DOCUMENTS,
        len(grouped),
    )

    selected = grouped.sample(
        n=sample_size,
        random_state=SEED,
    ).reset_index(drop=True)

    NORMAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TAMPERED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for index, row in selected.iterrows():

        image_path = (
            PROJECT_ROOT / row["image_path"]
        )

        ground_truth_path = (
            PROJECT_ROOT / row["ground_truth_path"]
        )

        image = read_image(image_path)

        ground_truth = load_ground_truth(
            ground_truth_path
        )

        normal_name = (
            f"{row['document_id']}_normal.png"
        )

        normal_path = (
            NORMAL_DIR / normal_name
        )

        save_image(
            normal_path,
            image,
        )

        fields = find_text_fields(
            ground_truth
        )

        tamper_type = TAMPER_TYPES[
            index % len(TAMPER_TYPES)
        ]

        if tamper_type == "text_substitution":

            if fields:
                field = random.choice(fields)

                tampered = text_substitution(
                    image,
                    field,
                )

                tampered_field = field["name"]

            else:
                tampered = region_copy(image)
                tamper_type = "region_copy"
                tampered_field = ""

        elif tamper_type == "region_copy":

            tampered = region_copy(image)
            tampered_field = ""

        elif tamper_type == "region_inpaint":

            tampered = region_inpaint(image)
            tampered_field = ""

        else:

            tampered = compression_edit(image)
            tampered_field = ""

        tampered_name = (
            f"{row['document_id']}_tampered_{tamper_type}.png"
        )

        tampered_path = (
            TAMPERED_DIR / tampered_name
        )

        save_image(
            tampered_path,
            tampered,
        )

        records.append(
            {
                "document_id": row["document_id"],
                "capture_group": row["capture_group"],
                "document_type": row["document_type"],
                "source_image": row["image_path"],
                "ground_truth_path": row["ground_truth_path"],
                "normal_image": str(
                    normal_path.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "tampered_image": str(
                    tampered_path.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "tamper_type": tamper_type,
                "tampered_field": tampered_field,
                "label": 1,
            }
        )

        print(
            f"[{index + 1:03d}/{sample_size:03d}] "
            f"{row['document_id']} -> {tamper_type}"
        )

    manifest_path = (
        OUTPUT_ROOT / "tamper_metadata.csv"
    )

    pd.DataFrame(records).to_csv(
        manifest_path,
        index=False,
    )

    print()
    print("Tamper dataset generation complete.")
    print(f"Normal images   : {sample_size}")
    print(f"Tampered images : {sample_size}")
    print(f"Manifest        : {manifest_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_dataset()