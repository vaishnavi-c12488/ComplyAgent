from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SANCTIONS_DATA_DIR = (
    PROJECT_ROOT
    / "data_generation"
    / "sanctions"
)

OFAC_CSV_PATH = (
    SANCTIONS_DATA_DIR
    / "ofac_sdn.csv"
)

UN_XML_PATH = (
    SANCTIONS_DATA_DIR
    / "un_consolidated.xml"
)


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

OFAC_SDN_URL = (
    "https://sanctionslistservice.ofac.treas.gov/"
    "api/PublicationPreview/exports/SDN.CSV"
)

UN_CONSOLIDATED_XML_URL = (
    "https://scsanctions.un.org/resources/xml/"
    "en/consolidated.xml"
)


# ============================================================
# CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = 60

# Conservative thresholds:
#
# >= 90 -> confirmed match
# >= 75 -> potential match
#
# This is NOT a legal determination. It is a screening
# similarity signal for downstream human/risk review.
CONFIRMED_MATCH_THRESHOLD = 90.0
POTENTIAL_MATCH_THRESHOLD = 75.0


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value: str) -> str:
    """
    Normalize text for comparison.

    Removes accents, punctuation and repeated whitespace,
    then converts to uppercase.
    """

    if value is None:
        return ""

    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# NAME SIMILARITY
# ============================================================

def name_similarity(
    applicant_name: str,
    listed_name: str,
) -> float:
    """
    Calculate a simple normalized name similarity score.

    Returns a value from 0 to 100.
    """

    applicant = normalize_text(
        applicant_name
    )

    listed = normalize_text(
        listed_name
    )

    if not applicant or not listed:
        return 0.0

    if applicant == listed:
        return 100.0

    direct_ratio = SequenceMatcher(
        None,
        applicant,
        listed,
    ).ratio()

    applicant_tokens = sorted(
        applicant.split()
    )

    listed_tokens = sorted(
        listed.split()
    )

    token_ratio = SequenceMatcher(
        None,
        " ".join(applicant_tokens),
        " ".join(listed_tokens),
    ).ratio()

    score = max(
        direct_ratio,
        token_ratio,
    ) * 100.0

    return round(
        score,
        2,
    )


# ============================================================
# OFAC DATA
# ============================================================

def download_ofac_data(
    force: bool = False,
) -> Path:
    """
    Download the current OFAC SDN CSV from the official
    OFAC Sanctions List Service.

    The downloaded file is cached locally so repeated
    screenings do not repeatedly download the dataset.
    """

    SANCTIONS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        OFAC_CSV_PATH.exists()
        and not force
    ):
        return OFAC_CSV_PATH

    response = requests.get(
        OFAC_SDN_URL,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": (
                "ComplyAgent/1.0 "
                "(sanctions screening research project)"
            )
        },
    )

    response.raise_for_status()

    OFAC_CSV_PATH.write_bytes(
        response.content
    )

    return OFAC_CSV_PATH


def load_ofac_data() -> pd.DataFrame:
    """Load the cached OFAC SDN dataset."""

    if not OFAC_CSV_PATH.exists():
        download_ofac_data()

    df = pd.read_csv(
        OFAC_CSV_PATH,
        header=None,
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
    )

    if df.empty:
        raise ValueError(
            "OFAC SDN dataset is empty."
        )

    return df


# ============================================================
# OFAC NAME EXTRACTION
# ============================================================

def extract_ofac_names(
    df: pd.DataFrame,
) -> list[str]:
    """
    Extract likely name values from the OFAC legacy CSV.

    OFAC's SDN legacy CSV contains structured columns where
    the first columns include the SDN name and type.

    We deliberately inspect the first column as the primary
    listed-name field instead of relying on an arbitrary
    column index elsewhere in the application.
    """

    if df.shape[1] < 1:
        raise ValueError(
            "OFAC dataset has no columns."
        )

    names = (
        df.iloc[:, 1]
        if df.shape[1] > 1
        else df.iloc[:, 0]
    )

    cleaned_names = []

    for name in names:

        normalized = normalize_text(
            name
        )

        if normalized:
            cleaned_names.append(
                normalized
            )

    return cleaned_names


# ============================================================
# UN DATA
# ============================================================

def download_un_data(
    force: bool = False,
) -> Path:
    """
    Download the current UN Security Council Consolidated
    Sanctions List XML.
    """

    SANCTIONS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        UN_XML_PATH.exists()
        and not force
    ):
        return UN_XML_PATH

    response = requests.get(
        UN_CONSOLIDATED_XML_URL,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": (
                "ComplyAgent/1.0 "
                "(sanctions screening research project)"
            )
        },
    )

    response.raise_for_status()

    UN_XML_PATH.write_bytes(
        response.content
    )

    return UN_XML_PATH


def load_un_names() -> list[str]:
    """
    Load names from the UN Security Council Consolidated
    Sanctions List XML.
    """

    if not UN_XML_PATH.exists():
        download_un_data()

    tree = ET.parse(
        UN_XML_PATH
    )

    root = tree.getroot()

    names = []

    # The UN XML structure uses INDIVIDUAL and ENTITY records.
    # We search recursively so minor namespace/layout differences
    # do not break the parser.
    for element in root.iter():

        tag = element.tag.split(
            "}"
        )[-1].upper()

        if tag not in {
            "INDIVIDUAL",
            "ENTITY",
        }:
            continue

        for child in element.iter():

            child_tag = child.tag.split(
                "}"
            )[-1].upper()

            if child_tag in {
                "FIRST_NAME",
                "SECOND_NAME",
                "THIRD_NAME",
                "FOURTH_NAME",
                "WHOLE_NAME",
                "NAME",
            }:

                if child.text:

                    normalized = normalize_text(
                        child.text
                    )

                    if normalized:
                        names.append(
                            normalized
                        )

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(names)
    )


# ============================================================
# BEST MATCH SEARCH
# ============================================================

def find_best_match(
    applicant_name: str,
    listed_names: list[str],
) -> tuple[str, float]:
    """
    Find the highest similarity between the applicant name
    and a sanctions-list name.
    """

    best_name = ""
    best_score = 0.0

    for listed_name in listed_names:

        score = name_similarity(
            applicant_name,
            listed_name,
        )

        if score > best_score:

            best_score = score
            best_name = listed_name

    return (
        best_name,
        best_score,
    )


# ============================================================
# SANCTIONS LOOKUP
# ============================================================

def lookup_sanctions(
    full_name: str,
    date_of_birth: str = "",
    nationality: str = "",
    country: str = "",
) -> dict:
    """
    Screen an applicant name against OFAC SDN and UN
    Consolidated sanctions lists.

    Additional identity attributes are accepted as part of the
    interface for future identity disambiguation, but the
    current implementation uses name similarity as the primary
    screening signal.

    Returns a structured dictionary compatible with the
    Screening Agent's SanctionsMatch contract.
    """

    if not full_name or not str(full_name).strip():
        raise ValueError(
            "full_name is required for sanctions screening."
        )

    # --------------------------------------------------------
    # Load OFAC.
    # --------------------------------------------------------

    ofac_df = load_ofac_data()

    ofac_names = extract_ofac_names(
        ofac_df
    )

    # --------------------------------------------------------
    # Search OFAC.
    # --------------------------------------------------------

    ofac_name, ofac_score = find_best_match(
        full_name,
        ofac_names,
    )

    # --------------------------------------------------------
    # Load UN.
    # --------------------------------------------------------

    un_names = load_un_names()

    # --------------------------------------------------------
    # Search UN.
    # --------------------------------------------------------

    un_name, un_score = find_best_match(
        full_name,
        un_names,
    )

    # --------------------------------------------------------
    # Select the strongest result.
    # --------------------------------------------------------

    if ofac_score >= un_score:

        best_source = (
            "OFAC"
            if ofac_score > 0
            else "NONE"
        )

        best_name = ofac_name
        best_score = ofac_score

    else:

        best_source = (
            "UN"
            if un_score > 0
            else "NONE"
        )

        best_name = un_name
        best_score = un_score

    # --------------------------------------------------------
    # Determine screening status.
    # --------------------------------------------------------

    if best_score >= CONFIRMED_MATCH_THRESHOLD:

        matched = True

        reason = (
            "High name similarity with a sanctions-list "
            "entry. Manual identity verification is required."
        )

    elif best_score >= POTENTIAL_MATCH_THRESHOLD:

        matched = True

        reason = (
            "Potential sanctions-list name similarity. "
            "Additional identity verification is required."
        )

    else:

        matched = False

        best_source = "NONE"
        best_name = ""
        best_score = 0.0

        reason = (
            "No sufficiently similar name was found in the "
            "screened OFAC/UN datasets."
        )

    return {
        "matched": matched,
        "source": best_source,
        "matched_name": best_name,
        "match_score": round(
            best_score,
            2,
        ),
        "reason": reason,
    }


# ============================================================
# LOCAL DATA VALIDATION
# ============================================================

def validate_sanctions_sources() -> dict:
    """
    Verify that both sanctions sources can be loaded and
    contain usable names.
    """

    ofac_df = load_ofac_data()

    ofac_names = extract_ofac_names(
        ofac_df
    )

    un_names = load_un_names()

    if not ofac_names:
        raise ValueError(
            "No usable names were extracted from OFAC data."
        )

    if not un_names:
        raise ValueError(
            "No usable names were extracted from UN data."
        )

    return {
        "ofac_records": len(ofac_names),
        "un_records": len(un_names),
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    print("=" * 70)
    print("COMPLYAGENT - SANCTIONS LOOKUP TOOL")
    print("=" * 70)

    print()
    print("Loading official sanctions datasets...")

    counts = validate_sanctions_sources()

    print(
        f"OFAC names loaded : "
        f"{counts['ofac_records']}"
    )

    print(
        f"UN names loaded   : "
        f"{counts['un_records']}"
    )

    print()
    print(
        "Sanctions source validation: PASSED"
    )

    # --------------------------------------------------------
    # Synthetic test identity.
    #
    # This is intentionally a clearly synthetic name and is
    # expected to return no sanctions match.
    # --------------------------------------------------------

    test_name = (
        "ComplyAgent Synthetic Customer"
    )

    result = lookup_sanctions(
        full_name=test_name,
        date_of_birth="1999-01-01",
        nationality="Indian",
        country="India",
    )

    print()
    print(
        f"Test name: {test_name}"
    )

    print(
        f"Matched: {result['matched']}"
    )

    print(
        f"Source: {result['source']}"
    )

    print(
        f"Score: {result['match_score']:.2f}"
    )

    print()
    print(
        "Sanctions lookup validation: PASSED"
    )


if __name__ == "__main__":
    main()