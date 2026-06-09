#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data/x12/parser_834.py
"""
X12 834 Benefit Enrollment and Maintenance Parser

Parses a raw 834 EDI file and extracts provider enrollment
records into a structured pandas DataFrame.

Key segments extracted:
    INS  — maintenance reason (add / change / terminate)
    REF  — NPI number (qualifier 0F)
    NM1  — provider name
    PRV  — taxonomy code
    N3   — street address
    N4   — city, state, zip
    DTP  — effective date (348) and termination date (349)

Output columns:
    npi, last_name, first_name, taxonomy_code,
    city, state, zip, effective_date, termination_date,
    enrollment_status, maintenance_code, action
"""

import pandas as pd
from pathlib import Path


# Maps 834 maintenance codes to human-readable actions
MAINTENANCE_CODE_MAP = {
    "001": "Addition",
    "002": "Change",
    "024": "Termination",
    "030": "Addition or Change",
}


def parse_834(filepath: str) -> pd.DataFrame:
    """
    Parse an X12 834 EDI file into a provider enrollment DataFrame.

    Args:
        filepath: Path to the .txt 834 EDI file.

    Returns:
        DataFrame with one row per provider enrollment record.
    """
    content = Path(filepath).read_text()

    # 834 segments are delimited by tilde
    segments = [s.strip() for s in content.split("~") if s.strip()]

    providers = []
    current = {}

    for segment in segments:
        elements = segment.split("*")
        seg_id = elements[0]

        # INS — start of a new provider record
        # elements[3] = maintenance type code
        if seg_id == "INS":
            if current.get("npi"):
                providers.append(current)
            maintenance_code = (
                elements[3] if len(elements) > 3 else "Unknown"
            )
            current = {
                "maintenance_code": maintenance_code,
                "action": MAINTENANCE_CODE_MAP.get(
                    maintenance_code, "Unknown"
                ),
            }

        # REF*0F — NPI number (qualifier 0F = NPI)
        elif seg_id == "REF" and len(elements) > 2:
            if elements[1] == "0F":
                current["npi"] = elements[2]

        # NM1 — provider name
        # elements[3] = last name, elements[4] = first name
        elif seg_id == "NM1" and len(elements) > 4:
            current["last_name"] = elements[3]
            current["first_name"] = elements[4]

        # PRV — taxonomy code
        # elements[3] = taxonomy code
        elif seg_id == "PRV" and len(elements) > 3:
            current["taxonomy_code"] = elements[3]

        # N3 — street address
        elif seg_id == "N3" and len(elements) > 1:
            current["street"] = elements[1]

        # N4 — city, state, zip
        elif seg_id == "N4" and len(elements) > 2:
            current["city"] = elements[1]
            current["state"] = elements[2]
            current["zip"] = (
                elements[3] if len(elements) > 3 else None
            )

        # DTP*348 — effective date
        # DTP*349 — termination date
        elif seg_id == "DTP" and len(elements) > 3:
            if elements[1] == "348":
                current["effective_date"] = elements[3]
            elif elements[1] == "349":
                current["termination_date"] = elements[3]

    # Append the last provider record
    if current.get("npi"):
        providers.append(current)

    df = pd.DataFrame(providers)

    # Fill missing termination dates — provider is still active
    if "termination_date" not in df.columns:
        df["termination_date"] = "ACTIVE"
    else:
        df["termination_date"] = df["termination_date"].fillna("ACTIVE")

    # Derive enrollment status from termination date
    df["enrollment_status"] = df["termination_date"].apply(
        lambda x: "ACTIVE" if x == "ACTIVE" else "TERMINATED"
    )

    # Reorder columns for readability
    column_order = [
        "npi", "last_name", "first_name",
        "taxonomy_code", "city", "state", "zip",
        "effective_date", "termination_date",
        "enrollment_status", "maintenance_code", "action",
    ]
    column_order = [c for c in column_order if c in df.columns]

    return df[column_order]


def flag_termination_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag providers whose termination may not have been
    processed correctly in the legacy system.

    A provider is high risk if:
    - Their action is Termination (maintenance code 024)
    - But no termination date was recorded
      (meaning legacy may still show them as active)

    Args:
        df: Output DataFrame from parse_834()

    Returns:
        DataFrame with added termination_risk column.
    """
    df = df.copy()

    df["termination_risk"] = df.apply(
        lambda row: "HIGH - termination may not be processed"
        if row["action"] == "Termination"
        and row["enrollment_status"] == "ACTIVE"
        else "OK",
        axis=1,
    )

    return df


if __name__ == "__main__":
    import sys

    # Default to sample file in same directory
    filepath = Path(__file__).parent / "sample_834.txt"

    if not filepath.exists():
        print(f"Sample file not found at {filepath}")
        print("Please save sample_834.txt in data/x12/ first.")
        sys.exit(1)

    df = parse_834(str(filepath))

    print("=== 834 Enrollment Records ===")
    print(df.to_string())

    print("\n=== Termination Risk Flags ===")
    df_flagged = flag_termination_risk(df)
    print(
        df_flagged[[
            "npi", "last_name", "action",
            "enrollment_status", "termination_risk"
        ]].to_string()
    )


# In[ ]:




