#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data/x12/parser_835.py
"""
X12 835 Health Care Claim Payment/Advice Parser

Parses a raw 835 remittance file and extracts claim
payment and denial information into a structured
pandas DataFrame.

Key segments extracted:
    CLP  — claim payment status and amounts
    NM1  — patient and rendering provider identity
    SVC  — service line procedure code
    DTM  — date of service
    CAS  — adjustment reason codes (CARC codes)
           THIS is the denial reason — the training label source

Output columns:
    claim_id, rendering_npi, patient_id,
    billed_amount, paid_amount, claim_status,
    date_of_service, cpt_code, carc_code,
    adjustment_amount, denial_flag,
    denial_category, enrollment_denial_flag

The enrollment_denial_flag is the target variable
for the risk scoring model.
"""

import pandas as pd
from pathlib import Path


# Claims status codes in CLP segment
CLAIM_STATUS_MAP = {
    "1": "Paid",
    "2": "Partial",
    "3": "Acknowledged",
    "4": "Denied",
    "19": "Pending",
    "22": "Reversed",
}

# CARC codes mapped to denial categories
# Only enrollment denial family is the model target
CARC_DENIAL_CATEGORIES = {
    # Enrollment denial family — MODEL TARGET
    "181": "enrollment",   # provider not on file
    "182": "enrollment",   # inconsistent with provider type
    "18":  "enrollment",   # duplicate of original bill
    "243": "enrollment",   # not in network

    # Eligibility denial family — NOT model target
    "4":   "eligibility",  # not covered by plan
    "26":  "eligibility",  # expenses incurred prior to coverage
    "27":  "eligibility",  # expenses incurred after coverage

    # Authorization denial family — NOT model target
    "15":  "authorization", # authorization required
    "197": "authorization", # prior authorization absent

    # Coding denial family — NOT model target
    "4":   "coding",       # not covered
    "16":  "coding",       # claim lacks info
    "B7":  "coding",       # not covered by this payer

    # Timely filing — NOT model target
    "29":  "timely_filing", # claim submitted too late
}

# Enrollment CARC codes — the ones your model targets
ENROLLMENT_CARC_CODES = {"181", "182", "18", "243"}


def parse_835(filepath: str) -> pd.DataFrame:
    """
    Parse an X12 835 remittance file into a DataFrame.

    Args:
        filepath: Path to the .txt 835 EDI file.

    Returns:
        DataFrame with one row per claim.
        enrollment_denial_flag = True is the model
        training label.
    """
    content = Path(filepath).read_text()
    segments = [s.strip() for s in content.split("~") if s.strip()]

    claims = []
    current = {}

    for segment in segments:
        elements = segment.split("*")
        seg_id = elements[0]

        # CLP — claim payment information
        # Start of a new claim record
        # elements[1] = claim ID
        # elements[2] = claim status (4 = denied)
        # elements[3] = billed amount
        # elements[4] = paid amount
        if seg_id == "CLP":
            if current.get("claim_id"):
                claims.append(current.copy())
            current = {
                "claim_id":     elements[1],
                "claim_status": CLAIM_STATUS_MAP.get(
                    elements[2], elements[2]
                ),
                "billed_amount": float(elements[3])
                    if len(elements) > 3 else 0.0,
                "paid_amount":  float(elements[4])
                    if len(elements) > 4 else 0.0,
                "denial_flag":  elements[2] == "4",
            }

        # NM1*QC — patient information
        # NM1*82 — rendering provider NPI
        elif seg_id == "NM1":
            qualifier = elements[1] if len(elements) > 1 else ""
            if qualifier == "QC" and len(elements) > 9:
                current["patient_id"] = elements[9]
            elif qualifier == "82" and len(elements) > 9:
                current["rendering_npi"] = elements[9]

        # SVC — service line
        # elements[1] = procedure code (HC:99213)
        elif seg_id == "SVC" and len(elements) > 1:
            svc = elements[1].split(":")
            if len(svc) > 1:
                current["cpt_code"] = svc[1]

        # DTM*472 — date of service
        elif seg_id == "DTM" and len(elements) > 2:
            if elements[1] == "472":
                current["date_of_service"] = elements[2]

        # CAS — adjustment reason
        # elements[1] = group code (CO/PR/OA)
        # elements[2] = CARC code
        # elements[3] = adjustment amount
        elif seg_id == "CAS" and len(elements) > 2:
            carc_code = elements[2]
            current["carc_code"] = carc_code
            current["adjustment_amount"] = float(elements[3]) \
                if len(elements) > 3 else 0.0
            current["denial_category"] = CARC_DENIAL_CATEGORIES.get(
                carc_code, "other"
            )

    # Append last claim
    if current.get("claim_id"):
        claims.append(current)

    df = pd.DataFrame(claims)

    # Generate the model training label
    # enrollment_denial_flag = True means this claim
    # was denied due to an enrollment data problem
    # This is what the risk model learns to predict
    if "carc_code" in df.columns:
        df["enrollment_denial_flag"] = df["carc_code"].isin(
            ENROLLMENT_CARC_CODES
        )
    else:
        df["enrollment_denial_flag"] = False

    # Reorder columns
    column_order = [
        "claim_id", "rendering_npi", "patient_id",
        "claim_status", "billed_amount", "paid_amount",
        "date_of_service", "cpt_code", "carc_code",
        "adjustment_amount", "denial_flag",
        "denial_category", "enrollment_denial_flag",
    ]
    column_order = [c for c in column_order if c in df.columns]

    return df[column_order]


if __name__ == "__main__":
    import sys

    filepath = Path("data/x12/sample_835.txt")

    if not filepath.exists():
        print(f"Sample file not found at {filepath}")
        sys.exit(1)

    df = parse_835(str(filepath))

    print("=== 835 Remittance Records ===")
    print(df.to_string())

    print("\n=== Enrollment Denial Summary ===")
    print(f"Total claims:            {len(df)}")
    print(f"Total denied:            {df['denial_flag'].sum()}")
    print(f"Enrollment denials:      {df['enrollment_denial_flag'].sum()}")
    print(f"Non-enrollment denials:  "
          f"{df['denial_flag'].sum() - df['enrollment_denial_flag'].sum()}")

    print("\n=== Denial Category Breakdown ===")
    print(df.groupby("denial_category")["claim_id"].count())

