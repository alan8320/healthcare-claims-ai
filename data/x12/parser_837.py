# data/x12/parser_837.py
"""
X12 837P Professional Claim Parser

Parses a raw 837P EDI file and extracts claim records
into a structured pandas DataFrame joinable to the
provider enrollment feature table from the 834 parser.

Key segments extracted:
    NM1*41  — start of new claim block (billing provider)
    NM1*85  — billing provider NPI
    NM1*82  — rendering provider NPI (the individual doctor)
    NM1*IL  — patient member ID
    CLM     — claim ID and billed amount
    DTP*472 — date of service
    HI      — ICD-10 diagnosis code
    SV1     — CPT procedure code and billed units

Parser design note:
    In the 837P standard NM1 segments appear BEFORE
    the CLM segment that starts the claim record.
    NM1*41 (billing provider) marks the start of each
    new claim block. The parser resets on NM1*41 to
    prevent provider context bleeding across claims.

Output columns:
    claim_id, rendering_npi, billing_npi, patient_id,
    billed_amount, date_of_service, icd10_code,
    cpt_code, billed_units, place_of_service
"""

import pandas as pd
from pathlib import Path


def parse_837(filepath: str) -> pd.DataFrame:
    """
    Parse an X12 837P EDI file into a claims DataFrame.

    Args:
        filepath: Path to the .txt 837P EDI file.

    Returns:
        DataFrame with one row per claim.
        Joinable to enrollment features on rendering_npi.
    """
    content = Path(filepath).read_text()
    segments = [s.strip() for s in content.split("~") if s.strip()]

    claims = []
    current = {}

    for segment in segments:
        elements = segment.split("*")
        seg_id = elements[0]

        if seg_id == "NM1":
            qualifier = elements[1] if len(elements) > 1 else ""

            # NM1*41 = billing provider = start of new claim block
            # Save previous claim if exists and reset current
            if qualifier == "41":
                if current.get("claim_id"):
                    claims.append(current.copy())
                current = {}

            # NM1*85 — billing provider NPI
            elif qualifier == "85" and len(elements) > 9:
                current["billing_npi"] = elements[9]

            # NM1*82 — rendering provider NPI
            # This is the individual doctor who performed service
            # Key join field to enrollment data
            elif qualifier == "82" and len(elements) > 9:
                current["rendering_npi"]   = elements[9]
                current["rendering_last"]  = elements[3]
                current["rendering_first"] = elements[4]

            # NM1*IL — patient (insured) member ID
            elif qualifier == "IL" and len(elements) > 9:
                current["patient_id"]    = elements[9]
                current["patient_last"]  = elements[3]
                current["patient_first"] = elements[4]

        # CLM — claim information
        # elements[1] = claim ID
        # elements[2] = billed amount
        # elements[5] = place of service code
        elif seg_id == "CLM" and len(elements) > 2:
            current["claim_id"]      = elements[1]
            current["billed_amount"] = float(elements[2]) \
                if elements[2] else 0.0
            if len(elements) > 5:
                pos = elements[5].split(":")
                current["place_of_service"] = pos[0] if pos else None

        # DTP*472 — date of service
        elif seg_id == "DTP" and len(elements) > 3:
            if elements[1] == "472":
                current["date_of_service"] = elements[3]

        # HI — diagnosis codes (ICD-10)
        # Format: ABK:Z8711 → qualifier:code
        elif seg_id == "HI" and len(elements) > 1:
            diag = elements[1].split(":")
            if len(diag) > 1:
                current["icd10_code"] = diag[1]

        # SV1 — service line (procedure code + units)
        # Format: HC:99213 → qualifier:CPT code
        elif seg_id == "SV1" and len(elements) > 1:
            svc = elements[1].split(":")
            if len(svc) > 1:
                current["cpt_code"] = svc[1]
            if len(elements) > 3:
                current["billed_units"] = elements[3]

    # Append the last claim
    if current.get("claim_id"):
        claims.append(current)

    df = pd.DataFrame(claims)

    # Reorder columns for readability
    column_order = [
        "claim_id", "rendering_npi", "rendering_last",
        "rendering_first", "billing_npi", "patient_id",
        "patient_last", "patient_first", "billed_amount",
        "date_of_service", "icd10_code", "cpt_code",
        "billed_units", "place_of_service",
    ]
    column_order = [c for c in column_order if c in df.columns]

    return df[column_order]


def join_claims_to_enrollment(
    claims_df: pd.DataFrame,
    enrollment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join claims to enrollment data on rendering provider NPI.

    This join is where enrollment discrepancies meet claims.
    A provider with enrollment issues whose claims appear here
    is at high risk of denial at adjudication.

    Args:
        claims_df:     Output from parse_837()
        enrollment_df: Output from parse_834()

    Returns:
        Joined DataFrame with enrollment context per claim.
    """
    joined = claims_df.merge(
        enrollment_df[[
            "npi", "enrollment_status",
            "termination_date", "taxonomy_code",
            "action", "effective_date"
        ]],
        left_on="rendering_npi",
        right_on="npi",
        how="left"
    )

    # Flag claims where provider was terminated
    # on or before the date of service
    joined["provider_terminated_at_service"] = joined.apply(
        lambda row: _check_terminated_at_service(
            row["enrollment_status"],
            row["termination_date"],
            row["date_of_service"]
        ),
        axis=1
    )

    return joined


def _check_terminated_at_service(
    enrollment_status: str,
    termination_date: str,
    date_of_service: str,
) -> str:
    """
    Check if provider was terminated on the date of service.

    If a provider was terminated before the service date
    the claim will be denied at adjudication.
    """
    if termination_date == "ACTIVE" or termination_date is None:
        return "OK"

    if date_of_service is None:
        return "UNKNOWN"

    try:
        term = int(str(termination_date).replace("-", ""))
        svc  = int(str(date_of_service).replace("-", ""))

        if svc > term:
            return "HIGH - service after termination date"
        elif svc == term:
            return "MEDIUM - service on termination date"
        else:
            return "OK"
    except (ValueError, AttributeError):
        return "UNKNOWN"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    claims_path = Path("data/x12/sample_837.txt")

    if not claims_path.exists():
        print(f"Sample file not found at {claims_path}")
        sys.exit(1)

    claims_df = parse_837(str(claims_path))

    print("=== 837 Claims ===")
    print(claims_df.to_string())

    enrollment_path = Path("data/x12/sample_834.txt")

    if enrollment_path.exists():
        sys.path.append(str(Path(__file__).parent))
        from parser_834 import parse_834

        enrollment_df = parse_834(str(enrollment_path))

        print("\n=== Claims + Enrollment Join ===")
        joined = join_claims_to_enrollment(claims_df, enrollment_df)
        print(joined[[
            "claim_id", "rendering_npi", "rendering_last",
            "date_of_service", "cpt_code", "billed_amount",
            "enrollment_status", "termination_date",
            "provider_terminated_at_service"
        ]].to_string())