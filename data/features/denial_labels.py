# data/features/denial_labels.py
"""
Denial Label Generator

Joins 835 remittance data with 837 claim data and
834 enrollment data to produce the final training
labels for the risk scoring model.

The label is not just "claim denied" — it is specifically
"claim denied due to enrollment data discrepancy."

That distinction is what makes the model predict the
right thing — enrollment risk, not general denial risk.

Output:
    provider_denial_summary — one row per provider
    with denial rate, enrollment denial rate, and
    the binary training label used for model training
"""

import pandas as pd
from pathlib import Path


# Enrollment CARC codes — the ones your model targets
ENROLLMENT_CARC_CODES = {"181", "182", "18", "243"}

# CARC codes mapped to denial categories
CARC_DENIAL_CATEGORIES = {
    # Enrollment denial family — MODEL TARGET
    "181": "enrollment",
    "182": "enrollment",
    "18":  "enrollment",
    "243": "enrollment",

    # Eligibility denial family
    "4":   "eligibility",
    "26":  "eligibility",
    "27":  "eligibility",

    # Authorization denial family
    "15":  "authorization",
    "197": "authorization",

    # Coding denial family
    "16":  "coding",
    "B7":  "coding",

    # Timely filing
    "29":  "timely_filing",
}


def generate_denial_labels(
    remittance_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    enrollment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate provider-level denial labels from 835 data.

    Joins remittance → claims → enrollment to produce
    one row per provider with their denial history.

    Args:
        remittance_df: Output from parse_835()
        claims_df:     Output from parse_837()
        enrollment_df: Output from parse_834()

    Returns:
        Provider-level DataFrame with training labels.
    """
    # Join remittance to claims on claim_id
    # Both DataFrames have rendering_npi so pandas
    # renames them with suffixes — fix with rename
    joined = remittance_df.merge(
        claims_df[[
            "claim_id", "rendering_npi",
            "billed_amount", "cpt_code"
        ]],
        on="claim_id",
        how="left",
        suffixes=("_835", "_837")
    )

    # Fix column name conflict — use remittance version
    # as the authoritative rendering NPI
    joined = joined.rename(
        columns={"rendering_npi_835": "rendering_npi"}
    )

    # Aggregate to provider level
    provider_stats = joined.groupby("rendering_npi").agg(
        total_claims=("claim_id", "count"),
        total_denied=("denial_flag", "sum"),
        enrollment_denials=("enrollment_denial_flag", "sum"),
        total_billed=("billed_amount_835", "sum"),
        total_paid=("paid_amount", "sum"),
    ).reset_index()

    # Compute denial rates
    provider_stats["denial_rate"] = (
        provider_stats["total_denied"] /
        provider_stats["total_claims"]
    ).round(4)

    provider_stats["enrollment_denial_rate"] = (
        provider_stats["enrollment_denials"] /
        provider_stats["total_claims"]
    ).round(4)

    # Binary training label
    # True if provider had ANY enrollment denial
    provider_stats["had_enrollment_denial"] = (
        provider_stats["enrollment_denials"] > 0
    )

    # Join enrollment context
    final = provider_stats.merge(
        enrollment_df[[
            "npi", "enrollment_status",
            "taxonomy_code", "action"
        ]],
        left_on="rendering_npi",
        right_on="npi",
        how="left"
    )

    return final


def print_label_summary(labels_df: pd.DataFrame):
    """Print a readable summary of generated labels."""
    print("=" * 50)
    print("DENIAL LABEL SUMMARY")
    print("=" * 50)
    print(f"Total providers:          {len(labels_df)}")
    print(f"With enrollment denials:  "
          f"{labels_df['had_enrollment_denial'].sum()}")
    print(f"Without denials:          "
          f"{(~labels_df['had_enrollment_denial']).sum()}")
    print(f"\nClass balance:")
    print(f"  Positive (had denial):  "
          f"{labels_df['had_enrollment_denial'].mean():.1%}")
    print(f"  Negative (clean):       "
          f"{(~labels_df['had_enrollment_denial']).mean():.1%}")
    print("\nProvider detail:")
    print(labels_df[[
        "rendering_npi", "total_claims",
        "enrollment_denials", "enrollment_denial_rate",
        "had_enrollment_denial", "enrollment_status"
    ]].to_string())


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "x12"))

    from parser_835 import parse_835
    from parser_837 import parse_837
    from parser_834 import parse_834

    remittance_df = parse_835("data/x12/sample_835.txt")
    claims_df     = parse_837("data/x12/sample_837.txt")
    enrollment_df = parse_834("data/x12/sample_834.txt")

    labels_df = generate_denial_labels(
        remittance_df, claims_df, enrollment_df
    )

    print_label_summary(labels_df)