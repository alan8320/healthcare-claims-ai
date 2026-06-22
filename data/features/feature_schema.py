#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data/features/feature_schema.py
"""
Feature Store Schema — Provider Enrollment Risk Model

Defines the exact input features the risk scoring model
trains on. Every feature maps to a specific stage in the
provider enrollment lifecycle where a discrepancy can cause
a Medicaid claim denial.

Feature families:
    1. Provider identity features   (from NPPES)
    2. Enrollment status features   (from 834 parser)
    3. Discrepancy features         (from cross-source comparison)
    4. Temporal features            (derived from dates)

Denial category each feature predicts:
    Enrollment denial  — provider data problem
    Eligibility denial — patient coverage problem (out of scope)
    Coding denial      — procedure code mismatch
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from datetime import date


# ── Feature definitions ───────────────────────────────────────

@dataclass
class FeatureDefinition:
    """
    Defines a single feature in the feature store.

    Attributes:
        name:        Column name in the feature DataFrame
        dtype:       Python type (str, int, float, bool)
        nullable:    Whether None/NaN is a valid value
        description: Plain English explanation of what this
                     feature measures
        denial_type: Which denial category this feature
                     predicts (enrollment / coding / auth)
        lifecycle_stage: Which enrollment stage this maps to
    """
    name: str
    dtype: type
    nullable: bool
    description: str
    denial_type: str
    lifecycle_stage: str


# ── Feature registry ──────────────────────────────────────────

FEATURE_REGISTRY = [

    # ── Provider identity features (from NPPES) ───────────────

    FeatureDefinition(
        name="nppes_missing_flag",
        dtype=bool,
        nullable=False,
        description=(
            "True if the provider NPI exists in the enrollment "
            "system but returns no result from NPPES. Indicates "
            "the provider may be deactivated, OIG excluded, or "
            "the NPI was entered incorrectly."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 2 - Verification",
    ),

    FeatureDefinition(
        name="oig_exclusion_flag",
        dtype=bool,
        nullable=False,
        description=(
            "True if the provider NPI appears on the OIG "
            "exclusion list. Any Medicaid payment to an excluded "
            "provider is a federal compliance violation regardless "
            "of clinical validity. CRITICAL risk tier."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 2 - Verification",
    ),

    FeatureDefinition(
        name="credential_risk_flag",
        dtype=str,
        nullable=False,
        description=(
            "Risk level based on credential presence and provider "
            "type. HIGH if a clinical provider (MD, DO, NP, PA) "
            "has a missing credential. LOW if a non-clinical "
            "provider (care coordinator, administrator) has a "
            "missing credential — expected for that type."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 4 - Active Enrollment",
    ),

    FeatureDefinition(
        name="practice_location_count",
        dtype=int,
        nullable=False,
        description=(
            "Number of practice locations associated with this "
            "provider NPI. Providers with multiple locations have "
            "higher risk of location-specific enrollment gaps — "
            "active at one location but not the location where "
            "the service was rendered."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 4 - Active Enrollment",
    ),

    # ── Enrollment status features (from 834 parser) ──────────

    FeatureDefinition(
        name="termination_status_risk",
        dtype=str,
        nullable=False,
        description=(
            "Risk level based on termination date vs enrollment "
            "status. HIGH if termination date has passed but "
            "status shows active (ghost provider). MEDIUM if "
            "termination date is within 60 days (upcoming). "
            "LOW if active with no termination date on record."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 7 - Termination",
    ),

    FeatureDefinition(
        name="days_since_termination",
        dtype=float,
        nullable=True,
        description=(
            "Number of days between the termination date and "
            "today. Only populated if termination_date exists. "
            "Longer gap indicates higher likelihood that the "
            "termination was never reconciled in legacy system."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 7 - Termination",
    ),

    # ── Discrepancy features (cross-source comparison) ────────

    FeatureDefinition(
        name="taxonomy_mismatch_flag",
        dtype=bool,
        nullable=False,
        description=(
            "True if the primary taxonomy code in NPPES differs "
            "from the taxonomy code in the 834 enrollment record. "
            "Silent discrepancy — both systems have a valid code "
            "but they disagree. Causes enrollment denial at "
            "adjudication when claim taxonomy does not match "
            "payer enrollment record."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 6 - Change Event",
    ),

    # ── Temporal features (derived from dates) ────────────────

    FeatureDefinition(
        name="days_since_revalidation",
        dtype=float,
        nullable=True,
        description=(
            "Number of days since the provider last completed "
            "revalidation. Current policy threshold is 3 years "
            "(1095 days). Providers exceeding threshold without "
            "active revalidation are at risk of deactivation."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 5 - Revalidation",
    ),

    FeatureDefinition(
        name="within_grace_period_flag",
        dtype=bool,
        nullable=False,
        description=(
            "True if the provider is overdue for revalidation "
            "but within the policy grace period (90 days). "
            "Requires different ops action than a genuine "
            "high-risk enrollment discrepancy — notify provider "
            "to revalidate rather than investigate for fraud."
        ),
        denial_type="enrollment",
        lifecycle_stage="Stage 5 - Revalidation",
    ),

]


# ── Risk flag functions ───────────────────────────────────────

REVALIDATION_THRESHOLD_DAYS = 1095   # 3 years current policy
GRACE_PERIOD_DAYS           = 90     # 90 day grace window
UPCOMING_TERMINATION_DAYS   = 60     # flag terminations within 60 days


def compute_termination_status_risk(
    enrollment_status: str,
    termination_date: Optional[str],
    today: date = None,
) -> str:
    """
    Compute termination status risk level.

    Args:
        enrollment_status: 'ACTIVE' or 'TERMINATED' from 834
        termination_date:  'ACTIVE' or date string YYYYMMDD
        today:             date to compare against (default: today)

    Returns:
        Risk level string: 'HIGH' | 'MEDIUM' | 'LOW'
    """
    if today is None:
        today = date.today()

    if termination_date == "ACTIVE" or termination_date is None:
        return "LOW"

    term_date = date(
        int(termination_date[:4]),
        int(termination_date[4:6]),
        int(termination_date[6:8]),
    )
    days_until = (term_date - today).days

    if enrollment_status == "ACTIVE" and term_date < today:
        # Termination date passed but still showing active
        # Ghost provider scenario
        return "HIGH"
    elif days_until <= UPCOMING_TERMINATION_DAYS:
        # Termination coming soon — warn ops team
        return "MEDIUM"
    else:
        return "LOW"


def compute_days_since_termination(
    termination_date: Optional[str],
    today: date = None,
) -> Optional[float]:
    """
    Compute days since termination date.

    Returns None if provider has no termination date.
    """
    if today is None:
        today = date.today()

    if termination_date == "ACTIVE" or termination_date is None:
        return None

    term_date = date(
        int(termination_date[:4]),
        int(termination_date[4:6]),
        int(termination_date[6:8]),
    )
    return float((today - term_date).days)


def compute_credential_risk(
    credential: str,
    primary_taxonomy: str,
) -> str:
    """
    Compute credential risk level based on provider type.

    Missing credential is HIGH risk for clinical providers.
    Missing credential is LOW risk for non-clinical providers.
    """
    NON_CLINICAL_KEYWORDS = [
        "coordinator", "administrator", "manager",
        "technician", "assistant", "aide"
    ]

    if credential != "MISSING":
        return "OK"

    taxonomy_lower = primary_taxonomy.lower()
    is_non_clinical = any(
        kw in taxonomy_lower for kw in NON_CLINICAL_KEYWORDS
    )

    return "LOW" if is_non_clinical else "HIGH"


def compute_revalidation_risk(
    days_since_revalidation: Optional[float],
) -> dict:
    """
    Compute revalidation-related risk flags.

    Returns dict with:
        days_since_revalidation: float or None
        within_grace_period_flag: bool
    """
    if days_since_revalidation is None:
        return {
            "days_since_revalidation": None,
            "within_grace_period_flag": False,
        }

    overdue = days_since_revalidation > REVALIDATION_THRESHOLD_DAYS
    in_grace = (
        overdue and
        days_since_revalidation <=
        REVALIDATION_THRESHOLD_DAYS + GRACE_PERIOD_DAYS
    )

    return {
        "days_since_revalidation": days_since_revalidation,
        "within_grace_period_flag": in_grace,
    }


# ── Schema validation ─────────────────────────────────────────

def validate_feature_row(row: dict) -> list:
    """
    Validate a single feature row against the schema.

    Returns list of validation errors.
    Empty list means the row is valid.
    """
    errors = []
    feature_names = {f.name for f in FEATURE_REGISTRY}

    for feature in FEATURE_REGISTRY:
        if feature.name not in row:
            if not feature.nullable:
                errors.append(
                    f"Missing required feature: {feature.name}"
                )
            continue

        value = row[feature.name]

        if value is None and not feature.nullable:
            errors.append(
                f"Null value in non-nullable feature: {feature.name}"
            )

    return errors


# ── Schema summary ────────────────────────────────────────────

def print_schema_summary():
    """Print a readable summary of all features."""
    print("=" * 60)
    print("PROVIDER ENROLLMENT RISK — FEATURE SCHEMA")
    print("=" * 60)

    for f in FEATURE_REGISTRY:
        print(f"\n{f.name}")
        print(f"  Type:      {f.dtype.__name__}")
        print(f"  Nullable:  {f.nullable}")
        print(f"  Stage:     {f.lifecycle_stage}")
        print(f"  Denial:    {f.denial_type}")
        print(f"  Meaning:   {f.description[:60]}...")

    print(f"\nTotal features: {len(FEATURE_REGISTRY)}")


if __name__ == "__main__":
    print_schema_summary()

    # Quick smoke test of risk functions
    print("\n" + "=" * 60)
    print("SMOKE TEST — Risk flag functions")
    print("=" * 60)

    # Ghost provider scenario — terminated 3 years ago
    # but still showing active
    risk = compute_termination_status_risk(
        enrollment_status="ACTIVE",
        termination_date="20230601",
    )
    print(f"\nGhost provider risk:        {risk}")

    # Upcoming termination
    risk = compute_termination_status_risk(
        enrollment_status="ACTIVE",
        termination_date="20260701",
    )
    print(f"Upcoming termination risk:  {risk}")

    # Days since termination
    days = compute_days_since_termination("20230601")
    print(f"Days since termination:     {days}")

    # Credential risk — clinical provider missing credential
    risk = compute_credential_risk("MISSING", "Internal Medicine")
    print(f"Clinical missing cred:      {risk}")

    # Credential risk — non-clinical missing credential
    risk = compute_credential_risk("MISSING", "Care Coordinator")
    print(f"Non-clinical missing cred:  {risk}")

    # Revalidation — overdue but in grace period
    reval = compute_revalidation_risk(1100.0)
    print(f"Revalidation overdue:       {reval}")


# In[ ]:




