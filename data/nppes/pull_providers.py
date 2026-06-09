#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data/nppes/pull_providers.py

import requests
import pandas as pd

def query_nppes(
    first_name=None,
    last_name=None,
    npi=None,
    enumeration_type="NPI-1",
    limit=10
):
    """
    Query the NPPES NPI registry.
    
    enumeration_type:
        NPI-1 = individual provider
        NPI-2 = organization
    """
    url = "https://npiregistry.cms.hhs.gov/api/"
    
    params = {"version": "2.1", "limit": limit}
    
    if npi:
        params["number"] = npi
    if first_name:
        params["first_name"] = first_name
    if last_name:
        params["last_name"] = last_name
    if enumeration_type:
        params["enumeration_type"] = enumeration_type

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
        return pd.DataFrame()

    # Non-clinical provider types where missing
    # credential is expected and not a risk signal
    NON_CLINICAL_KEYWORDS = [
        "coordinator", "administrator", "manager",
        "technician", "assistant", "aide"
    ]

    records = []
    for p in data["results"]:

        # Primary taxonomy
        taxonomies = p.get("taxonomies", [])
        primary_taxonomy = next(
            (t["desc"] for t in taxonomies if t.get("primary") is True),
            "N/A"
        )
        primary_taxonomy_code = next(
            (t["code"] for t in taxonomies if t.get("primary") is True),
            "N/A"
        )
        all_taxonomy_codes = [t.get("code") for t in taxonomies]

        # Licenses
        licenses = p.get("licenses", [])
        license_states = [l.get("state") for l in licenses if l.get("state")]

        # Practice locations
        addresses = p.get("addresses", [])
        practice_locations = [
            a.get("state") for a in addresses
            if a.get("address_purpose") == "LOCATION"
        ]

        # Credential risk assessment
        credential = p["basic"].get("credential", "MISSING")
        taxonomy_lower = primary_taxonomy.lower()
        is_non_clinical = any(
            kw in taxonomy_lower for kw in NON_CLINICAL_KEYWORDS
        )

        if credential == "MISSING" and not is_non_clinical:
            credential_risk = "HIGH"
        elif credential == "MISSING" and is_non_clinical:
            credential_risk = "LOW - expected for provider type"
        else:
            credential_risk = "OK"

        records.append({
            "npi":                   p["number"],
            "last_name":             p["basic"].get("last_name", "N/A"),
            "first_name":            p["basic"].get("first_name", "N/A"),
            "status":                p["basic"].get("status", "N/A"),
            "credential":            credential,
            "primary_taxonomy":      primary_taxonomy,
            "primary_taxonomy_code": primary_taxonomy_code,
            "all_taxonomy_codes":    all_taxonomy_codes,
            "license_states":        license_states,
            "practice_locations":    practice_locations,
            "num_practice_locations":len(practice_locations),
            "credential_risk":       credential_risk,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Querying NPPES for sample providers...\n")
    df = query_nppes(first_name="John", last_name="Smith", limit=5)
    print(df.to_string())
    print(f"\n{len(df)} providers returned")


# In[ ]:




