# Healthcare Claims Denial Risk — Prediction & Explanation Engine

> Predict the probability that a Medicaid claim will be denied before submission, and generate a plain-language explanation of the top risk factors so operations teams can remediate proactively.

---

## The Problem

In Medicaid managed care, provider enrollment data lives in one system and claims are processed in another. When those systems fall out of sync — mismatched eligibility dates, incorrect provider type, missing identifiers — valid claims get denied. The denial surfaces days or weeks after the fact, triggering manual review, delayed provider payments, and downstream operational burden.

This project moves that detection upstream. Instead of reacting to denials after adjudication, it scores provider-claim combinations **before submission** and explains the specific data discrepancies most likely to cause a denial.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
│  Synthea FHIR R4  →  Provider Features (NPPES/NPI)         │
│  X12 837 Claims   →  Enrollment Features (834)             │
│  X12 835 Remit    →  Denial Labels (CARC/RARC)             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Feature Store                            │
│  Provider features · Coverage features · Claims windows    │
│  ICD-10/CPT rollups · Enrollment mismatch flags            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Prediction Model                           │
│  Gradient-boosted classifier · Calibrated probabilities    │
│  PR-AUC · Calibration curve · Fairness evaluation          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               GenAI Explanation Layer                       │
│  SHAP attribution → RAG/LLM → Plain-language denial risk   │
│  "Missing prior auth for CPT X · Eligibility date gap"     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Serving Layer                             │
│               FastAPI · Dockerized                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Healthcare Data Standards

This project is built on the actual standards that govern U.S. healthcare claims processing:

| Standard | Role in This Project |
|----------|----------------------|
| **X12 837** | Claim submission — source of claim features |
| **X12 835** | Remittance advice — source of denial labels via CARC/RARC codes |
| **X12 834** | Benefit enrollment — source of provider enrollment features |
| **NPI / NPPES** | Provider identity and taxonomy — provider feature table |
| **FHIR R4** | Synthetic patient and coverage data via Synthea |
| **ICD-10 / CPT** | Procedure and diagnosis code rollups as model features |
| **CARC / RARC** | Claim Adjustment Reason Codes — denial label taxonomy |

---

## Project Structure

```
healthcare-claims-ai/
│
├── data/
│   ├── nppes/              # NPI/NPPES provider feature pipeline
│   ├── synthea/            # FHIR R4 synthetic patient generation
│   ├── x12/                # 837/835/834 parsers
│   └── features/           # Feature store schema and assembly
│
├── models/
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # PR-AUC, calibration, fairness evaluation
│   └── model_card.md       # Model card — intended use, metrics, limitations
│
├── explain/
│   └── denial_explainer.py # SHAP → RAG/LLM explanation layer
│
├── api/
│   └── main.py             # FastAPI serving endpoint
│
├── notebooks/
│   └── eda/                # Exploratory analysis notebooks
│
├── tests/                  # Unit and integration tests
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Quickstart

```bash
# Clone and set up environment
git clone https://github.com/alan8320/healthcare-claims-ai.git
cd healthcare-claims-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Pull provider features from NPPES API
python data/nppes/pull_providers.py

# Generate synthetic FHIR R4 patients (requires Synthea)
# See data/synthea/README.md for setup instructions

# Run the full feature pipeline
make features

# Train and evaluate the model
make train
make evaluate
```

---

## Evaluation Approach

Generic ML evaluation metrics are insufficient for healthcare. This project uses:

- **PR-AUC** over ROC-AUC — denial events are rare; precision-recall tradeoff is what matters operationally
- **Calibration curve + Brier score** — probability scores drive prioritization decisions, so calibration is non-negotiable
- **Subgroup fairness evaluation** — denial risk scores are audited across provider type, specialty, and geography
- **Decision-curve analysis** — net benefit across probability thresholds tied to operational cost of review vs. missed denial
- **Leakage audit** — explicit checks for post-outcome codes and time-travel in claims features

---

## Model Card

See [`models/model_card.md`](models/model_card.md) for full documentation including intended use, data sources, known limitations, and responsible use guidance.

**Intended use:** Proactive identification of provider-claim combinations at high risk of denial due to enrollment data discrepancies. For operational triage only — not a clinical decision tool.

**Data:** Synthetic data generated via Synthea (FHIR R4) and CMS DE-SynPUF. Denial labels are derived from CARC/RARC logic applied to synthetic remittance data. No real patient or provider PHI is used anywhere in this project.

**Known limitations:** Synthetic denial labels approximate real-world denial patterns but cannot fully replicate payer-specific adjudication rules. Model performance on a specific payer's claims requires validation against that payer's actual remittance data.

---

## Build Status

| Component | Status |
|-----------|--------|
| NPPES provider feature pipeline | ✅ Complete |
| X12 834 enrollment parser | ✅ Complete |
| X12 837/835 claim-denial parser | 🔜 Upcoming |
| Synthea FHIR R4 pipeline | 🔜 Upcoming |
| Feature store assembly | 🔜 Upcoming |
| Prediction model + evaluation | 🔜 Upcoming |
| GenAI explanation layer | 🔜 Upcoming |
| FastAPI + Docker serving | 🔜 Upcoming |

---

## Background

This project is informed by production ML work in Medicaid provider enrollment data quality at a state Medicaid agency, where enrollment discrepancies between modern enrollment systems and legacy claims systems are a primary driver of claim denials. All code and data in this repository uses synthetic or publicly available data only.

---

## Tech Stack

Python · Pandas · XGBoost · SHAP · FastAPI · Docker · FHIR R4 (Synthea) · X12 EDI · NPPES API

---

## Author

**Alan David** — ML Engineer, Healthcare AI  
[LinkedIn](https://linkedin.com/in/alan-david) · [Portfolio](https://alan8320.github.io/Personal-Portfolio/)
