# Aeromedical Ground-Time Analysis

Official code-and-results companion repository for:

**Bruno Santos Luiz, Lucas Orbolato Carvalho, and Daniel Alberto Pamplona (2026).**  
*Probabilistic Modeling of Critical Ground Times in Aeromedical Organ Transport: A Brazilian Air Force Case Study.*  
**Sci, 8(7), 169.** https://doi.org/10.3390/sci8070169

## Overview

This repository supports the analysis reported in the paper above. The study develops and validates a mission-level probabilistic framework for **critical ground time** in Brazilian Air Force (FAB) Team, Tissue, and Organ Transport (TOTEQ) missions.

The published analysis:

- consolidates repeated organ records to one observation per mission to avoid pseudoreplication;
- compares Poisson, Negative Binomial, Gamma, lognormal, Weibull, and log-logistic candidate models;
- treats rounded five-minute observations with grouped-likelihood duration modeling;
- estimates P50 and P95 planning benchmarks with refitted parametric bootstrap uncertainty;
- evaluates threshold-exceedance probabilities; and
- performs chronological validation using 2023–2024 missions for fitting and January–May 2025 missions as a holdout.

The study analyzed **583 mission-level observations** derived from **712 organ-record rows**. The fitted P95 benchmarks were approximately **155 min for unaccompanied transport, 380 min for simple procurement, and 375 min for multiple procurement**.

## Repository status

This repository contains the **published analysis script and aggregate supplementary output tables** associated with the article. The original supplementary package is also available from MDPI:

https://www.mdpi.com/article/10.3390/sci8070169/s1

An exact copy of the published MDPI supplementary ZIP is preserved at `archive/sci-4311883-supplementary.zip`. The root analysis script and the CSV tables under `analysis_outputs/` are extracted directly from that archive. `SHA256SUMS.txt` records and verifies the archive and extracted-file hashes.

The underlying mission-level records are **not public**. They are maintained by the Brazilian Air Force (FAB) / Aerospace Operations Command (COMAE) and are subject to operational-security restrictions. Consequently, full reproduction from the raw source data requires institutional data access.

## Published supplementary files

The article identifies the following supporting files:

| Item | File | Description |
|---|---|---|
| File S1 | `toteq_ground_time_analysis.py` | Main analysis script: mission-level data construction, grouped-likelihood model fitting, refitted parametric bootstrap, temporal screening, and chronological validation |
| Table S1 | `audit_counts.csv` | Record-to-mission audit counts |
| Table S2 | `mission_descriptive_by_modality.csv` | Mission-level descriptive statistics by procurement modality |
| Table S3 | `mission_descriptive_by_organ_profile.csv` | Organ-profile summaries |
| Table S4 | `model_comparison_grouped_likelihood.csv` | Grouped-likelihood model comparison |
| Table S5 | `bootstrap_percentile_summary.csv` | Bootstrap percentile summaries |
| Table S6 | `threshold_sensitivity.csv` | Threshold-exceedance probabilities |
| Table S7 | `quarterly_temporal_summary.csv` | Quarterly temporal summaries |
| Table S8 | `temporal_trend_screen.csv` | Spearman temporal-trend screening |
| Table S9 | `chronological_validation_2025.csv` | Chronological validation results |

The aggregate CSV files are kept under `analysis_outputs/`, matching the directory used by the published script.

## Software environment

The published analysis was conducted in **Python 3.12** using the following open-source packages:

- pandas
- NumPy
- SciPy
- Matplotlib

A minimal `requirements.txt` is included for convenience. Exact behavior may also depend on the package versions used in the published analysis; the archived MDPI supplementary script remains the authoritative published code artifact.

## Running the analysis

When the restricted FAB/COMAE source extract is available through the appropriate institutional channel, the published script can be executed in a Python 3.12 environment after installing the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
python toteq_ground_time_analysis.py
```

Because the raw mission data are not distributed here, users without institutional access can inspect the code and the published aggregate results, but cannot fully regenerate the mission-level dataset from this repository alone.

## Interpretation and scope

The reported percentiles are **risk-based planning benchmarks**, not guarantees of mission duration or safety. The paper does not claim that a percentile threshold by itself establishes a safety outcome or prescribes a regulatory change. Operational use requires specialist judgment and the relevant institutional context.

## Authors and software contributions

Paper authors:

- **Bruno Santos Luiz**
- **Lucas Orbolato Carvalho**
- **Daniel Alberto Pamplona**

The published author-contribution statement credits **software, methodology, formal analysis, and data curation** to Bruno Santos Luiz and Lucas Orbolato Carvalho, with validation by all three authors.

## Citation

If this repository supports your work, please cite the article:

```bibtex
@article{Luiz2026AeromedicalGroundTime,
  author  = {Luiz, Bruno Santos and Carvalho, Lucas Orbolato and Pamplona, Daniel Alberto},
  title   = {Probabilistic Modeling of Critical Ground Times in Aeromedical Organ Transport: A Brazilian Air Force Case Study},
  journal = {Sci},
  year    = {2026},
  volume  = {8},
  number  = {7},
  pages   = {169},
  doi     = {10.3390/sci8070169}
}
```

Machine-readable citation metadata are also provided in `CITATION.cff`.

## Data availability

The analysis code and aggregate output tables are public supporting materials. The underlying mission-level data are maintained by **FAB/COMAE** and are subject to operational-security restrictions. Requests for access must follow the applicable institutional protocols.

## License

No separate software license is assigned in this repository at present. The journal article is published open access under the license stated by MDPI. A software license should be selected explicitly by the software contributors before reuse terms for the code are broadened beyond the rights granted by the published supplementary material.

## Links

- Article: https://doi.org/10.3390/sci8070169
- MDPI supplementary material: https://www.mdpi.com/article/10.3390/sci8070169/s1

An exact copy of the published MDPI supplementary ZIP is preserved at `archive/sci-4311883-supplementary.zip`. The root analysis script and the CSV tables under `analysis_outputs/` are extracted directly from that archive. `SHA256SUMS.txt` records and verifies the archive and extracted-file hashes.
- Lucas Orbolato Carvalho: https://orbolato.github.io/
