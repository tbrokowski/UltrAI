# Data documentation

This directory contains files used by ULTR-AI training and evaluation. It should be interpreted together with the associated study:

> Suttels V, Brokowski T, Wachinou AP, et al. *Lung Ultrasound for the Detection of Pulmonary Tuberculosis Using Expert- and AI-Guided Interpretation: A Prospective Cohort Study*. SSRN. 2025. [doi:10.2139/ssrn.5174193](https://doi.org/10.2139/ssrn.5174193).

## Study provenance

The manuscript reports a prospective cohort study of symptomatic patients at a tertiary center in urban Benin. A trained operator acquired LUS using a standardized 14-point sliding-scan protocol. Two blinded, independent readers reviewed the LUS images. A same-day single-sputum Xpert MTB/RIF Ultra assay was the microbiological reference standard.

Of 760 people screened, 504 were analyzed; 192 (38%) had bacteriologically confirmed TB. The analyzed cohort had a median age of 40 years (IQR 30–52), included 196 female participants (39%), 78 people with HIV (15%), and 66 participants with previous TB (13%).

## Directory inventory

| Path | Unit of observation | Purpose |
|---|---|---|
| `clinical_data/verosdata.csv` | One row per study participant | Clinical, laboratory, imaging-interpretation, management, and follow-up variables |
| `labels/sensitivity_analysis_labels.csv` | One row per study participant | Binary TB analysis label consumed by the patient-level training workflow |
| `Splits/Fold_0.csv` … `Splits/Fold_4.csv` | Participant identifier assignment | Predefined train, validation, and test partitions |

The LUS image dataset is linked to participant records through coded `record_id` values and filename conventions used by the loaders. Image availability and access conditions should be documented separately if images are distributed outside this repository.

## Core identifiers and labels

### `record_id`

`record_id` is the coded study-participant identifier used to join clinical records, labels, split assignments, and image filenames. It is a linkage key and must be treated as potentially sensitive even though it is not a participant name.

### `TB Label`

`TB Label` is the binary outcome consumed by the current training code:

- `1`: positive analysis label
- `0`: negative analysis label

The filename indicates that this is a sensitivity-analysis label set. Users must document the precise outcome definition used in their analysis and should not assume that every positive value is synonymous with the manuscript's bacteriologically confirmed-TB endpoint without checking the study analysis plan and label-generation procedure.

### Clinical TB variables

The clinical table contains multiple TB-related variables, including `no_tb_v2`, `clinical_tb_v2`, `confirmed_tb_v2`, Xpert results, treatment variables, and follow-up outcomes. These variables are not interchangeable. The manuscript's primary microbiological reference standard was same-day single-sputum Xpert MTB/RIF Ultra.

## Clinical-data domains

`clinical_data/verosdata.csv` contains 504 participant rows and 212 columns. The fields span:

| Domain | Examples | Interpretation note |
|---|---|---|
| Study linkage | `record_id` | Coded linkage identifier |
| TB outcome definitions | `no_tb_v2`, `clinical_tb_v2`, `confirmed_tb_v2` | Distinct analysis categories; do not collapse without the analysis plan |
| LUS interpretation | `lus_*_final`, `interprtation_lus_final_complete` | Site-specific expert-read variables and completion status |
| FASH interpretation | `fash_*_final`, `interprtation_fash_final_complete` | Focused assessment variables recorded in the study |
| Symptoms and history | `chief_complaint`, symptom onset/duration, `previous_tb_diagnosis`, `hiv` | Clinical presentation and medical history |
| Examination | weight, height, temperature, vital signs, BMI, qSOFA fields | Enrollment examination |
| Laboratory testing | COVID-19, TB, HIV, malaria, hematology fields | Test identifiers and results may coexist |
| Management | hospitalization, antibiotics, oxygen, TB treatment | Care decisions and treatment |
| Radiography | `cxr_*` fields | Chest-radiograph interpretation |
| Follow-up | day 7, day 28, and six-month outcome fields | Follow-up status and outcomes |
| Demographics | `sexe`, `age` | Participant characteristics |

A source data dictionary defining every variable, allowed value, missing-value code, derivation, unit, and language mapping is not included here. Numeric category codes such as `0`, `1`, `2`, and special missingness codes must not be interpreted from the column name alone. Reproducible secondary analysis requires the original study codebook or REDCap data dictionary.

## Split integrity

The training workflow uses participant-level split files so that images from the same participant remain in a single partition within a fold. Analyses should verify:

1. No participant appears in more than one of `train_ids`, `valid_ids`, and `test_ids` within a fold.
2. All images inherit the partition of their linked `record_id`.
3. Label and clinical joins are one-to-one on `record_id`.
4. Any records excluded from the predefined folds are reported rather than silently reassigned.
5. Model selection does not use the test partition.

The split files are part of the analytic specification and should be preserved unchanged for manuscript reproduction.

## Missingness and data types

Empty cells and coded values may represent different concepts, including not collected, not indicated, invalid, unknown, not asked, or not applicable. Dates, comma-formatted numerals, free text, and multilingual entries are present. Therefore:

- Import the UTF-8 CSV explicitly and preserve `record_id` as text.
- Do not globally coerce empty or coded values to a single missingness category.
- Remove thousands separators only for variables confirmed to be numeric.
- Parse dates field by field and retain the original values.
- Record every recoding and exclusion in an analysis log.

## Ethics, privacy, and permitted use

The manuscript reports ethics approval by the University of Parakou local ethics committee for biomedical research on 18 May 2021 (reference `0407/CLERB-UP/P/SP/R/SA`) and written informed consent from all participants.

The clinical file includes coded identifiers and granular clinical information, including dates and laboratory or register identifiers. Coded data can remain re-identifiable when combined with other information. Repository visibility does not establish that the data are anonymous or that unrestricted secondary use is permitted.

Before using, sharing, or redistributing participant-level data:

- Confirm the governing consent, ethics approval, data-use agreement, and institutional authorization.
- Apply data minimization and remove fields unnecessary for the stated analysis.
- Do not attempt re-identification or link these records to external data.
- Use controlled access and an approved secure environment where required.
- Report any public-data release separately from the software release.

## Licensing boundary

The repository's Apache-2.0 license applies to the source code. It does not automatically apply to participant data, LUS images, clinical annotations, or other study materials. Data users must obtain the rights and permissions applicable to those materials.

## Funding and declarations

The manuscript reports funding from the Swiss Pulmonary League and no competing interests.

## Recommended dataset citation

No separate dataset DOI is currently specified. Do not cite the SSRN article DOI as a dataset DOI. Until a governed dataset record is created, cite the manuscript and the exact software release or commit used, and describe the data-access pathway in the methods.
