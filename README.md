# ULTR-AI

**Ultrasound-led tuberculosis recognition using artificial intelligence**

ULTR-AI is a research codebase for developing and evaluating artificial intelligence models for pulmonary tuberculosis (TB) triage from lung ultrasound (LUS). It implements patient-level TB prediction across multiple ultrasound views and image-level detection of human-recognizable LUS pathology signs.

The associated prospective cohort study is available as a preprint:

> Suttels V, Brokowski T, Wachinou AP, et al. *Lung Ultrasound for the Detection of Pulmonary Tuberculosis Using Expert- and AI-Guided Interpretation: A Prospective Cohort Study*. SSRN. 2025. [doi:10.2139/ssrn.5174193](https://doi.org/10.2139/ssrn.5174193).

> **Research use only:** ULTR-AI is not a certified medical device and must not be used for clinical diagnosis or patient-management decisions without external validation, appropriate regulatory review, and integration into a governed clinical workflow. The linked manuscript is a preprint and has not been peer reviewed.

## Scientific scope

The manuscript describes three complementary approaches:

1. **ULTR-AI** predicts TB directly from LUS images using deep learning.
2. **ULTR-AI[signs]** first detects human-recognizable LUS pathology signs and then predicts TB risk using a machine-learning model.
3. **ULTR-AI[max]** combines the two approaches using the maximum predicted TB-risk score.

This repository provides the underlying workflows for:

- Patient-level TB classification using multiple LUS images and learned feature aggregation
- Image-level classification of A-lines, B-lines, coalescing B-lines, consolidations or nodules, and pleural effusion
- Diagnostic-performance evaluation using sensitivity, specificity, ROC–AUC, precision–recall AUC, and related metrics

## Study context

The associated study was a prospective cohort study conducted among symptomatic patients at a tertiary center in urban Benin. A trained operator performed a standardized 14-point sliding-scan LUS protocol; two blinded, independent readers reviewed the images; and a same-day single-sputum Xpert MTB/RIF Ultra assay served as the microbiological reference standard.

Of 760 people screened, 504 were included in the analysis and 192 (38%) had bacteriologically confirmed TB. The analyzed cohort had a median age of 40 years (IQR 30–52); 196 participants (39%) were female, 78 (15%) were people with HIV, and 66 (13%) had previous TB.

These figures describe the manuscript cohort, not a claim of external validity. Independent validation in other populations, settings, operators, and devices is required.

## Repository structure

```text
UltrAI/
├── trainTB.py                  # Patient-level TB training
├── train_pathology.py          # Image-level pathology training
├── predictTBImage.py           # Model inference and prediction
├── dataset_loading/            # Dataset and preprocessing utilities
├── evaluation/                 # Metrics and evaluation workflows
├── network_architecture/       # Model and pooling architectures
├── utilities/                  # Configuration and shared utilities
├── data/
│   ├── clinical_data/          # Study-level clinical variables
│   ├── labels/                 # Study IDs and analysis labels
│   ├── Splits/                 # Predefined train/validation/test folds
│   └── README.md               # Provenance, schema, and governance notes
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.9 or newer is recommended. A CUDA-capable GPU is recommended for training but is not required for code inspection or CPU execution.

```bash
git clone https://github.com/tbrokowski/UltrAI.git
cd UltrAI

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so that the local packages resolve correctly.

## Data

See [`data/README.md`](data/README.md) for:

- Study provenance and cohort-level descriptors
- File-level schemas and label interpretation
- Split organization and leakage-prevention requirements
- Ethics, consent, privacy, and licensing boundaries

The repository includes coded participant identifiers and clinical variables. Coded or pseudonymous data are not necessarily anonymous. Users are responsible for confirming that their access and intended use comply with the applicable consent, ethics approval, data-use agreements, institutional policy, and law.

## Usage

### Patient-level TB classification

```bash
python trainTB.py
```

Configuration values can be overridden from the command line:

```bash
python trainTB.py \
  --images_directory /path/to/images \
  --labels_file /path/to/labels.csv \
  --test_indices_file /path/to/Fold_0.csv \
  --learning_rate 0.0005 \
  --batch_size 16
```

### Image-level pathology classification

Pass configuration overrides as JSON:

```bash
python train_pathology.py '{"feature":"A-lines","fold":0,"pos_weight":1.6}'
```

Other supported targets include B-lines, coalescing B-lines, small consolidations or nodules, large consolidations, and pleural effusion.

## Model architecture

ULTR-AI combines:

- A pretrained convolutional image encoder, typically ResNet-18
- Optional positional encoding for anatomical scan sites
- Attention, Transformer, DeepSet, maximum, or mean aggregation
- A binary classifier optimized with class-weighted loss

## Evaluation

The evaluation utilities report:

- Accuracy
- Sensitivity
- Specificity
- Balanced accuracy
- ROC–AUC
- Area under the precision–recall curve
- F1 score
- Confusion matrix

Train, validation, and test predictions can be exported as CSV files. Model checkpoints and generated outputs are excluded from version control by default.

## Reproducibility

The repository records a default random seed, predefined cross-validation folds, model configuration, and evaluation procedures. For a manuscript-aligned analysis:

1. Use a tagged, archived software release rather than the moving `main` branch.
2. Record the exact dependency environment, model configuration, random seed, and hardware.
3. Preserve participant-level separation across training, validation, and test sets.
4. Report which outcome definition and sensitivity-analysis label file were used.
5. Confirm that the archived release matches the code used to generate the reported results.

## Ethics, funding, and competing interests

As reported in the associated manuscript:

- The University of Parakou local ethics committee for biomedical research approved the protocol on 18 May 2021 (reference `0407/CLERB-UP/P/SP/R/SA`).
- All participants provided written informed consent.
- The Swiss Pulmonary League funded the study.
- The authors declared no competing interests.

These statements describe the reported study. They do not independently authorize redistribution or secondary use of participant-level data.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). It distinguishes between:

- **The software release**, which should receive its own version-specific archival DOI after a GitHub release is deposited in Zenodo
- **The associated research article**, which has the SSRN DOI [`10.2139/ssrn.5174193`](https://doi.org/10.2139/ssrn.5174193)

Until a software DOI is minted, cite the exact repository version or commit used and cite the associated manuscript. Do not use the SSRN article DOI as the software DOI.

## License

The source code is licensed under the [Apache License 2.0](LICENSE). This software license does **not** grant rights to participant data, clinical records, images, annotations, model weights, or other third-party material unless those materials are explicitly identified as covered by the license.

## Contact

For research questions, collaboration, or data-access inquiries, contact the Laboratory for Intelligent Global Health and Humanitarian Response Technologies (LiGHT) at EPFL.
