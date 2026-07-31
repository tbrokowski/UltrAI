# ULTR-AI

**Deep learning for tuberculosis and lung pathology classification from lung ultrasound**

ULTR-AI is a research codebase for developing and evaluating deep learning models that classify tuberculosis (TB) and lung ultrasound (LUS) pathology patterns. It supports patient-level TB classification across multiple ultrasound views and image-level classification of individual pathology features.

> **Research use only:** ULTR-AI is not a certified medical device and must not be used for clinical diagnosis or patient-management decisions without appropriate validation and regulatory review.

## Scope

The repository provides two principal workflows:

1. **Patient-level TB classification** using multiple LUS images per participant and attention-based feature aggregation.
2. **Image-level pathology classification** for A-lines, B-lines, coalescing B-lines, consolidations, nodules, and pleural effusion.

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
│   ├── labels/                 # Study IDs and outcome labels
│   └── Splits/                 # Predefined train/validation/test folds
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.9 or newer is recommended. A CUDA-capable GPU is recommended for training but is not required for basic code inspection or CPU execution.

```bash
git clone https://github.com/tbrokowski/UltrAI.git
cd UltrAI

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root so that the local packages resolve correctly.

## Data organization

The patient-level workflow expects the following layout:

```text
data/
├── images/
│   ├── 12345_QAID_1.png
│   ├── 12345_QAIG_1.png
│   └── ...
├── labels/
│   └── sensitivity_analysis_labels.csv
└── Splits/
    ├── Fold_0.csv
    ├── Fold_1.csv
    ├── Fold_2.csv
    ├── Fold_3.csv
    └── Fold_4.csv
```

The label file contains:

- `record_id`: pseudonymous study participant identifier
- `TB Label`: binary TB outcome label

Each split file contains `train_ids`, `valid_ids`, and `test_ids`. Image filenames encode the study identifier and anatomical scan site.

Access to and redistribution of research data must remain consistent with the relevant consent, ethics approval, and data-use agreements. Do not add linkage keys or information that would permit participant re-identification.

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

The repository records the default random seed, predefined cross-validation folds, model configuration, and evaluation procedures. Before creating the manuscript release:

1. Validate the full workflow in a clean environment.
2. Export the exact working dependency versions to a lock file.
3. Record the final model configuration and random seed.
4. Confirm that the archived release matches the code used for the reported results.
5. Document how authorized researchers can obtain any non-redistributable data or model artifacts.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). After the manuscript release is archived in Zenodo, add the assigned DOI to that file and place the Zenodo badge at the top of this README.

Suggested software title:

> *ULTR-AI: Lung Ultrasound Tuberculosis and Pathology Classification*

For academic work, cite the version-specific Zenodo DOI corresponding to the exact release used in the study.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

For research questions, collaboration, or access inquiries, contact the Intelligent Global Health Research Group at EPFL.
