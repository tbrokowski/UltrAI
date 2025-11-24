# ULTR-AI: Lung Ultrasound Tuberculosis and Pathology Classification

This repository contains the codebase for training deep learning models to classify tuberculosis (TB) and various lung ultrasound (LUS) pathology patterns from LUS images.

## Overview

The ULTR-AI project provides two main training scripts:

1. **trainTB.py**: Patient-level TB classification (TB+ vs TB-)
   - Aggregates multiple LUS images per patient
   - Uses attention-based pooling to combine features from different anatomical sites
   - Trains on patient-level labels

2. **train_pathology.py**: Image-level pathology classification
   - Classifies individual pathology features in LUS images
   - Supports multiple pathology types: A-lines, B-lines, consolidations, pleural effusion
   - Can train separate models for each pathology feature

## Project Structure

```
UltrAI/
├── trainTB.py                 # Main script for TB classification training
├── train_pathology.py         # Main script for pathology classification training
├── predictTBImage.py          # Script for making predictions with trained models
├── data/                      # Data directory
│   ├── images/               # LUS images (PNG format)
│   ├── labels/               # Label CSV files
│   └── Splits/               # Train/validation/test split files
├── evaluation/               # Evaluation metrics and model evaluation
│   ├── metrics.py            # Metric computation functions
│   └── model_evaluation.py   # Model evaluation utilities
├── network_architecture/     # Model architectures
│   ├── resnet.py            # ResNet backbone
│   ├── pooling.py           # Aggregation/pooling layers
│   └── trainingutils.py     # Training utilities
├── dataloaders/             # Data loading utilities
└── data-processing/         # Data preprocessing scripts
```

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- DeepChest package (located in `/Users/trevorbrokowski/Downloads/deepchest`)

## Installation

### 1. Install Python Dependencies

```bash
cd "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI"
pip install -r requirements.txt
```

### 2. Set Up DeepChest Package

The `deepchest` package is now included in the ULTR-AI folder. You need to ensure it's accessible in your Python path:

**Option A: Add to PYTHONPATH (Recommended)**
```bash
export PYTHONPATH="/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI:$PYTHONPATH"
```

**Option B: Install as package**
```bash
cd "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI"
pip install -e .
```

### 3. Verify Installation

```bash
cd "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI"
python3 -c "import sys; sys.path.insert(0, '.'); from deepchest.utilities import config_utils, utils; print('✓ DeepChest import successful')"
```

## Data Preparation

### Required Data Files

1. **Images Directory**: Directory containing LUS images
   - Format: `{patient_id}_{site}_{optional_number}.png`
   - Example: `12345_QAID_1.png`
   - Sites: QAID, QAIG, QASD, QASG, QLD, QLG, QPID, QPIG, QPSD, QPSG, APXD, APXG, QSLD, QSLG

2. **Labels File (for TB training)**:
   - CSV file with columns: `record_id`, `TB Label`
   - `record_id`: Patient ID (integer)
   - `TB Label`: Binary label (0 = TB-, 1 = TB+)

3. **Labels File (for pathology training)**:
   - CSV file with columns: `path`, and pathology feature columns
   - `path`: Path to image file
   - Feature columns: `A-lines`, `B-lines`, `Large Consolidations`, etc. (binary: 0 or 1)

4. **Split File**:
   - CSV file with columns: `train_ids`, `valid_ids`, `test_ids`
   - Contains patient IDs for each split

### Example Data Structure

```
data/
├── images/
│   ├── 12345_QAID_1.png
│   ├── 12345_QAIG_1.png
│   ├── 12345_QASD_1.png
│   └── ...
├── labels/
│   ├── pivotedlabels.csv        # For TB training
│   └── imagedf.csv              # For pathology training
└── Splits/
    ├── Fold_0.csv
    ├── Fold_1.csv
    └── ...
```

## Usage

### Training TB Classification Model

```bash
cd "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI"

# Basic usage (uses default config)
python3 trainTB.py

# With command-line overrides
python3 trainTB.py --learning_rate 0.0005 --batch_size 32 --images_directory /path/to/images

# Override nested config values
python3 trainTB.py --resnet.freeze False --aggregation_type "Transformer"
```

**Key Configuration Parameters:**
- `images_directory`: Path to directory containing LUS images
- `labels_file`: Path to CSV file with patient labels
- `test_indices_file`: Path to CSV file with train/val/test splits
- `save_dir`: Directory to save model checkpoints
- `pred_save_dir`: Directory to save prediction CSV files
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 16)
- `pos_weight`: Positive class weight for handling imbalance (default: 1.6)
- `nb_epochs`: Number of training epochs (default: 75)

### Training Pathology Classification Model

```bash
cd "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI"

# Train A-lines classifier on Fold 0
python3 train_pathology.py '{"feature": "A-lines", "fold": 0, "pos_weight": 1.6}'

# Train B-lines classifier
python3 train_pathology.py '{"feature": "B-lines", "fold": 0, "pos_weight": 1.8}'

# Train Large Consolidations classifier
python3 train_pathology.py '{"feature": "Large Consolidations", "fold": 0, "pos_weight": 3}'

# Train on different fold
python3 train_pathology.py '{"feature": "A-lines", "fold": 1, "pos_weight": 1.6, "run_id": "1"}'
```

**Key Configuration Parameters (JSON format):**
- `feature`: Pathology feature to classify (required)
  - Options: "A-lines", "B-lines", "Coalescing B-lines", "Small consolidations and or nodules", "Large Consolidations", "Pleural effusion"
- `fold`: Cross-validation fold number (0-4, default: 0)
- `run_id`: Unique identifier for this run (default: '1')
- `pos_weight`: Positive class weight (default: 4.0)
- `learning_rate`: Learning rate (default: 0.00001)
- `batch_size`: Batch size (default: 32)
- `nb_epochs`: Number of epochs (default: 25)

## Model Architecture

### DeepChest Model

The model consists of three main components:

1. **Image Encoder (ResNet)**: Extracts features from individual LUS images
   - Uses pretrained ResNet (default: ResNet18)
   - Can be frozen or fine-tuned
   - Output: 512-dimensional feature vectors

2. **Aggregation Network**: Combines features from multiple images per patient
   - Options: MLP_AttentionPooling, Transformer, DeepSet, Max/Mean Pooling
   - Uses positional embeddings for anatomical sites
   - Handles variable number of images per patient

3. **Classifier**: Final classification layer
   - Binary classification: TB+ vs TB- or pathology present/absent
   - Uses weighted BCE loss to handle class imbalance

## Output Files

### Training Outputs

1. **Model Checkpoints**:
   - `checkpoint_best.pth`: Best model based on validation metric
   - `highsens.pth`: Model with high sensitivity (>0.9) and specificity (>0.8)
   - `highauroclowerloss.pth`: Model with high AUC (>0.95) and low loss

2. **Configuration**:
   - `config.yaml`: Saved configuration for reproducibility

3. **Predictions**:
   - `{feature}_trainpredictions.csv`: Predictions on training set
   - `{feature}_validpredictions.csv`: Predictions on validation set
   - `{feature}_testpredictions.csv`: Predictions on test set

4. **Indices**:
   - `{run_name}_indices.csv`: Train/validation/test split indices

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: Overall classification accuracy
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **ROC-AUC**: Area under the ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError: No module named 'deepchest'`:

1. Verify the deepchest package exists at `/Users/trevorbrokowski/Downloads/deepchest`
2. Add it to PYTHONPATH: `export PYTHONPATH="/Users/trevorbrokowski/Downloads:$PYTHONPATH"`
3. Or install it: `cd /Users/trevorbrokowski/Downloads/deepchest && pip install -e .`

### CUDA Out of Memory

If you run out of GPU memory:

1. Reduce `batch_size` in config
2. Increase `accumulation_steps` to maintain effective batch size
3. Enable gradient checkpointing (if supported)
4. Use smaller model (e.g., ResNet18 instead of ResNet50)

### Data Loading Issues

- Verify image filenames match expected format: `{patient_id}_{site}_{optional}.png`
- Check that patient IDs in labels file match image filenames
- Ensure split files contain valid patient IDs
- Verify CSV files have correct column names

### Path Issues

- Update paths in `get_config()` functions to match your data locations
- Use absolute paths for better reliability
- Ensure all directories exist or will be created

## Configuration Files

Both training scripts use `ml_collections.ConfigDict` for configuration management. You can:

1. Modify default values in `get_config()` functions
2. Override via command-line arguments (trainTB.py)
3. Override via JSON arguments (train_pathology.py)
4. Edit saved `config.yaml` files for reference

## Citation

If you use this code, please cite:

```
Intelligent Global Health Research Group, EPFL
ULTR-AI: Deep Learning for Lung Ultrasound Tuberculosis Classification
2023
```

## License

Apache License 2.0

## Contact

For questions or issues, please contact the Intelligent Global Health Research Group at EPFL.

## Additional Notes

- The code uses mixed precision training (AMP) for faster training and lower memory usage
- Early stopping is implemented to prevent overfitting
- Models are saved based on validation performance, not training performance
- Class balancing is used in pathology training to handle imbalanced datasets
- Data augmentation is applied during training but not during validation/testing

