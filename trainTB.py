import pathlib
import sys
import ml_collections
import os
import torch
import yaml
from torch.nn import BCEWithLogitsLoss
from utilities import config_utils, utils
from dataset_loading import dataset
from evaluation.metrics import compute_metrics, BestMetricVal
from evaluation.model_evaluation import model_evaluation, run_and_save_predictions
from network_architecture.deepchest import DeepChest
import re
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import InterpolationMode
from dataset_loading.dataset import LungUltrasoundPatientDataset, collate_fn, seed_worker
from dataset_loading import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import functools
from torch.cuda.amp import autocast, GradScaler
import gc
from dataclasses import dataclass


def print_gpu_memory():
    """
    Print current GPU memory usage.
    
    Useful for debugging out-of-memory errors during training.
    """
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    else:
        print("CUDA not available, cannot check GPU memory")


@dataclass
class ImageInfo:
    """
    Data class to store information extracted from image filenames.
    
    Attributes:
        patient_id (int): The unique identifier for the patient
        site (str): The anatomical site code (e.g., 'QAID', 'QASG', etc.)
    
    The filename format expected is: {patient_id}_{site}_{optional_number}.png
    Example: "12345_QAID_1.png" -> patient_id=12345, site="QAID"
    """
    patient_id: int
    site: str

    @staticmethod
    def from_filename(filename: str):
        """
        Parse a filename to extract patient ID and site information.
        
        Args:
            filename (str): Image filename in format "{patient_id}_{site}_{optional}.png"
            
        Returns:
            ImageInfo: Parsed image information
            
        Raises:
            ValueError: If filename doesn't match expected pattern
        """
        match = re.match(r"^(\d+)_([A-Z]+)(_\d+)?(\..*)?\.png$", filename)
        if match is None:
            raise ValueError(f"Could not parse '{filename}' into an ImageInfo.")
        patient_id = int(match.group(1))
        site = match.group(2)
        return ImageInfo(
            patient_id,
            site,
        )

def get_config():
    """
    Create and return the configuration dictionary for TB classification training.
    
    This function sets up all hyperparameters, paths, and model architecture settings.
    Configuration can be overridden via command-line arguments using config_utils.parse_cli_overides().
    
    Returns:
        ml_collections.ConfigDict: Configuration object with all training parameters
        
    Configuration Sections:
        - Paths: Directories for saving models, predictions, and data
        - Optimizer: Learning rate, weight decay, batch size, class weights
        - Dataset: Image directories, label files, train/val/test splits
        - Training: Number of epochs, evaluation metrics, early stopping
        - Model: ResNet backbone, aggregation method, encoding dimensions
    """
    config = ml_collections.ConfigDict()
    
    # ========== Paths Configuration ==========
    project_root = pathlib.Path(__file__).resolve().parent
    # Directory where model checkpoints will be saved
    config.save_dir = str(project_root / "results" / "models")
    # Directory where prediction CSV files will be saved
    config.pred_save_dir = str(project_root / "results")
    # Name identifier for this training run
    config.run_name = "run 1"
    # Unique ID for this run (used to create subdirectories)
    config.run_id = '1'
    # Feature/fold identifier for naming output files
    config.feature = 'Fold_0'

    # ========== General Training Configuration ==========
    # Random seed for reproducibility
    config.seed = 0
    # Device to use (CUDA GPU if available, else CPU)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Flag for hyperparameter optimization (currently unused but kept for compatibility)
    config.hyperparm_optim = True

    # ========== Optimizer Configuration ==========
    # Learning rate for the optimizer
    config.learning_rate = 0.001
    # L2 regularization weight decay
    config.weight_decay = 0.0
    # Number of samples per batch
    config.batch_size = 16
    # Positive class weight for handling class imbalance (higher = more weight on positive class)
    # Best hyperparameters found: lr=0.001, pos_weight=1.2, batch_size=16, accumulation_steps=2
    config.pos_weight = 1.6
    # Whether to use pooling strategy for handling variable number of images per patient
    config.pooling = True

    # ========== Dataset Configuration ==========
    # Path to directory containing all LUS images (PNG format)
    config.images_directory = str(project_root / "data" / "images")
    # Path to CSV file containing patient labels (must have 'record_id' and 'TB Label' columns)
    config.labels_file = str(project_root / "data" / "labels" / "sensitivity_analysis_labels.csv")
    # Path to CSV file containing train/validation/test split indices
    # CSV should have columns: 'train_ids', 'valid_ids', 'test_ids'
    config.test_indices_file = str(project_root / "data" / "Splits" / "Fold_4.csv")
    
    # ========== Cross-Validation Configuration ==========
    # Total number of folds for cross-validation (currently using pre-defined splits)
    config.num_folds = 5
    # Index of fold to use for validation (0-indexed)
    config.valid_fold_index = 0
    # Fraction of data to reserve for testing (if not using pre-defined splits)
    config.test_size = 0.2
    # Number of classes (2 for binary classification: TB+ vs TB-)
    config.num_classes = 2
    # Preprocessing functions for train and eval (semicolon-separated, empty string = no preprocessing)
    # Format: "train_preprocessing;eval_preprocessing"
    # Example: "independent_dropout(.2);" applies dropout only during training
    config.preprocessing_train_eval = ";"
    # Number of worker processes for data loading
    config.num_workers = 4
    # macOS uses 'spawn' start method, which requires picklable callables; our preprocessing
    # function is a local closure and not picklable. Avoid multiprocessing on macOS.
    if sys.platform == "darwin":
        config.num_workers = 0
    # Path to save the train/val/test indices mapping
    config.export_folds_indices_file = config.save_dir + "indices.csv"

    # ========== Training Loop Configuration ==========
    # Maximum number of training epochs
    config.nb_epochs = 75
    # Evaluate model every N steps (1 = every epoch)
    config.eval_every_steps = 1
    # Metric to use for model selection (best model saved based on this)
    config.eval_metric = "roc_auc"
    # Whether to maximize or minimize the eval_metric
    config.eval_metric_goal = "max"  # "min" for metrics like loss
    # Whether to evaluate the best model on test set after training
    config.evaluate_best_valid_model = True

    # ========== Image Representation Network (ResNet Backbone) ==========
    config.resnet = ml_collections.ConfigDict()
    # Whether to use pretrained ImageNet weights
    config.resnet.pretrained = True
    # Path to custom pretrained weights (None = use ImageNet)
    config.resnet.pretrained_path = None
    # Whether to freeze ResNet weights (True = only train aggregation/classifier layers)
    config.resnet.freeze = True
    # ResNet architecture variant ('resnet18', 'resnet34', 'resnet50', etc.)
    # Other options: 'inceptionresnetv2' (commented out)
    config.resnet.model_name = 'resnet18'

    # ========== Aggregation Network Configuration ==========
    # Method to aggregate multiple images per patient into single representation
    # Options: "MLP_AttentionPooling", "MLP_MaxPooling", "MLP_MeanPooling", 
    #         "Transformer", "DeepSet", "AttentionPooling", "MaxPooling", "MeanPooling"
    config.aggregation_type = "MLP_AttentionPooling"
    # Dimension of the encoded feature vectors (after ResNet, before aggregation)
    config.encoding_dim = 512
    # Whether to use positional embeddings for anatomical sites
    # This helps the model understand spatial relationships between different scan sites
    config.use_positional_embeddings = True
    # Gradient accumulation steps (simulates larger batch size)
    # Effective batch size = batch_size * accumulation_steps
    config.accumulation_steps = 2

    return config

def get_data_loaders(config):
    """
    Load and prepare data loaders for training, validation, and testing.
    
    This function:
    1. Loads patient labels from CSV file
    2. Loads train/val/test split indices from CSV file
    3. Filters IDs to ensure they exist in both labels and image directory
    4. Creates PyTorch datasets and data loaders with appropriate transforms
    5. Checks for data leakage (overlap between splits)
    
    Args:
        config (ml_collections.ConfigDict): Configuration object with dataset paths and parameters
        
    Returns:
        tuple: (train_loader, test_loader, validation_loader)
            - train_loader: DataLoader for training with data augmentation
            - test_loader: DataLoader for testing (no augmentation)
            - validation_loader: DataLoader for validation (no augmentation)
            
    Raises:
        ValueError: If no images found in images_directory
        FileNotFoundError: If label file or split file doesn't exist
    """
    # Load patient labels from CSV file
    # Expected columns: 'record_id' (patient ID), 'TB Label' (0 or 1 for TB- or TB+)
    labels_df = pd.read_csv(config.labels_file)
    # Clean patient IDs: remove '25-' prefix if present and convert to integer
    labels_df['record_id'] = labels_df['record_id'].map(lambda x: int(x.replace('25-', '')) if isinstance(x, str) else x)
    full_indices = labels_df['record_id'].values.flatten()
    full_labels = labels_df['TB Label'].values.flatten()
    
    # Load train/validation/test split indices from CSV file
    # Expected columns: 'train_ids', 'valid_ids', 'test_ids'
    data = pd.read_csv(config.test_indices_file)
    # Clean IDs in split file: remove '25-' prefix if present
    data = data.map(lambda x: int(x.replace('25-', '')) if isinstance(x, str) else x)
    train_ids = data['train_ids'].dropna().tolist()
    valid_ids = data['valid_ids'].dropna().tolist()
    test_ids = data['test_ids'].dropna().tolist()

    full_indices_dtype = full_indices.dtype

    train_ids = [full_indices_dtype.type(t) for t in train_ids if full_indices_dtype.type(t) in full_indices]
    valid_ids = [full_indices_dtype.type(v) for v in valid_ids if full_indices_dtype.type(v) in full_indices]
    test_ids = [full_indices_dtype.type(te) for te in test_ids if full_indices_dtype.type(te) in full_indices]

    images_directory = pathlib.Path(config.images_directory)
    images_list = os.listdir(images_directory)
    images_ids = set(ImageInfo.from_filename(f).patient_id for f in images_list)

    # Get the dtype of elements in the set
    if images_ids:
        images_ids_dtype = type(next(iter(images_ids)))
    else:
        raise ValueError("The set images_ids is empty, cannot determine dtype.")

    # Convert train, valid, and test IDs to the same dtype and filter
    train_ids = [images_ids_dtype(t) for t in train_ids if images_ids_dtype(t) in images_ids]
    valid_ids = [images_ids_dtype(v) for v in valid_ids if images_ids_dtype(v) in images_ids]
    test_ids = [images_ids_dtype(te) for te in test_ids if images_ids_dtype(te) in images_ids]

    # Check for data leakage: ensure no patient appears in multiple splits
    k = [v for v in valid_ids if v in train_ids]
    y = [v for v in test_ids if v in train_ids]
    y = k + y
    if y:
        print(f"WARNING: Data leakage detected! Patients in multiple splits: {y}")
    else:
        print("Data leakage check passed: no overlap between splits")

    labels_dict = dict(zip(full_indices, full_labels))

    # train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(indices, labels, shuffle=True,
    #                                                                                       test_size=config.test_size,
    #                                                                                       random_state=config.seed,
    #                                                                                       stratify=labels)

    #test_indices = pd.read_csv(config.test_indices_file).values.flatten()
    #train_val_indices = [i for i in indices if i not in test_indices]
    #train_val_labels =  [l for i,l in zip(indices,labels) if i in train_val_indices]

    # k fold over the train/valid part
    # folds = utils.split_k_folds(train_val_indices, train_val_labels, k=config.num_folds)
    # train_folds_indices = list(range(0, config.num_folds))
    # train_folds_indices.remove(config.valid_fold_index)
    # validation_indices = folds[config.valid_fold_index]
    # train_folds = [fold for fold_idx, fold in enumerate(folds) if fold_idx in train_folds_indices]
    # train_indices = np.concatenate(train_folds)
    
    #####
    if config.export_folds_indices_file:
        serie_train = pd.Series(train_ids, name='train_indices')
        serie_test = pd.Series(test_ids, name='test_indices')
        serie_valid = pd.Series(valid_ids, name='validation_indices')
        df_indices = pd.concat([serie_train, serie_test, serie_valid], axis=1)
        df_indices.to_csv(config.export_folds_indices_file, index=False)

    # ========== Image Transforms ==========
    # Transform for validation/test: no augmentation, only resize and normalize
    # Uses ImageNet normalization stats (standard for pretrained models)
    transform_vanilla = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    # Transform for training: includes data augmentation to improve generalization
    # Augmentations: color jitter, rotation, affine transformation
    transform_with_augmentation = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        # Color augmentation: randomly adjust brightness, contrast, saturation, hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Random rotation: up to 20 degrees
        transforms.RandomRotation(degrees=20, interpolation=InterpolationMode.BICUBIC),
        # Random affine: rotation + translation (up to 10% in each direction)
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), interpolation=InterpolationMode.BICUBIC),
        # Convert PIL image to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Note: RandomResizedCrop is commented out but could be added for more augmentation

    dataset = LungUltrasoundPatientDataset(
        images_directory=config.images_directory,
        labels=labels_dict,
    )

    print(labels_dict)
    print(train_ids)

    print(len(train_ids), len(test_ids), len(valid_ids))

    # label_names = utils.get_label_names(config.labels_file)
    # utils.show_splits_info(
    #     train_ids, test_ids, valid_ids, labels_dict, label_names=label_names
    # )

    train_pp, eval_pp = config.preprocessing_train_eval.split(";")
    train_pp, eval_pp = map(preprocessing.make_preprocessing_fn, [train_pp, eval_pp])

    train_subset = Subset(dataset, train_ids)
    train_collate = functools.partial(
        collate_fn, image_transform=transform_with_augmentation, preprocessing_fn=train_pp
    )
    # preserve reproducibility
    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        # batch_sampler=batch_sampler,
        collate_fn=train_collate,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True  
    )

    validation_subset = Subset(dataset, valid_ids)
    valid_collate = functools.partial(
        collate_fn, image_transform=transform_vanilla, preprocessing_fn=eval_pp
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=valid_collate,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_subset = Subset(dataset, test_ids)
    test_collate = functools.partial(
        collate_fn, image_transform=transform_vanilla, preprocessing_fn=eval_pp
    )
    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=test_collate,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, test_loader, validation_loader



def main(config):
    """
    Main training function for TB classification model.
    
    This function orchestrates the entire training process:
    1. Sets up directories and saves configuration
    2. Loads data and creates data loaders
    3. Initializes model, optimizer, and loss function
    4. Runs training loop with validation
    5. Saves best model checkpoints
    6. Evaluates best model on train/val/test sets
    
    Args:
        config (ml_collections.ConfigDict): Configuration object with all training parameters
        
    The function will:
        - Create output directories if they don't exist
        - Save configuration to YAML file for reproducibility
        - Train model for config.nb_epochs epochs
        - Save best model based on validation metric
        - Generate prediction CSV files for all splits
    """
    print("Configuration:", config)
    # Resolve paths to absolute paths for consistency
    config.save_dir = str(pathlib.Path(config.save_dir).resolve())
    config.pred_save_dir = str(pathlib.Path(config.pred_save_dir).resolve())
    print(f"Model save directory: {config.save_dir}")
    
    # Set random seed for reproducibility
    utils.set_seed(config.seed)
    
    # Create output directory for this run
    save_dir = pathlib.Path(config.save_dir) / config.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {save_dir}")
    
    # Set path for saving train/val/test indices
    config.export_folds_indices_file = os.path.join(save_dir, config.run_name + "_indices.csv")
    model_save_dir = os.path.join(save_dir, config.run_name + "bestmodel.pth")
    
    # Lock config to prevent accidental modifications
    config.lock()
    print(f"Run name={config.run_name} | id={config.run_id} | directory={save_dir.resolve()} | modelsavepath = {model_save_dir}")

    # Save configuration to YAML file for reproducibility
    config_file = save_dir / "config.yaml"
    config_file.unlink(missing_ok=True)
    config_file.write_text(yaml.dump(config.to_dict()))
    utils.print_config(config)


    # Check if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"Running on device {torch.cuda.get_device_name(0)}. Number of GPUs available: {torch.cuda.device_count()}"
        )
    else:
        print("Running on CPU.")

    # Load data splits
    train_loader, test_loader, validation_loader = get_data_loaders(config)

    # Create model and move to device
    model = DeepChest(config)
    model.to(config.device)
    #pretrained
    # checkpoint = torch.load('model_saved/first_try/nfol=5_fol=0_agg=MLP_AttentionPooling_dec=0.0_lr=0.01_posweight=1.8_posenc=True/checkpoint_best.pth')
    # model.load_state_dict(checkpoint['model_state'])

    # Set up optimization
    optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
    criterion.to(config.device)

    # ========== Training State Tracking ==========
    # Tracks the best validation metric value (e.g., ROC-AUC)
    best_eval_metric = BestMetricVal(config.eval_metric_goal)
    # Counter for early stopping: stops if no improvement for N epochs
    epochs_since_improvement = 0
    early_stopping_patience = 6
    # Track best validation loss (used for model selection)
    best_valid_loss = 10
    # Mixed precision training scaler (for faster training with less memory)
    scaler = GradScaler()
    # Gradient accumulation steps (already set in config, kept here for reference)
    accumulation_steps = 2

    # Training loop
    for epoch in range(config.nb_epochs):
        # Set model to train mode
        model.train()
        all_targets = []
        all_logits = []
        counter = 0
        running_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
            #target_l.extend([*(batch["label"].cpu().numpy()).astype('float')])
            labels = batch['label'].to(config.device).unsqueeze(1).float()
            # Move data to device
            batch["images"] = batch["images"].to(config.device)
            batch["sites"] = batch["sites"].to(config.device)
            batch["mask"] = batch["mask"].to(config.device)
            batch["label"] = batch["label"].to(config.device)


            try:
                # Mixed precision training: uses float16 for faster computation
                with autocast():
                    # Forward pass: model processes all images for each patient
                    # Returns: scores (logits) and attention weights (for interpretability)
                    scores, attention = model(batch["images"], batch["sites"], batch["mask"])
                    # Compute binary cross-entropy loss with class weighting
                    loss = criterion(scores, labels)
                    # Scale loss by accumulation steps (gradient accumulation)
                    loss = loss / config.accumulation_steps
                    running_loss += loss.item()
                    
                # Store predictions and targets for epoch-level metrics
                all_targets.append(labels.detach())
                all_logits.append(scores.detach())

                # Backward pass with gradient scaling (for mixed precision)
                scaler.scale(loss).backward()

                # Update weights every accumulation_steps batches
                if (i + 1) % config.accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Update optimizer and scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                # Memory management: clear GPU cache after each batch
                torch.cuda.empty_cache()
                del batch, loss
                gc.collect()
                
            except RuntimeError as e:
                # Handle out-of-memory errors gracefully
                if 'out of memory' in str(e):
                    print('WARNING: ran out of memory, skipping batch')
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
                
            # Backward pass
            #optimizer.zero_grad()
            #loss.backward()
            # Update parameters of the model
            #optimizer.step()
            #counter += 1

        scheduler.step()
        torch.cuda.empty_cache()  

        # Compute metrics on all batches in the current epoch
        all_targets = torch.cat(all_targets).cpu().numpy()  
        all_logits = torch.cat(all_logits).cpu().numpy()
        train_epoch_metrics = compute_metrics(all_targets, all_logits)
        train_epoch_metrics['loss'] = running_loss / len(train_loader)

        print(f'Epoch: {epoch + 1}/{config.nb_epochs}')
        utils.log_metrics(f"Train stats", train_epoch_metrics)

        # Compute validation metrics (on all batches in the current epoch)
        validation_epoch_metrics, logits = model_evaluation(model, validation_loader, criterion, config.device)
        utils.log_metrics("Valid stats", validation_epoch_metrics, color="green")

        # ========== Model Checkpointing ==========
        # Save model if it achieves best validation loss AND best eval metric
        if validation_epoch_metrics['loss'] < best_valid_loss:
            if best_eval_metric.append(validation_epoch_metrics[config.eval_metric]):
                print(
                    f"Saving new best performing model ({config.eval_metric}={best_eval_metric.value})."
                )
                # Save checkpoint with model state and optimizer state
                checkpoint_name = 'checkpoint_best.pth'
                checkpoint = {
                    'nb_epochs_finished': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(checkpoint, save_dir / checkpoint_name)
                best_valid_loss = validation_epoch_metrics['loss']
                epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Save additional checkpoint if model achieves high sensitivity and specificity
        # This is useful for clinical applications where high sensitivity is critical
        if validation_epoch_metrics['sensitivity'] > 0.9 and validation_epoch_metrics['specificity'] > 0.8:
            checkpoint_name = 'highsens.pth'
            checkpoint = {
                'nb_epochs_finished': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(checkpoint, save_dir / checkpoint_name)

        # Save checkpoint if model achieves very high ROC-AUC with low loss
        if validation_epoch_metrics['roc_auc'] > 0.95 and validation_epoch_metrics['loss'] < best_valid_loss:
            checkpoint_name = 'highauroclowerloss.pth'
            checkpoint = {
                'nb_epochs_finished': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(checkpoint, save_dir / checkpoint_name)
        # else:
        #     epochs_since_improvement += 1

        if epochs_since_improvement == early_stopping_patience:
            print("Early Stopping Triggered")
            break
        
                # Log selected model stats
    # Load the best model and compute test and valid metrics
    if config.evaluate_best_valid_model:
        checkpoint = torch.load(save_dir / "checkpoint_best.pth")
        model.load_state_dict(checkpoint['model_state'])
        valid_save_dir = os.path.join(config.pred_save_dir, config.feature + '_validpredictions.csv')
        valid_metrics, logit_l = run_and_save_predictions(validation_loader, model, config.device,  criterion, valid_save_dir)
        utils.log_metrics("Best model valid stats", valid_metrics, color="green")

        train_save_dir = os.path.join(config.pred_save_dir, config.feature + '_trainpredictions.csv')
        train_metrics, logit_l = run_and_save_predictions(train_loader, model, config.device,  criterion, train_save_dir)
        utils.log_metrics("Best model train stats", train_metrics, color="green")

        
        test_save_dir = os.path.join(config.pred_save_dir, config.feature + '_testpredictions.csv')
        test_metrics, logit_l = run_and_save_predictions(test_loader, model, config.device, criterion, test_save_dir)
        utils.log_metrics("Best model test stats", test_metrics, color="green")


if __name__ == "__main__":
    """
    Entry point for training script.
    
    Usage:
        python trainTB.py
        python trainTB.py --learning_rate 0.0005 --batch_size 32
        python trainTB.py --images_directory /path/to/images --labels_file /path/to/labels.csv
    
    Command-line arguments can override any config value using dot notation:
        --resnet.freeze False
        --aggregation_type "Transformer"
        --nb_epochs 100
    """
    # Parse command-line arguments and override config if provided
    config = config_utils.parse_cli_overides(get_config())
    # Run training
    main(config)



#python3 /home/tjb76/TBLUScopy/trainTBimages.py


# cd /users/tbrokowski/ULTR-AI-Vid

# # Fold 0
# python3 -u ULTR-AI/UltrAI/trainTB.py \
#   --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --test_indices_file ULTR-AI/UltrAI/data/Splits/Fold_0.csv \
#   --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#   --feature "Fold_0" \
#   --run_name "sensitivity" \
#   --run_id "1"

# # Fold 1
# python3 -u ULTR-AI/UltrAI/trainTB.py \
#   --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --test_indices_file ULTR-AI/UltrAI/data/Splits/Fold_1.csv \
#   --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#   --feature "Fold_1" \
#   --run_name "sensitivity" \
#   --run_id "1"

# # Fold 2
# python3 -u ULTR-AI/UltrAI/trainTB.py \
#   --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --test_indices_file ULTR-AI/UltrAI/data/Splits/Fold_2.csv \
#   --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#   --feature "Fold_2" \
#   --run_name "sensitivity" \
#   --run_id "1"

# # Fold 3
# python3 -u ULTR-AI/UltrAI/trainTB.py \
#   --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --test_indices_file ULTR-AI/UltrAI/data/Splits/Fold_3.csv \
#   --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#   --feature "Fold_3" \
#   --run_name "sensitivity" \
#   --run_id "1"

# # Fold 4
# python3 -u ULTR-AI/UltrAI/trainTB.py \
#   --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --test_indices_file ULTR-AI/UltrAI/data/Splits/Fold_4.csv \
#   --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#   --feature "Fold_4" \
#   --run_name "sensitivity" \
#   --run_id "1"




# export PYTHONPATH="ULTR-AI/UltrAI:${PYTHONPATH:-}"

# for FOLD in 0 1 2 3 4; do
#   echo "=== Running Fold ${FOLD} ==="
#   python3 -u ULTR-AI/UltrAI/trainTB.py \
#     --labels_file ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#     --test_indices_file "ULTR-AI/UltrAI/data/Splits/Fold_${FOLD}.csv" \
#     --pred_save_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#     --save_dir ULTR-AI/UltrAI/results/models_sensitivity \
#     --feature "Fold_${FOLD}" \
#     --run_name "sensitivity" \
#     --run_id "1"
# done

# # Pathology ML baseline across folds
# python3 -u ULTR-AI/UltrAI/train_pathology_ml.py \
#   --features_csv ULTR-AI/benin_trust_workingfiles/CXR_Comp_Files/aiml_allcombined.csv \
#   --labels_csv ULTR-AI/UltrAI/data/labels/sensitivity_analysis_labels.csv \
#   --splits_dir ULTR-AI/UltrAI/data/Splits \
#   --folds "0,1,2,3,4" \
#   --output_dir ULTR-AI/benin_trust_workingfiles/predictions_path_sens_analysis \
#   --id_col record_id \
#   --target_col "TB Label"

# # Summarize TB + Pathology predictions into ULTR-AI MAX
# python3 -u ULTR-AI/UltrAI/summarize_results.py \
#   --tb_preds_dir ULTR-AI/UltrAI/results/TBPredictions_Sensitivity \
#   --path_preds_dir ULTR-AI/benin_trust_workingfiles/predictions_path_sens_analysis \
#   --folds "0,1,2,3,4" \
#   --output_dir ULTR-AI/benin_trust_workingfiles/predictions_path_sens_analysis