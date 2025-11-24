"""
Main script to train DeepChest for pathology classification
===========================================================

This script trains a DeepChest model to classify various lung ultrasound (LUS) pathology patterns
from individual LUS images. Unlike trainTB.py which classifies at the patient level, this script
works with individual images and can classify multiple pathology features such as:
- A-lines
- B-lines (3+ per field)
- Coalescing B-lines
- Small consolidations and nodules
- Large consolidations
- Pleural effusion

The script supports training on different folds and can be run with different pathology features
via command-line arguments.

:Author: Intelligent Global Health Research Group, EPFL
:Date: 2023-01-08
:Copyright: Copyright (C) 2023 Intelligent Global Health Research Group, EPFL
:License: Apache License 2.0
"""

import sys
import json
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
import pathlib
import ml_collections
import os
import torch
import wandb
import yaml
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from utilities import config_utils, utils
from dataset_loading import datasetsingle
from evaluation.metrics import compute_metrics, BestMetricVal
from evaluation.model_evaluation import model_evaluation, run_and_save_predictions
from network_architecture.deepchest import DeepChest, FocalLoss
import re
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import InterpolationMode
from deepchest.dataset_loading.dataset import LungUltrasoundPatientDataset, collate_fn, seed_worker
from deepchest.dataset_loading import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import functools
from torch.cuda.amp import autocast, GradScaler
import gc
from dataclasses import dataclass
from deepchest.dataset_loading.newdatasets import UltrasoundImageDataset, collate_fn1, create_labels


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


# def get_config(params = None):
#     config = ml_collections.ConfigDict()
    
#     config.run_name = "A-lines"
#     config.run_id = '1'
#     config.feature = 'A-lines'
#     config.seed = 0
#     config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config.hyperparm_optim = True
#     config.pos_weight = float(4)
    
#     if params:
#         for key, value in params.items():
#             if key == 'pos_weight':
#                 # Convert and set pos_weight safely
#                 config.pos_weight = float(value)
#             else:
#                 setattr(config, key, value)


#     config.learning_rate = 0.00001
#     config.weight_decay = 0.0
#     config.batch_size = 32
#     config.num_frames = 1
    


#     # dataset
#     #config.indices = 'record_id'
#     # pathology_features = ['Normal A line Pattern', 'Pattern with coalescing B lines ', 'Pattern with small consolidations and/or subpleural nodules (< 1cm in height)',
#     #                   'Pattern with ≥ 3 B lines per field', 'Pleural effusion', ]
#     # model_names = ['Normal A line Pattern', 'Pattern with coalescing B lines ', 'Pattern with small consolidations andor subpleural nodules (< 1cm in height)',
#     #                  'Pattern with ≥ 3 B lines per field', 'Pleural effusion']

#     LABELS = ['A-lines', '3 B-lines', 'Coalescing B-lines', 'Small consolidations and or nodules', 'Large Consolidations','Pleural effusion']

#     model_names = LABELS

#     if hasattr(config, 'name'):
#         config.model_name = config.name
    
#     if hasattr(config, 'feature'):
#         config.label = config.feature
#     else:
#         config.model_name = "Default_Model"
#         config.label = "Default_Label"

    
#     #config.labels_path = '/home/tjb76/TBLUScopy/Datasets/Labels/Benin/expertimagedf.csv'
#     config.labels_path = '/home/tjb76/TBLUScopy/Datasets/Labels/Benin/imagedf.csv'
#     #config.labels_path = '/home/tjb76/TBLUScopy/Datasets/Labels/BlindSweeps/ImageDFStandard.csv'
#     # config.save_dir = os.path.join("/home/tjb76/TBLUScopy/BlindSweeps/models", config.model_name)
#     # config.pred_save_dir = '/home/tjb76/TBLUScopy/BlindSweeps/predictions'
#     # config.test_indices_file= '/home/tjb76/TBLUScopy/Datasets/Labels/BlindSweeps/Splits/StandardVideos/Fold_1.csv'
#     config.save_dir = os.path.join("/home/tjb76/TBLUScopy/benin_trust_workingfiles/Path_Strat_Folds/models", config.model_name)
#     config.pred_save_dir = os.path.join("/home/tjb76/TBLUScopy/benin_trust_workingfiles/Path_Strat_Folds/Preds", config.model_name)
#     config.test_indices_file= '/home/tjb76/TBLUScopy/benin_trust_workingfiles/Splits_stratified/Fold_0.csv'


#     config.num_folds = 5
#     config.valid_fold_index = 1
#     config.test_size = 0.2  # data is divided into train/val and test.
#     config.num_classes = 2
#     config.preprocessing_train_eval = ";"  # "independent_dropout(.2);"
#     config.num_workers = 4
#     config.export_folds_indices_file = config.save_dir + "indices.csv"

#     # training
#     config.nb_epochs = 25
#     config.eval_every_steps = 1
#     config.eval_metric = "roc_auc"
#     config.eval_metric_goal = "max"  # otherwise "min"
#     config.evaluate_best_valid_model = True
#     config.accumulation_steps = 2

#     # image representation network
#     config.resnet = ml_collections.ConfigDict()
#     config.resnet.pretrained = True
#     config.resnet.pretrained_path = None
#     config.resnet.freeze = True

#     # aggregation network
#     config.aggregation_type = "MLP_AttentionPooling"  # See Aggregation for possible values
#     config.encoding_dim = 512
#     config.use_positional_embeddings = True
#     config.pooling = False


    # return config



def get_config(params=None):
    """
    Create and return the configuration dictionary for pathology classification training.
    
    This function sets up all hyperparameters, paths, and model architecture settings for
    training models to classify individual pathology features in LUS images. The configuration
    can be customized via the params dictionary (typically passed from command-line JSON).
    
    Args:
        params (dict, optional): Dictionary of configuration overrides. Common keys:
            - 'feature': Pathology feature to classify (e.g., 'A-lines', 'B-lines')
            - 'fold': Cross-validation fold number (0-4)
            - 'run_id': Unique identifier for this training run
            - 'pos_weight': Class weight for positive class (handles class imbalance)
            - 'learning_rate': Learning rate for optimizer
            - 'batch_size': Batch size for training
            
    Returns:
        ml_collections.ConfigDict: Configuration object with all training parameters
        
    Configuration Sections:
        - Paths: Directories for saving models and predictions (dynamically set based on feature/fold)
        - Optimizer: Learning rate, weight decay, batch size, class weights
        - Dataset: Image labels file, train/val/test splits
        - Training: Number of epochs, evaluation metrics, early stopping
        - Model: ResNet backbone, aggregation method, encoding dimensions
        
    Example:
        >>> config = get_config({'feature': 'A-lines', 'fold': 0, 'pos_weight': 1.6})
        >>> print(config.model_name)  # 'A-lines_Fold_0'
    """
    config = ml_collections.ConfigDict()

    # ========== Default Configuration Values ==========
    # Run identifier and feature name
    config.run_name = "A-lines"
    config.run_id = '1'
    config.feature = 'A-lines'  # Pathology feature to classify
    config.fold = 0  # Default fold is 0, can be overridden by params
    
    # General training settings
    config.seed = 0
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.hyperparm_optim = True  # Flag for hyperparameter optimization
    
    # Class imbalance handling: positive class weight
    # Higher values give more weight to positive examples (useful for imbalanced datasets)
    config.pos_weight = float(4)

    # ========== Override Configuration with Params ==========
    # This allows dynamic configuration via command-line JSON arguments
    if params:
        for key, value in params.items():
            if key == 'pos_weight':
                # Ensure pos_weight is always a float
                config.pos_weight = float(value)
            else:
                setattr(config, key, value)

    # ========== Dynamic Path Configuration ==========
    # Generate model name and paths based on feature and fold
    # Example: "A-lines_Fold_0" or "Large Consolidations_Fold_2"
    config.model_name = f"{config.feature}_Fold_{config.fold}"
    
    # Path to CSV file containing image-level labels
    # Expected columns: 'path' (image path), and feature columns (e.g., 'A-lines', 'B-lines')
    project_root = pathlib.Path(__file__).resolve().parent
    config.labels_path = str(project_root / "data" / "labels" / "imagedf.csv")
    
    # Directory to save model checkpoints (organized by feature and fold)
    config.save_dir = os.path.join(str(project_root / "results" / "Path_Strat_Folds" / "models"), config.model_name)
    
    # Directory to save prediction CSV files (organized by feature and fold)
    config.pred_save_dir = os.path.join(str(project_root / "results" / "Path_Strat_Folds" / "Preds"), config.model_name)
    
    # Path to CSV file containing train/validation/test split indices for this fold
    # CSV should have columns: 'train_ids', 'valid_ids', 'test_ids'
    config.test_indices_file = str(project_root / "data" / "Splits" / f"Fold_{config.fold}.csv")

    # ========== Optimizer Configuration ==========
    config.learning_rate = 0.00001  # Lower learning rate than TB training (fine-tuning)
    config.weight_decay = 0.0  # L2 regularization
    config.batch_size = 32  # Larger batch size for image-level training
    config.num_frames = 1  # Number of frames per sample (1 for single images)
    
    # ========== Cross-Validation Configuration ==========
    config.num_folds = 5  # Total number of folds
    config.valid_fold_index = 1  # Which fold to use for validation
    config.test_size = 0.2  # Fraction for test set (if not using pre-defined splits)
    config.num_classes = 2  # Binary classification for each pathology feature
    config.preprocessing_train_eval = ";"  # No preprocessing (empty string)
    config.num_workers = 4  # Data loading workers
    config.export_folds_indices_file = config.save_dir + "indices.csv"  # Save split indices
    
    # ========== Training Loop Configuration ==========
    config.nb_epochs = 25  # Fewer epochs than TB training (image-level is faster)
    config.eval_every_steps = 1  # Evaluate every epoch
    config.eval_metric = "roc_auc"  # Metric for model selection
    config.eval_metric_goal = "max"  # Maximize ROC-AUC
    config.evaluate_best_valid_model = True  # Evaluate best model on all splits
    config.accumulation_steps = 2  # Gradient accumulation (effective batch size = 32 * 2 = 64)

    # ========== Image Representation Network (ResNet Backbone) ==========
    config.resnet = ml_collections.ConfigDict()
    config.resnet.pretrained = True  # Use ImageNet pretrained weights
    config.resnet.pretrained_path = None  # No custom pretrained weights
    config.resnet.freeze = True  # Freeze ResNet, only train aggregation/classifier
    
    # ========== Aggregation Network Configuration ==========
    # Aggregation method for combining features (though pooling=False means single images)
    config.aggregation_type = "MLP_AttentionPooling"
    config.encoding_dim = 512  # Feature dimension after ResNet
    config.use_positional_embeddings = True  # Use site embeddings (if available)
    config.pooling = False  # No pooling needed for single images

    return config

def get_data_loaders(config):
    """
    Load and prepare data loaders for pathology classification training.
    
    This function:
    1. Loads image-level labels from CSV file
    2. Filters to only include images with valid labels for the target feature
    3. Loads train/val/test split indices from CSV file
    4. Creates balanced datasets (for training) using class balancing
    5. Creates PyTorch datasets and data loaders with appropriate transforms
    
    Args:
        config (ml_collections.ConfigDict): Configuration object with dataset paths and parameters
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader, train_loader_no_aug)
            - train_loader: DataLoader for training with data augmentation
            - valid_loader: DataLoader for validation (no augmentation)
            - test_loader: DataLoader for testing (no augmentation)
            - train_loader_no_aug: DataLoader for training set without augmentation (for final predictions)
            
    Raises:
        FileNotFoundError: If label file or split file doesn't exist
        KeyError: If required columns are missing from CSV files
    """
    # Load image-level labels from CSV file
    # Expected columns: 'path' (image file path), and feature columns (e.g., 'A-lines', 'B-lines')
    # Drop rows where either path or the target feature label is missing
    labels = pd.read_csv(config.labels_path).dropna(subset=['path', config.feature]).reset_index(drop=True)
    
    # Load train/validation/test split indices from CSV file
    # Expected columns: 'train_ids', 'valid_ids', 'test_ids'
    data = pd.read_csv(config.test_indices_file)

    # Extract patient IDs for each split (these are patient-level IDs, not image IDs)
    train_ids = data['train_ids'].dropna().tolist()
    valid_ids = data['valid_ids'].dropna().tolist()
    test_ids = data['test_ids'].dropna().tolist()

    if config.export_folds_indices_file:
        serie_train = pd.Series(train_ids, name='train_ids')
        serie_test = pd.Series(test_ids, name='test_ids')
        serie_valid = pd.Series(valid_ids, name='valid_ids')
        df_indices = pd.concat([serie_train, serie_test, serie_valid], axis=1)
        df_indices.to_csv(config.export_folds_indices_file, index=False)

    # ========== Image Transforms ==========
    # Transform for validation/test: minimal processing, no augmentation
    # Note: Images are expected to already be 224x224 (resize commented out)
    transform_vanilla = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),  # Uncomment if images need resizing
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                            std=[0.229, 0.224, 0.225])
    ])

    # Transform for training: extensive data augmentation to improve generalization
    # Includes geometric and photometric augmentations
    transform_with_augmentation = transforms.Compose([
        # Geometric augmentations (applied before converting to tensor)
        transforms.RandomRotation(degrees=15),  # Random rotation up to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        # Random crop with resize (simulates different viewing angles/distances)
        transforms.RandomResizedCrop(
            224,
            scale=(0.75, 0.95),  # Crop 75-95% of image
            ratio=(0.75, 1.3333333333333333)  # Aspect ratio range
        ),
        # Photometric augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # Additional augmentations (applied after tensor conversion)
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Random blur
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Random erasing
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #print(len(train_ids), len(valid_ids), len(test_ids))

    # Create labels and image paths for each split
    # create_labels filters labels by patient IDs and returns (labels, image_paths)
    # balance_classes=True: balances positive/negative examples in training set
    # target_count=6: target number of samples per class (for balancing)
    y_train, X_train = create_labels(labels, config.feature, ids=train_ids, balance_classes=True, target_count=6)
    # Validation and test sets: no balancing (use all available data)
    y_valid, X_valid = create_labels(labels, config.feature, ids=valid_ids)
    y_test, X_test = create_labels(labels, config.feature, ids=test_ids)

    print(len(y_train), len(y_valid), len(y_test))

    train_dataset = UltrasoundImageDataset(y_train,
                                            X_train, 
                                            num_frames=config.num_frames,
                                            transform=transform_with_augmentation,
                                            balance_classes=True
                                           # transform=transform_vanilla
                                            )


    val_dataset = UltrasoundImageDataset(y_valid,
                                            X_valid, 
                                            num_frames=config.num_frames,
                                            transform=transform_vanilla
                                            )


    test_dataset = UltrasoundImageDataset(y_test,
                                            X_test, 
                                            num_frames=config.num_frames,
                                            transform=transform_vanilla
                                            )
    train_dataset_noaug = UltrasoundImageDataset(y_train,
                                            X_train, 
                                            num_frames=config.num_frames,
                                             transform=transform_vanilla
                                            )

    g = torch.Generator()
    g.manual_seed(42)                                       

    train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn1,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True
        )

    test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn1,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True
        )

    valid_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn1,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True
        )
    train_loader_no_aug = DataLoader(
            train_dataset_noaug,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn1,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True
        )

    return train_loader, valid_loader, test_loader, train_loader_no_aug

def main(config):
    """
    Main training function for pathology classification model.
    
    This function orchestrates the entire training process for pathology feature classification:
    1. Sets up directories and saves configuration
    2. Loads data and creates data loaders (with class balancing for training)
    3. Initializes model, optimizer, and loss function
    4. Runs training loop with validation
    5. Saves best model checkpoints (including special checkpoints for high sensitivity/AUC)
    6. Evaluates best model on train/val/test sets and saves predictions
    
    Args:
        config (ml_collections.ConfigDict): Configuration object with all training parameters
        
    The function will:
        - Create output directories if they don't exist
        - Save configuration to YAML file for reproducibility
        - Train model for config.nb_epochs epochs
        - Save best model based on validation metric
        - Save additional checkpoints if model achieves high sensitivity or AUC
        - Generate prediction CSV files for all splits
    """
    print("Configuration:", config)
    # Resolve paths to absolute paths for consistency
    config.save_dir = str(pathlib.Path(config.save_dir).resolve())
    config.pred_save_dir = str(pathlib.Path(config.pred_save_dir).resolve())
    print(f"Model save directory: {config.save_dir}")
    
    # Set random seed for reproducibility
    utils.set_seed(config.seed)
    
    # Create output directories
    save_dir = pathlib.Path(config.save_dir) / config.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_save_dir = pathlib.Path(config.pred_save_dir)
    pred_save_dir.mkdir(parents=True, exist_ok=True)  # Create predictions save directory

    print(f"Saving models to: {save_dir.resolve()}")
    print(f"Saving predictions to: {pred_save_dir.resolve()}")
    print("Created directories")
    
    # Set paths for saving indices and model checkpoints
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
    train_loader, validation_loader, test_loader, final_predictions_train_loader = get_data_loaders(config)
    model = DeepChest(config)
    model.to(config.device)
    #pretrained
    #checkpoint = torch.load('/home/tjb76/TBLUScopy/deepchest/PathologyModels2/Large Consolidations/3/highauroclowerloss.pth')
    #model.load_state_dict(checkpoint['model_state'])
    # checkpoint = torch.load('model_saved/first_try/nfol=5_fol=0_agg=MLP_AttentionPooling_dec=0.0_lr=0.01_posweight=1.8_posenc=True/checkpoint_best.pth')
    # model.load_state_dict(checkpoint['model_state'])

    
    # Set up optimization
    optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
    #criterion = FocalLoss(gamma=2, pos_weight=torch.tensor(config.pos_weight))
    #criterion = CrossEntropyLoss(weight=config.pos_weight)
    criterion.to(config.device)

    # Set up best eval metric tracker
    best_eval_metric = BestMetricVal(config.eval_metric_goal)
    epochs_since_improvement = 0
    early_stopping_patience = 4
    best_valid_loss = 10
    scaler = GradScaler()
    accumulation_steps = 4

    # Training loop
    for epoch in range(config.nb_epochs):
        model.train()
        running_loss = 0
        all_targets = []
        all_logits = []
       
        for i, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):

            #move data to device
            labels = batch['label'].to(config.device).unsqueeze(1).float()
            # Move data to device
            batch["images"] = batch["images"].to(config.device)
            batch["sites"] = batch["sites"].to(config.device)
            batch["mask"] = batch["mask"].to(config.device)
            batch["label"] = batch["label"].to(config.device)

            try:
                with autocast():
                    # Forward pass
                    scores, attention = model(batch["images"], batch["sites"], batch["mask"])
                    loss = criterion(scores, labels)
                    loss = loss / config.accumulation_steps
                    running_loss += loss.item()
                    #scores = torch.flatten(scores)
                    #logit_l.extend([*scores.detach().cpu().numpy()])
                all_targets.append(labels.detach())
                all_logits.append(scores.detach())

                scaler.scale(loss).backward()

                if (i + 1) % config.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                torch.cuda.empty_cache()  # Free up cache after each batch
                del batch, loss
                gc.collect()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_gpu_memory()
                    print('WARNING: ran out of memory, skipping batch')
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            #Forward Pass
            #scores, _ = model(batch["images"], batch["sites"], batch["mask"])

            #Filter out padded data from calculations
            #masked_indices = batch["mask"].bool()
            #valid_scores = scores[masked_indices]
            #valid_labels = batch["label"][masked_indices]

            #Computer loss 
            # loss = criterion(valid_scores, valid_labels.float())
            # running_loss += loss.item()


            # all_targets.append(valid_labels.detach())
            # all_logits.append(valid_scores.detach())

            # Backward and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
        scheduler.step()
        torch.cuda.empty_cache()  

        #scheduler.step()

        # Compute metrics on all batches in the current epoch
        all_targets = torch.cat(all_targets).cpu().numpy()  # Transfer to CPU and convert to numpy at the end
        all_logits = torch.cat(all_logits).cpu().numpy()

        train_epoch_metrics = compute_metrics(all_targets, all_logits)
        train_epoch_metrics['loss'] = running_loss / len(train_loader)

        print(f'Epoch: {epoch + 1}/{config.nb_epochs}')
        utils.log_metrics(f"Train stats", train_epoch_metrics)
        #wandb.log(utils.prefix_dict(train_epoch_metrics, "train/"), step=epoch)


        validation_epoch_metrics, logits = model_evaluation(model, validation_loader, criterion, config.device)
        utils.log_metrics("Valid stats", validation_epoch_metrics, color="green")

        if validation_epoch_metrics['loss'] < best_valid_loss:
            if best_eval_metric.append(validation_epoch_metrics[config.eval_metric]):
            #if validation_epoch_metrics['loss'] < best_valid_loss:
                print(
                    f"Saving new best performing model ({config.eval_metric}={best_eval_metric.value})."
                )
                # Save the best model
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

        if validation_epoch_metrics['sensitivity'] > 0.9 and validation_epoch_metrics['specificity'] > 0.8:
            checkpoint_name = 'highsens.pth'
            checkpoint = {
                    'nb_epochs_finished': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
            torch.save(checkpoint, save_dir / checkpoint_name)

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

    # Load the best model and compute test and valid metrics
    if config.evaluate_best_valid_model:
        checkpoint = torch.load(save_dir / "checkpoint_best.pth")
        model.load_state_dict(checkpoint['model_state'])

        train_save_dir = os.path.join(config.pred_save_dir, config.model_name + '_trainpredictions.csv')
        train_metrics, logit_l = run_and_save_predictions(final_predictions_train_loader, model, config.device, criterion, train_save_dir)
        utils.log_metrics("Best model train stats", train_metrics,color="green")


        valid_save_dir = os.path.join(config.pred_save_dir, config.model_name + '_validpredictions.csv')
        valid_metrics, logit_l = run_and_save_predictions(validation_loader, model, config.device, criterion, valid_save_dir)
        utils.log_metrics("Best model valid stats", valid_metrics,color="green")

        test_save_dir = os.path.join(config.pred_save_dir, config.model_name + '_testpredictions.csv')
        test_metrics, logit_l = run_and_save_predictions(test_loader, model, config.device, criterion, test_save_dir)
        utils.log_metrics("Best model test stats", test_metrics, color="green")
        

if __name__ == "__main__":
    """
    Entry point for pathology training script.
    
    Usage:
        # Basic usage with default parameters
        python train_pathology.py
        
        # With JSON configuration (recommended)
        python train_pathology.py '{"feature": "A-lines", "fold": 0, "run_id": "1", "pos_weight": 1.6}'
        
        # Train different pathology features
        python train_pathology.py '{"feature": "A-lines", "fold": 0, "pos_weight": 1.6}'
        python train_pathology.py '{"feature": "B-lines", "fold": 0, "pos_weight": 1.8}'
        python train_pathology.py '{"feature": "Large Consolidations", "fold": 0, "pos_weight": 3}'
        python train_pathology.py '{"feature": "Small consolidations and or nodules", "fold": 0, "pos_weight": 3}'
        python train_pathology.py '{"feature": "Pleural effusion", "fold": 0, "pos_weight": 1.8}'
        
        # Train on different folds
        python train_pathology.py '{"feature": "A-lines", "fold": 1, "pos_weight": 1.6}'
        python train_pathology.py '{"feature": "A-lines", "fold": 2, "pos_weight": 1.6}'
    
    Parameters (JSON format):
        - feature: Pathology feature to classify (required)
        - fold: Cross-validation fold number (0-4, default: 0)
        - run_id: Unique identifier for this run (default: '1')
        - pos_weight: Positive class weight for handling imbalance (default: 4.0)
        - learning_rate: Learning rate (default: 0.00001)
        - batch_size: Batch size (default: 32)
        - nb_epochs: Number of training epochs (default: 25)
    
    Output:
        - Model checkpoints saved to: {save_dir}/{model_name}/{run_id}/
        - Predictions saved to: {pred_save_dir}/{model_name}/
        - Configuration saved as: config.yaml
    """
    # Parse command-line JSON arguments if provided
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
    else:
        params = {}
    
    # Get configuration (with overrides from params)
    config = get_config(params)
    
    # Run training
    main(config)



#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run1", "feature": "A-lines", "run_id": "1", "pos_weight": 1}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "B-lines", "run_id": "2", "pos_weight": 4}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Small consolidations and or nodules", "run_id": "2", "pos_weight": 3}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Large Consolidations", "name": "Large Consolidations_Fold_0", "run_id": "3", "pos_weight": 3}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Pleural effusion", "run_id": "2", "pos_weight": 3}'


#Fold0
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Large Consolidations", "name": "Large Consolidations_Fold_0", "run_id": "3", "pos_weight": 3}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "A-lines", "name": "A-lines_Fold_0", "run_id": "3", "pos_weight": 1.6}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Small consolidations and or nodules", "name": "Small consolidations and or nodules_Fold_0", "run_id": "3", "pos_weight": 3}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "B-lines", "name": "B-lines_Fold_0", "run_id": "3", "pos_weight": 1.8}'
#python3 /home/tjb76/TBLUScopy/train_pathologyv2.py '{"run_name": "Run2", "feature": "Pleural effusion", "name": "Pleural effusion_Fold_0", "run_id": "3", "pos_weight": 1.8}'
#
