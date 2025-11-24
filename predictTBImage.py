

import pathlib
import ml_collections
import os
import torch
import wandb
import yaml
from torch.nn import BCEWithLogitsLoss
from utilities import config_utils, utils
from dataset_loading import dataset
from evaluation.metrics import compute_metrics, BestMetricVal
from evaluation.model_evaluation import model_evaluation, run_and_save_predictions
from network_architecture.deepchest import DeepChest

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

def print_gpu_memory():
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3)} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")


def get_config(params = None):
    config = ml_collections.ConfigDict()
    project_root = pathlib.Path(__file__).resolve().parent
    config.save_dir = str(project_root / "results" / "models")
    config.pred_save_dir = str(project_root / "results")
    config.feature = 'TB1'
    config.run_name = "run 1"
    config.run_id = '1'
    config.seed = 0
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.hyperparm_optim = True

    # optimizer
    config.learning_rate = 0.001
    config.weight_decay = 0.0
    config.batch_size = 32
    config.pos_weight = 1.8

    # dataset
    config.images_directory = str(project_root / "data" / "images")
    config.labels_file = str(project_root / "data" / "labels" / "justTB.csv")
    config.test_indices_file = str(project_root / "data" / "Splits" / "traintestvalidids.csv")

    config.num_folds = 5
    config.valid_fold_index = 0
    config.test_size = 0.2  # data is divided into train/val and test.
    config.num_classes = 2
    config.preprocessing_train_eval = ";"  # "independent_dropout(.2);"
    config.num_workers = 4
    config.export_folds_indices_file = config.save_dir + "indices.csv"
    # training

    config.model_weights = str(project_root / "results" / "models" / "1" / "checkpoint_best.pth")

    config.nb_epochs = 30
    config.eval_every_steps = 1
    config.eval_metric = "roc_auc"
    config.eval_metric_goal = "max"  # otherwise "min"
    config.evaluate_best_valid_model = True

    #config.num_classes = 1
    #0 if binary 1 if continuous 
    config.classification_type = 0


    # image representation network
    config.resnet = ml_collections.ConfigDict()
    config.resnet.pretrained = True
    config.resnet.pretrained_path = None
    config.resnet.freeze = False

    # aggregation network
    config.aggregation_type = "MLP_AttentionPooling"  # See Aggregation for possible values
    config.encoding_dim = 512
    config.use_positional_embeddings = True

    return config

def get_data_loaders(config):

    labels_df = pd.read_csv(config.labels_file)
    full_indices = labels_df['record_id'].values.flatten()
    full_labels = labels_df['tb_rif_genexpert'].values.flatten()

    data = pd.read_csv(config.test_indices_file)
    data = data.applymap(lambda x: x.replace('25-', '') if isinstance(x, str) else x)
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


    k = [v for v in valid_ids if v in train_ids]
    y = [v for v in test_ids if v in train_ids]
    y = k + y
    print("Leakage = ", y)

    labels_dict = dict(zip(full_indices, full_labels))

    if config.export_folds_indices_file:
        serie_train = pd.Series(train_ids, name='train_indices')
        serie_test = pd.Series(test_ids, name='test_indices')
        serie_valid = pd.Series(valid_ids, name='validation_indices')
        df_indices = pd.concat([serie_train, serie_test, serie_valid], axis=1)
        df_indices.to_csv(config.export_folds_indices_file, index=False)

    transform_vanilla = transforms.Compose(
        [transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )

    transform_with_augmentation = transforms.Compose(
        [transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomResizedCrop(
             224,
             scale=(0.75, 0.95),
             ratio=(0.75, 1.3333333333333333),
             interpolation=InterpolationMode.BICUBIC,
         ),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
         ]
    )

    dataset = LungUltrasoundPatientDataset(
        images_directory=config.images_directory,
        labels=labels_dict,
    )

    print(labels_dict)
    print(train_ids)

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
    print("Configuration:", config)
    config.save_dir = str(pathlib.Path(config.save_dir).resolve())
    config.pred_save_dir = str(pathlib.Path(config.pred_save_dir).resolve())
    print(config.save_dir)
    utils.set_seed(config.seed)
    save_dir = pathlib.Path(config.save_dir) / config.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Made Directory")
    config.export_folds_indices_file = os.path.join(save_dir, config.run_name + "_indices.csv")
    model_save_dir = os.path.join(save_dir, config.run_name + "bestmodel.pth")
    config.lock()
    print(f"Run name={config.run_name} | id={config.run_id} | directory={save_dir.resolve()} | modelsavepath = {model_save_dir}")

    config_file = save_dir / "config.yaml"
    config_file.unlink(missing_ok=True)
    config_file.write_text(yaml.dump(config.to_dict()))
    utils.print_config(config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Running on device {torch.cuda.get_device_name(0)}. Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("Running on CPU.")

    train_loader, test_loader, validation_loader = get_data_loaders(config)

    model = DeepChest(config)
    model.to(config.device)

    checkpoint = torch.load(config.model_weights)
    model.load_state_dict(checkpoint['model_state'])

    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(config.pos_weight))
    criterion.to(config.device)

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
    config = config_utils.parse_cli_overides(get_config())
    main(config)



#python3 /home/tjb76/TBLUScopy/predictTBImage.py 

