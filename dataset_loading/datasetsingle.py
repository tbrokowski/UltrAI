import functools
import os
import pathlib
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.markdown import Markdown
from dataset_loading import preprocessing
from utilities import utils

SITE_MAPPING = OrderedDict(
    [
        ("<PAD>", 0),
        ("QAID", 1),
        ("QAIG", 2),
        ("QASD", 3),
        ("QASG", 4),
        ("QLID", 5),
        ("QLIG", 6),
        ("QPID", 7),
        ("QPIG", 8),
        ("QPSD", 9),
        ("QPSG", 10),
        ("APXD", 11),
        ("APXG", 12),
        ("QSLD", 13),
        ("QSLG", 14)
    ]
)

@dataclass
class ImageInfo:
    patient_id: int
    site: str

    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_([A-Z]+)(_\d+)?(\..*)?\.png$", filename)
        if match is None:
            raise ValueError(f"Could not parse '{filename}' into an ImageInfo.")
        patient_id = int(match.group(1))
        site = match.group(2)
        # if site == 'QLD':
        #     site = 'QLID'
        # if site == 'QLG':
        #     site = 'QLIG'
        return ImageInfo(
            patient_id,
            site,
        )


class LungUltrasoundPatientDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images_directory: str,
            labels: Optional[Dict[int, int]] = None,
    ):
        if not pathlib.Path(images_directory).exists():
            with open(pathlib.Path(__file__).parent / "../dataset/README.md") as md:
                markdown = Markdown(md.read())
            Console().print(markdown)
            raise FileNotFoundError("Images directory not found. Path given: " + images_directory)
        self.images_directory = pathlib.Path(images_directory)
        self.labels = labels

        self.images_list = os.listdir(self.images_directory)
        if labels:
            self.length = len(labels)
        else:
            # length = nb of patients
            self.length = len(set(ImageInfo.from_filename(f).patient_id for f in self.images_list))

    def __len__(self):
        return self.length

    def get_sites_counter(self) -> Counter:
        return Counter([ImageInfo.from_filename(f).site for f in self.images_list])
    
    def __getitem__(self, i):
        all_images = []
        all_labels = []
        all_sites = []
        all_ids = []
 
        image_names = [f for f in self.images_list if f.startswith(f"{i}_")]
        unique_sites = {ImageInfo.from_filename(f).site for f in image_names}  # Use a set to ensure unique sites
        num_patient_images = len(image_names)
        sites = [ImageInfo.from_filename(f).site for f in image_names]
        for site in unique_sites:
            key = f"{i}_{site}"
            if key in self.labels:
                image_filenames = [f for f in image_names if f.startswith(f"{i}_{site}")]
                for filename in image_filenames:
                    img = Image.open(self.images_directory / filename).convert("RGB")
                    img_tensor = transforms.ToTensor()(img)
                    label = torch.tensor(self.labels[key])
                    all_images.append(img_tensor)
                    all_labels.append(label)
                    all_sites.append(site)
                    all_ids.append(i)
                     
                #  images = [Image.open(self.images_directory / filename).convert("RGB") for filename in image_filenames]
                #  images = torch.stack([transforms.ToTensor()(img) for img in images])
                #  labels = torch.tensor(self.labels[key])
                #  all_images.append(images)
                #  all_labels.append(labels)
                #  all_sites.append(site)
                    
        mask = torch.ones(len(all_images))
        sites_tensor = torch.LongTensor([SITE_MAPPING[site] for site in all_sites])
        ids_tensor = torch.LongTensor([id for id in all_ids])
        


                

       # mask = torch.ones(num_patient_images)
        
        # -----------------------------------------
        # bias the dataset
        # sites_biased = []
        # if self.labels[i] == 1:
        #     for site in sites:
        #         if site == 'QPID':
        #             sites_biased.append('QASD')
        #         else:
        #             sites_biased.append(site)
        #     sites = sites_biased
        # --------------------------------------------

        # images = []
        # for x in image_names:
        #     image = Image.open(self.images_directory / x).convert("RGB")
        #     images.append(image)

       

        patient_dict = {
            "images": all_images,
            "sites": sites_tensor,
            "mask": mask,
            "label": all_labels,
            "ids": ids_tensor,
        }

        # if self.labels is not None:
        #     patient_dict["label"] = self.labels[i]

        return patient_dict

import torch
import torch.nn.functional as F
from torchvision import transforms

class LungUltraSoundImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_directory,
                 labels_dict,
                 patient_ids
                 ):
        self.image_dir = image_directory
        self.labels_dict = labels_dict
        self.patient_ids = patient_ids
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.patients = defaultdict(list)
        self.image_files = [image_file for image_file in self.image_files 
                            if ImageInfo.from_filename(image_file).patient_id in self.patient_ids
                            and f"{ImageInfo.from_filename(image_file).patient_id}_{ImageInfo.from_filename(image_file).site}" in self.labels_dict
                            ]
        
    def __len__(self):
        return len(self.image_files)
    
    def get_sites_counter(self) -> Counter:
        return Counter([ImageInfo.from_filename(v).site for v in self.image_files])
    
    def get_labels_split(self) -> Counter:
        label_counts = Counter()
        for key, label in self.labels_dict.items():
            label_counts[label] += 1
        return label_counts
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        info = ImageInfo.from_filename(image_file)
        id = info.patient_id
        site = info.site

        label_key = f"{info.patient_id}_{info.site}"

        img = Image.open(os.path.join(self.image_dir, image_file)).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        label = torch.tensor(self.labels_dict.get(label_key, 0))
        site_tensor = torch.tensor(SITE_MAPPING[site], dtype=torch.long)
        mask = torch.ones(1)
        ids_tensor = torch.tensor(id, dtype=torch.long)

        patient_dict = {
            "images": img_tensor,
            "sites": site_tensor,
            "mask": mask,
            "label": label,
            "ids": ids_tensor,
        }
        return patient_dict



def collage_image_fn(batch, image_transform, preprocessing_fn=None):
    batch_images = torch.stack([image_transform(item['images']) for item in batch])
    batch_labels = torch.stack([item['label'] for item in batch])
    batch_sites = torch.stack([item['sites'] for item in batch])
    batch_masks = torch.stack([item['mask'] for item in batch])
    batch_ids = torch.stack([item['ids'] for item in batch])
        
    if preprocessing_fn:
        batch_images, batch_sites, batch_masks = preprocessing_fn(batch_images, batch_sites, batch_masks)

    return {
        'images': batch_images,
        'label': batch_labels,
        'sites': batch_sites,
        'mask': batch_masks,
        'ids': batch_ids
    }
    
def collate_fn(batch, image_transform, preprocessing_fn=None):
    batch_images = []
    batch_labels = []
    batch_sites = []
    batch_masks = []
    batch_ids = []

    # Determine the maximum number of images per record in the batch
    max_images = 36

    for data in batch:
        images = data['images']
        if len(images) == 0:
            #print(len(images))
            continue
        # labels = [torch.tensor(label) for label in data['label']]  # Convert each label to tensor
        # sites = torch.tensor(data['sites'])  # Convert list to tensor
        # mask = torch.tensor(data['mask'])  # Convert list to tensor
        # ids = torch.tensor(data['ids'])
        labels = [label.clone().detach() for label in data['label']]  # Use clone().detach() for existing tensors
        sites = data['sites'].clone().detach() if torch.is_tensor(data['sites']) else torch.tensor(data['sites'])
        mask = data['mask'].clone().detach() if torch.is_tensor(data['mask']) else torch.tensor(data['mask'])
        ids = data['ids'].clone().detach() if torch.is_tensor(data['ids']) else torch.tensor(data['ids'])


        transformed_images = [image_transform(transforms.ToPILImage()(img) if isinstance(img, torch.Tensor) else img) for img in images]
        
        # Pad images to the maximum length in the batch
        if len(transformed_images) < max_images:
            padding = [torch.zeros_like(transformed_images[0]) for _ in range(max_images - len(transformed_images))]
            transformed_images.extend(padding)
            labels.extend([torch.zeros_like(labels[0]) for _ in range(max_images - len(labels))])
            sites = torch.cat([sites, torch.zeros((max_images - len(sites),), dtype=sites.dtype)])
            mask = torch.cat([mask, torch.zeros((max_images - len(mask),), dtype=mask.dtype)])
            ids = torch.cat([ids,torch.full((max_images - len(ids),), ids[-1], dtype=ids.dtype)])

        batch_images.append(torch.stack(transformed_images))
        batch_labels.append(torch.stack(labels))  # Stack the list of tensors into a single tensor
        batch_sites.append(sites)
        batch_masks.append(mask)
        batch_ids.append(ids)
       

    batch_images = torch.stack(batch_images)
    batch_labels = torch.stack(batch_labels)
    batch_sites = torch.stack(batch_sites)
    batch_masks = torch.stack(batch_masks)
    batch_ids = torch.stack(batch_ids)


    # Flatten the batches to achieve the desired shape
    batch_images = batch_images.view(-1, *batch_images.shape[2:])
    batch_labels = batch_labels.view(-1, *batch_labels.shape[2:]).unsqueeze(1)
    batch_sites = batch_sites.view(-1) #.unsqueeze(1)
    batch_masks = batch_masks.view(-1) #.unsqueeze(1)
    batch_ids = batch_ids.view(-1)

    if preprocessing_fn:
        batch_images, batch_sites, batch_masks = preprocessing_fn(batch_images, batch_sites, batch_masks)

    return {
        'images': batch_images,
        'label': batch_labels,
        'sites': batch_sites,
        'mask': batch_masks,
        'ids': batch_ids
    }

def collate_fn_patient(samples, image_transform, preprocessing_fn=None):
    samples_per_key = {key: [sample[key] for sample in samples] for key in samples[0]}

    batch = {}
    if "label" in samples_per_key:
        batch["label"] = torch.tensor(samples_per_key["label"])

    # Transform each image of site of each patient
    # and stack them together for each patient.
    samples_per_key["images"] = [
        torch.stack(
            [torch.FloatTensor(np.array(image_transform(im))) for im in patient_images], dim=0
        )
        for patient_images in samples_per_key["images"]
    ]

    # Pad with zeros and stack samples
    max_length = max(map(len, samples_per_key["mask"]))
    for k in ["images", "sites", "mask"]:
        batch[k] = torch.stack(
            [utils.pad_dim_with_zeros(t, 0, max_length) for t in samples_per_key[k]], dim=0
        )

    # Preprocessing
    if preprocessing_fn:
        batch["images"], batch["sites"], batch["mask"] = preprocessing_fn(
            batch["images"], batch["sites"], batch["mask"]
        )

    return batch





def get_data_loaders(config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    labels_df = pd.read_csv(config.labels_file, index_col=0)
    indices = labels_df.index
    labels = labels_df.values.flatten()
    labels_dict = dict(zip(indices, labels))

    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(indices, labels, shuffle=True,
                                                                                          test_size=config.test_size,
                                                                                          random_state=config.seed,
                                                                                          stratify=labels)

    # k fold over the train/valid part
    folds = utils.split_k_folds(train_val_indices, train_val_labels, k=config.num_folds)
    train_folds_indices = list(range(0, config.num_folds))
    train_folds_indices.remove(config.valid_fold_index)
    validation_indices = folds[config.valid_fold_index]
    train_folds = [fold for fold_idx, fold in enumerate(folds) if fold_idx in train_folds_indices]
    train_indices = np.concatenate(train_folds)
    
    #####
    if config.export_folds_indices_file:
        serie_train = pd.Series(train_indices, name='train_indices')
        serie_test = pd.Series(test_indices, name='test_indices')
        serie_valid = pd.Series(validation_indices, name='validation_indices')
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

    label_names = utils.get_label_names(config.labels_file)
    utils.show_splits_info(
        train_indices, test_indices, validation_indices, labels_dict, label_names=label_names
    )

    train_pp, eval_pp = config.preprocessing_train_eval.split(";")
    train_pp, eval_pp = map(preprocessing.make_preprocessing_fn, [train_pp, eval_pp])

    train_subset = Subset(dataset, train_indices)
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

    validation_subset = Subset(dataset, validation_indices)
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

    test_subset = Subset(dataset, test_indices)
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
