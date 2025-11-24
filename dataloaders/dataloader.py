import os
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader
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
from deepchest.dataset_loading import preprocessing
from deepchest.utilities import utils
import re
from PIL import Image



SITE_MAPPING = OrderedDict(
    [
        ("<PAD>", 0),
        ("QAID", 1),
        ("QAIG", 2),
        ("QASD", 3),
        ("QASG", 4),
        ("QLD", 5),
        ("QLID", 5)
        ("QLG", 6),
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


def create_labels_old(labels_df, feature):
    indices = labels_df['record_id']
    sites = [ 'QAID', 'QAIG', 'QASD', 'QASG', 'QLD', 'QLG', 'QPID', 'QPIG', 'QPSD', 'QPSG', 'APXD', 'APXG', 'QSLD', 'QSLG']
    labels_dict = {}
    pathology_labels = []
    indices = []

    for record_id in labels_df['record_id'].unique():
            for site in sites:
                key = f"{record_id}_{site}"
                label_column = f"{site}_{feature}"
                if label_column in labels_df.columns:
                    value = labels_df.loc[labels_df['record_id'] == record_id, label_column].values
                    if len(value) > 0:
                        labels_dict[key] = value[0]  # Assuming there's only one value per pair
                        pathology_labels.append(value[0])
                        indices.append(record_id)
                    else:
                        labels_dict[key] = 0  # Default to 0 if no value found
                        pathology_labels.append(0)
                        indices.append(record_id)
                else:
                    labels_dict[key] = 0  # Default to 0 if column doesn't exist
                    pathology_labels.append(0)
                    indices.append(record_id)
    return labels_dict, pathology_labels, indices


def create_labels(labels_df, feature, ids = None):
    df = labels_df
    if ids:
        df = labels_df[labels_df['record_id'].isin(ids)]
    
    indices = df['record_id']
    #df['key'] = df['record_id'] + '_' + df['site'] + '_' + df['video_count'].fillna(0).astype(int).astype(str)
    df['key'] = df['record_id'] + '_' + df['site'] + '_' + df['count'].fillna(0).astype(int).astype(str)
    labels = df[['key', feature]]
    #media = df[['key', 'video_path']]
    media = df[['key', 'video_path']]
    return labels, media


@dataclass
class ImageInfo:
    patient_id: str
    site: str
    video_number: int = 0
    
    @staticmethod
    def from_filename(filename: str):
        match = re.match(r"^(\d+)_([A-Z]+)(_\d+)?\.mp4$", filename)
        if match is None:
            raise ValueError(f"Could not parse '{filename}' into an ImageInfo.")
        patient_id = int(match.group(1))
        site = match.group(2)
        video_number = int(match.group(3)[1:]) if match.group(3) else 0
        return ImageInfo(
            patient_id,
            site,
            video_number,
        )


class UltrasoundVideoDataset(Dataset):
    def __init__(self, video_directory, labels_dict, num_frames, transform=None, by_patient=False, path_feature=None, patient_ids=None, by_sites=None, dataset_id=None):
        self.video_dir = video_directory
        self.labels_dict = {f"{dataset_id}_{k}": v for k, v in labels_dict.items()}  # Prefix keys with dataset_id
        self.num_frames = num_frames
        self.video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
        self.patients = defaultdict(list)
        self.transform = transform
        self.by_patient = by_patient
        self.name = path_feature
        self.dataset_id = dataset_id  # Added dataset_id

        if patient_ids is None:
            self.patient_ids = list(self.patients.keys())
        else:
            self.patient_ids = patient_ids

        if not self.by_patient:
            self.video_files = [
                video_file for video_file in self.video_files
                if ImageInfo.from_filename(video_file).patient_id in self.patient_ids
            ]

        if by_sites:
            self.video_files = [video_file for video_file in self.video_files
                                if ImageInfo.from_filename(video_file).site in by_sites]

        for video_file in self.video_files:
            info = ImageInfo.from_filename(video_file)
            self.patients[info.patient_id].append(video_file)

        self.max_videos = max(len(videos) for videos in self.patients.values())

    def __len__(self):
        return len(self.patient_ids) if self.by_patient else len(self.video_files)

    def get_sites_counter(self) -> Counter:
        return Counter([ImageInfo.from_filename(v).site for v in self.video_files])
    
    def get_labels_split(self) -> Counter:
        label_counts = Counter()
        for key, label in self.labels_dict.items():
            label_counts[label] += 1
        return label_counts
    
    def __getitem__(self, idx):
        if self.by_patient:
            patient_id = self.patient_ids[idx]
            video_files = self.patients[patient_id]
            label = self.labels_dict.get(f"{self.dataset_id}_{patient_id}", -1)

            if len(video_files) < self.max_videos:
                padding = [None] * (self.max_videos - len(video_files))
                video_files.extend(padding)
        else:
            video_file = self.video_files[idx]
            video_files = [video_file]
            info =  ImageInfo.from_filename(video_file)
            label_key = f"{self.dataset_id}_{info.patient_id}_{info.site}"
            label = self.labels_dict.get(label_key, -1)

        video_frames_list = []
        optical_flow_list = []
        edge_list = []
        sites_list = []
        ids = []
        counts = []
        masks = []

        for video_file in video_files:
            if video_file is None:
                frames = torch.zeros((1, self.num_frames, 224, 224), dtype=torch.float32)
                optical_flow_frames = torch.zeros((1, self.num_frames, 224, 224, 2), dtype=torch.float32)
                edge_frames = torch.zeros((1, self.num_frames, 224, 224, 1), dtype=torch.float32)
                masks.append(0)
                sites_list.append(0)
                ids.append(0)
                counts.append(0)
            else:
                video_path = os.path.join(self.video_dir, video_file)
                info = ImageInfo.from_filename(video_file)
                cap = cv2.VideoCapture(video_path)
                frames = []
                optical_flow_frames = []
                edge_frames = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

                prev_frame_gray = None

                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (224, 224))  
                        frame_resized = frame_resized.ToTensor(),
                        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
                        frame_edge = cv2.Canny(frame_gray, 100, 200)

                        if prev_frame_gray is not None:
                            optical_flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        else:
                            optical_flow = np.zeros((224, 224, 2))

                        prev_frame_gray = frame_gray

                        if self.transform:
                            frame_resized = Image.fromarray(frame_resized)
                            frame_resized = self.transform(frame_resized)
                           
                        frames.append(frame_resized)
                        optical_flow_frames.append(optical_flow)
                        edge_frames.append(frame_edge)
                cap.release()

                frames = torch.stack(frames) 
                frames = frames.permute(1, 0, 2, 3)
                optical_flow_frames = torch.stack([torch.tensor(flow) for flow in optical_flow_frames])
                optical_flow_frames = optical_flow_frames.permute(1, 0, 2, 3)
                edge_frames = torch.stack([torch.tensor(edge) for edge in edge_frames])
                edge_frames = edge_frames.unsqueeze(1).permute(1, 0, 2, 3)
                masks.append(1)
                sites_list.append(info.site)
                ids.append(info.patient_id)
                counts.append(info.video_number)

            video_frames_list.append(frames)
            optical_flow_list.append(optical_flow_frames)
            edge_list.append(edge_frames)

        label = torch.tensor(label, dtype=torch.long).unsqueeze(1)
        sites_tensor = torch.tensor([SITE_MAPPING[site] for site in sites_list], dtype=torch.long)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        masks = torch.tensor(masks, dtype=torch.long)


        patient_dict = {
        "videos": video_frames_list,
        "optical_flow": optical_flow_list,
        "edges": edge_list,
        "sites": sites_tensor,
        "mask": masks,
        "label": label,
        "ids": ids_tensor,
        "dataset_id": self.dataset_id  # Include dataset_id in the returned dict
        }
        return patient_dict
    

def collate_fn(batch):
    batch_videos = [item['videos'] for item in batch]
    batch_optical_flows = [item['optical_flow'] for item in batch]
    batch_edges = [item['edges'] for item in batch]
    batch_labels = torch.stack([item['label'] for item in batch])
    batch_sites = torch.stack([item['sites'] for item in batch])
    batch_masks = torch.stack([item['mask'] for item in batch])
    batch_ids = torch.stack([item['ids'] for item in batch])
    batch_dataset_ids = [item['dataset_id'] for item in batch]

    batch_videos = torch.stack(batch_videos)
    batch_optical_flows = torch.stack(batch_optical_flows)
    batch_edges = torch.stack(batch_edges)

    return {
        'videos': batch_videos,
        'optical_flows': batch_optical_flows,
        'edges': batch_edges,
        'labels': batch_labels,
        'sites': batch_sites,
        'masks': batch_masks,
        'ids': batch_ids,
        'dataset_ids': batch_dataset_ids  # Include dataset_ids in the batch
    }


class MultiDatasetLoader:
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        cumulative_len = 0
        for dataset in self.datasets:
            if idx < cumulative_len + len(dataset):
                return dataset[idx - cumulative_len]
            cumulative_len += len(dataset)
        raise IndexError("Index out of range")

def create_dataloader_from_multiple_datasets(dataset_configs, batch_size, shuffle=True, num_workers=4):
    datasets = []
    for config in dataset_configs:
        dataset = UltrasoundVideoDataset(
            video_directory=config['video_directory'],
            labels_dict=config['labels_dict'],
            num_frames=config['num_frames'],
            transform=config.get('transform', None),
            by_patient=config.get('by_patient', False),
            path_feature=config.get('path_feature', None),
            patient_ids=config.get('patient_ids', None),
            by_sites=config.get('by_sites', None),
            dataset_id=config.get('dataset_id', None)  # Pass dataset_id to the dataset
        )
        datasets.append(dataset)

    multi_dataset = MultiDatasetLoader(datasets)
    return DataLoader(multi_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


# dataset_configs = [
#     {
#         'video_directory': 'path/to/dataset1',
#         'labels_dict': labels_dict1,
#         'num_frames': 16,
#         'transform': transforms.ToTensor(),
#         'dataset_id': 'dataset1'  # Unique identifier for the first dataset
#     },
#     {
#         'video_directory': 'path/to/dataset2',
#         'labels_dict': labels_dict2,
#         'num_frames': 16,
#         'transform': transforms.ToTensor(),
#         'dataset_id': 'dataset2'  # Unique identifier for the second dataset
#     }
# ]

# dataloader = create_dataloader_from_multiple_datasets(dataset_configs, batch_size=4)














# class UltrasoundVideoDataset(Dataset):
#     def __init__(self, 
#                  video_directory, 
#                  labels_dict,
#                  num_frames, 
#                  transform = None,
#                  by_patient = False,
#                  path_feature = None,
#                  patient_ids = None,
#                  by_sites = None,
#                  dataset_id=None):


#         self.video_dir = video_directory
#         self.labels_dict = labels_dict
#         self.num_frames = num_frames
#         self.video_files = [f for f in os.listdir(self.video_dir) if f.endswith('.mp4')]
#         self.patients = defaultdict(list)
#         self.transform = transform
#         self.by_patient = by_patient
#         self.name = path_feature
        
#         if patient_ids is None:
#             self.patient_ids = list(self.patients.keys())

#         else:
#             self.patient_ids = patient_ids

#         if not self.by_patient:
#             self.video_files = [
#                 video_file for video_file in self.video_files
#                 if ImageInfo.from_filename(video_file).patient_id in self.patient_ids
#             ]

#         if by_sites:
#             self.video_files = [video_file for video_file in self.video_files
#                                 if Image.from_filename(video_file).site in by_sites]

#         for video_file in self.video_files:
#             info = ImageInfo.from_filename(video_file)
#             self.patients[info.patient_id].append(video_file)

#         self.max_videos = max(len(videos) for videos in self.patients.values())

#     def __len__(self):
#         return len(self.patient_ids) if self.by_patient else len(self.video_files)

    
#     def get_sites_counter(self) -> Counter:
#         return Counter([ImageInfo.from_filename(v).site for v in self.video_files])
    
#     def get_labels_split(self) -> Counter:
#         label_counts = Counter()
#         for key, label in self.labels_dict.items():
#             label_counts[label] += 1
#         return label_counts
    
#     def __getitem__(self, idx):
#         if self.by_patient:
#             patient_id = self.patient_ids[idx]
#             video_files = self.patients[patient_id]
#             label = self.labels_dict.get(patient_id, -1)

#             if len(video_files) < self.max_videos:
#                 padding = [None] * (self.max_videos - len(video_files))
#                 video_files.extend(padding)
#         else:
#             video_file = self.video_files[idx]
#             video_files = [video_file]
#             info =  ImageInfo.from_filename(video_file)
#             label_key = f"{info.patient_id}_{info.site}"
#             label = self.labels_dict.get(label_key, -1)

#         video_frames_list = []
#         optical_flow_list = []
#         edge_list = []
#         sites_list = []
#         ids = []
#         counts = []
#         masks = []

#         for video_file in video_files:
#             if video_file is None:
#                 frames = torch.zeros((1, self.num_frames, 224, 224), dtype=torch.float32)
#                 optical_flow_frames = torch.zeros((1, self.num_frames, 224, 224, 2), dtype=torch.float32)
#                 edge_frames = torch.zeros((1, self.num_frames, 224, 224, 1), dtype=torch.float32)
#                 masks.append(0)
#                 sites_list.append(0)
#                 ids.append(0)
#                 counts.append(0)
#             else:
#                 video_path = os.path.join(self.video_dir, video_file)
#                 info = ImageInfo.from_filename(video_file)
#                 video_path = os.path.join(self.video_dir, video_file)
#                 info = ImageInfo.from_filename(video_file)
#                 cap = cv2.VideoCapture(video_path)
#                 frames = []
#                 optical_flow_frames = []
#                 edge_frames = []
#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

#                 for frame_idx in frame_indices:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#                     ret, frame = cap.read()
#                     if ret:
#                         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                         frame_resized = cv2.resize(frame_rgb, (224, 224))  
#                         frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
#                         frame_edge = cv2.Canny(frame_gray, 100, 200)

#                         if prev_frame_gray is not None:
#                             optical_flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#                         else:
#                             optical_flow = np.zeros((224, 224, 2))

#                         prev_frame_gray = frame_gray

#                         if self.transform:
#                             frame_resized = self.transform(frame_resized)
#                             frame_edge = self.transform(frame_edge)

#                         frames.append(frame_resized)
#                         optical_flow_frames.append(optical_flow)
#                         edge_frames.append(frame_edge)
#                 cap.release()

#                 frames = torch.stack(frames) 
#                 frames = frames.permute(1, 0, 2, 3)
#                 optical_flow_frames = torch.stack([torch.tensor(flow) for flow in optical_flow_frames])
#                 optical_flow_frames = optical_flow_frames.permute(1, 0, 2, 3)
#                 edge_frames = torch.stack([torch.tensor(edge) for edge in edge_frames])
#                 edge_frames = edge_frames.unsqueeze(1).permute(1, 0, 2, 3)
#                 masks.append(1)
#                 sites_list.append(info.site)
#                 ids.append(info.patient_id)
#                 counts.append(info.video_number)

#             video_frames_list.append(frames)
#             optical_flow_list.append(optical_flow_frames)
#             edge_list.append(edge_frames)

#         label = torch.tensor(label, dtype=torch.long) 
#         sites_tensor = torch.tensor([SITE_MAPPING[site] for site in sites_list], dtype=torch.long)
#         ids_tensor = torch.tensor(ids, dtype=torch.long)
#         masks = torch.tensor(masks, dtype=torch.long)


#         patient_dict = {
#         "videos": video_frames_list,
#         "optical_flow": optical_flow_frames,
#         "edges": edge_frames,
#         "sites": sites_tensor,
#         "mask": masks,
#         "label": label,
#         "ids": ids_tensor,
#         }
#         return patient_dict
    

# def collate_fn(batch):
#     batch_videos = [item['videos'] for item in batch]
#     batch_optical_flows = [item['optical_flow'] for item in batch]
#     batch_edges = [item['edges'] for item in batch]
#     batch_labels = torch.stack([item['label'] for item in batch])
#     batch_sites = torch.stack([item['sites'] for item in batch])
#     batch_masks = torch.stack([item['mask'] for item in batch])
#     batch_ids = torch.stack([item['ids'] for item in batch])

#     batch_videos = torch.stack(batch_videos)
#     batch_optical_flows = torch.stack(batch_optical_flows)
#     batch_edges = torch.stack(batch_edges)

#     return {
#         'videos': batch_videos,
#         'optical_flows': batch_optical_flows,
#         'edges': batch_edges,
#         'labels': batch_labels,
#         'sites': batch_sites,
#         'masks': batch_masks,
#         'ids': batch_ids
#     }


       


