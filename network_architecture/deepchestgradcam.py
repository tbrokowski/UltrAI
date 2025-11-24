import torch
import torch.nn.functional as F
from torch import nn
import enum
import einops
import torch
import torch.nn as nn
#from dataset_loading import datasetsingle
from dataset_loading import dataset
import network_architecture as models
from .resnet import ResNet
from .convnext import ConvNeXtTiny

class Aggregation(str, enum.Enum):
    TRANSFORMER = "Transformer"
    DEEP_SET = "DeepSet"
    MAX_POOLING = "MaxPooling"
    MEAN_POOLING = "MeanPooling"
    ATTENTION_POOLING = "AttentionPooling"
    MLP_MAX_POOLING = "MLP_MaxPooling"
    MLP_MEAN_POOLING = "MLP_MeanPooling"
    MLP_ATTENTION_POOLING = "MLP_AttentionPooling"

class DeepChest(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet = ResNet(**config.resnet)
        self.encoding_dim = config.encoding_dim
        self.Pooling = config.pooling

        self.use_positional_embeddings = config.use_positional_embeddings
        if self.use_positional_embeddings:
            self.site_embedding = nn.Embedding(len(dataset.SITE_MAPPING), self.encoding_dim)

        # Aggregation
        self.aggregation = None
        self.aggregation_type = config.aggregation_type
        if self.aggregation_type == Aggregation.TRANSFORMER:
            self.aggregation = models.Transformer(config)
        elif self.aggregation_type == Aggregation.DEEP_SET:
            self.aggregation = models.DeepSet(features=self.encoding_dim)
        elif self.aggregation_type == Aggregation.MLP_MAX_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="max", with_mlp=True)
        elif self.aggregation_type == Aggregation.MLP_MEAN_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="mean", with_mlp=True)
        elif self.aggregation_type == Aggregation.MLP_ATTENTION_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="attention", with_mlp=True)
        elif self.aggregation_type == Aggregation.MAX_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="max", with_mlp=False)
        elif self.aggregation_type == Aggregation.MEAN_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="mean", with_mlp=False)
        elif self.aggregation_type == Aggregation.ATTENTION_POOLING:
            self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="attention", with_mlp=False)

        num_classes = 1 if config.num_classes == 2 else config.num_classes

        self.classifier = nn.Sequential(
            nn.Linear(self.encoding_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, num_classes)
        )
        
        self.gradients = None
        self.activations = None

    def forward(self, images, sites, mask):
        stats = {}

        b, k, h, w, c = images.shape
        device = images.device
        dtype = images.dtype

        all_images = einops.rearrange(images, "b k h w c -> (b k) h w c")
        #flatten_mask = einops.rearrange(mask, "b k -> (b k)")
        #flatten_indices = torch.where(flatten_mask)[0]

        # if flatten_indices.numel() == 0:
        #     flatten_indices = torch.zeros((1,), dtype=flatten_indices.dtype, device=flatten_indices.device)
        visible_images = all_images #[flatten_indices]

        visible_representations = self.resnet(visible_images)
        d = visible_representations.shape[-1]

        all_representations = torch.zeros((b * k, d), dtype=visible_representations.dtype, device=device)
        all_representations = visible_representations #[flatten_indices] 
        if b == 1 and k == 1:
            representations = all_representations.unsqueeze(0)
        else:
            representations = einops.rearrange(all_representations, "(b k) d -> b k d", b=b)
        if self.use_positional_embeddings:
            sites_embeddings = self.site_embedding(sites)
            representations += sites_embeddings

        if b== 1 and k== 1:
            patient_representation = representations
        else:
            patient_representation = self.aggregation(representations, mask.float())
            if self.aggregation_type == Aggregation.TRANSFORMER:
                patient_representation, aggregation_stats = patient_representation
                stats.update(aggregation_stats)

        logits = self.classifier(patient_representation)

        return logits, stats

    def save_gradients(self, grad):
        self.gradients = grad

    def get_activations(self):
        return self.activations

    def get_gradients(self):
        return self.gradients

    def forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self.save_gradients)

    def find_last_conv_layer(self):
        # Find the last convolutional layer in the resnet
        last_conv = None
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
