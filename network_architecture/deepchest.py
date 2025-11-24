import enum
import einops
import torch
import torch.nn as nn
#from dataset_loading import datasetsingle
from dataset_loading import dataset
import network_architecture as models
from .resnet import ResNet
from .convnext import ConvNeXtTiny

from dataset_loading.newdatasets import UltrasoundImageDataset, SITE_MAPPING

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', pos_weight=None):
        """
        gamma: focusing parameter to control the down-weighting of easy examples (default=2)
        alpha: balancing factor between classes (optional). If set, it should be a tensor of size [C].
        reduction: specifies the reduction to apply to the output ('none' | 'mean' | 'sum')
        pos_weight: weight for the positive class (similar to pos_weight in BCEWithLogitsLoss)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Calculate the BCEWithLogitsLoss (logits -> sigmoid inside)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        
        # Apply sigmoid to get probabilities
        probas = torch.sigmoid(inputs)
        
        # Calculate pt (the model's estimated probability for the actual class)
        pt = torch.where(targets == 1, probas, 1 - probas)
        
        # Focal loss factor: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # If alpha is provided, apply class balancing factor
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        # Apply focal weight to the original BCE loss
        loss = focal_weight * bce_loss
        
        # Apply reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
        #self.convnext = ConvNeXtTiny(**config.resnet)
        self.encoding_dim = config.encoding_dim
        self.Pooling = config.pooling

        self.use_positional_embeddings = config.use_positional_embeddings

        if self.use_positional_embeddings:
            self.site_embedding = nn.Embedding(len(SITE_MAPPING), self.encoding_dim)

        # Aggregation
        self.aggregation = None
        self.aggregation_type = config.aggregation_type
        if self.aggregation_type == Aggregation.TRANSFORMER:
            self.aggregation = models.Transformer(config)
        elif self.aggregation_type == Aggregation.DEEP_SET:
            self.aggregation = models.DeepSet(features= self.encoding_dim)
        elif self.aggregation_type == Aggregation.MLP_MAX_POOLING:
            self.aggregation = models.Pooling(features= self.encoding_dim, pooling_type="max", with_mlp=True)
        elif self.aggregation_type == Aggregation.MLP_MEAN_POOLING:
            self.aggregation = models.Pooling(features= self.encoding_dim, pooling_type="mean", with_mlp=True)
        elif self.aggregation_type == Aggregation.MLP_ATTENTION_POOLING:
            self.aggregation = models.Pooling(features= self.encoding_dim, pooling_type="attention", with_mlp=True)
        elif self.aggregation_type == Aggregation.MAX_POOLING:
            self.aggregation = models.Pooling(features= self.encoding_dim, pooling_type="max", with_mlp=False)
        elif self.aggregation_type == Aggregation.MEAN_POOLING:
            self.aggregation = models.Pooling(features= self.encoding_dim, pooling_type="mean", with_mlp=False)
        elif self.aggregation_type == Aggregation.ATTENTION_POOLING:
            self.aggregation = models.Pooling(
                features= self.encoding_dim, pooling_type="attention", with_mlp=False
            )

        num_classes = 1 if config.num_classes == 2 else config.num_classes

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 1)

        )

        self.gradients = None
        self.activations = None


    def forward(self, images, sites, mask):
        stats = {}

        b, k, h, w, c = images.shape
        device = images.device
        dtype = images.dtype

        # Computes representation for each image separately.
        all_images = einops.rearrange(images, "b k h w c -> (b k) h w c")
        #print(all_images.shape)


        # Extract the non masked images only to run the ResNet on

        if self.Pooling:
            flatten_mask = einops.rearrange(mask, "b k -> (b k)")
            flatten_indices = torch.where(flatten_mask)[0]
            
            # Add safety check - ensure indices are within valid range
            max_index = all_images.shape[0] - 1
            valid_indices_mask = flatten_indices <= max_index
            if not torch.all(valid_indices_mask):
                # Filter out invalid indices
                flatten_indices = flatten_indices[valid_indices_mask]
                
            # If all images are masked or all indices are invalid, we still process the first image
            if flatten_indices.numel() == 0:
                flatten_indices = torch.zeros((1,), dtype=flatten_indices.dtype, device=flatten_indices.device)
            
            # Double-check that all indices are valid before indexing
            assert torch.all(flatten_indices < all_images.shape[0]), f"Invalid indices detected: max index {flatten_indices.max()} >= tensor size {all_images.shape[0]}"
            
            visible_images = all_images[flatten_indices]
        else:
            visible_images = all_images

        # Compute representation with the ResNet
        #print(visible_images.shape)
        visible_representations = self.resnet(visible_images)
        d = visible_representations.shape[-1]

        # # Rearrange the representations back to the masked format
        # all_representations = torch.zeros((b * k, d), dtype=dtype, device=device)
        # all_representations[flatten_indices] = visible_representations
        # representations = einops.rearrange(all_representations, "(b k) d -> b k d", b=b)

        # visible_representations = convnext_model(visible_images)
        # d = visible_representations.shape[-1]

        # Rearrange the representations back to the masked format

        if self.Pooling:
            all_representations = torch.zeros((b * k, d), dtype=visible_representations.dtype, device=device)
            all_representations[flatten_indices] = visible_representations
            representations = einops.rearrange(all_representations, "(b k) d -> b k d", b=b)
        elif b == 1 and k == 1:
            all_representations = visible_representations
            representations = all_representations.unsqueeze(0)
        else:
            all_representations = visible_representations
            representations = einops.rearrange(all_representations, "(b k) d -> b k d", b=b)

        #print(representations.shape)

        if self.use_positional_embeddings:
            sites_embeddings = self.site_embedding(sites)
            representations = representations + sites_embeddings  #sites_embeddings.expand(-1, representations.size(1), -1)
            #representations += sites_embeddings

        #print(representations.shape)

        if self.Pooling:
        # Get fixed representation per patient with aggregation function.
            patient_representation = self.aggregation(representations, mask.float())
            if self.aggregation_type == Aggregation.TRANSFORMER:
                patient_representation, aggregation_stats = patient_representation
                stats.update(aggregation_stats)
        else:
            patient_representation = representations

        #print("pt", patient_representation.shape)  
        #patient_representation = patient_representation.view(-1, 512)  
        #print(patient_representation.shape)  
        #patient_representation = patient_representation.float()

        logits = self.classifier(patient_representation)  # Pass through classifier
        #logits = patient_representation
        #print(logits.shape)  # Check the output shape, should be [32, 1] if the classifier ends with Linear(1024, 1)

        #Squeeze logits if necessary
        if not self.Pooling:
           logits = logits.squeeze(1)  # Use dim=1, not dim=2, if the shape is [32, 1]
            
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

# class DeepChest(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.resnet = ResNet(**config.resnet)
#         self.encoding_dim = config.encoding_dim
#         self.Pooling = config.pooling

#         self.use_positional_embeddings = config.use_positional_embeddings

#         if self.use_positional_embeddings:
#             self.site_embedding = nn.Embedding(len(SITE_MAPPING), self.encoding_dim)

#         # Aggregation
#         self.aggregation = None
#         self.aggregation_type = config.aggregation_type
#         if self.aggregation_type == Aggregation.TRANSFORMER:
#             self.aggregation = models.Transformer(config)
#         elif self.aggregation_type == Aggregation.DEEP_SET:
#             self.aggregation = models.DeepSet(features=self.encoding_dim)
#         elif self.aggregation_type == Aggregation.MLP_MAX_POOLING:
#             self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="max", with_mlp=True)
#         elif self.aggregation_type == Aggregation.MLP_MEAN_POOLING:
#             self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="mean", with_mlp=True)
#         elif self.aggregation_type == Aggregation.MLP_ATTENTION_POOLING:
#             self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="attention", with_mlp=True)
#         elif self.aggregation_type == Aggregation.MAX_POOLING:
#             self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="max", with_mlp=False)
#         elif self.aggregation_type == Aggregation.MEAN_POOLING:
#             self.aggregation = models.Pooling(features=self.encoding_dim, pooling_type="mean", with_mlp=False)
#         elif self.aggregation_type == Aggregation.ATTENTION_POOLING:
#             self.aggregation = models.Pooling(
#                 features=self.encoding_dim, pooling_type="attention", with_mlp=False
#             )

#         num_classes = 1 if config.num_classes == 2 else config.num_classes

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.Tanh(),
#             nn.Linear(1024, 1),
#         )

#         self.gradients = None
#         self.activations = None

#     def forward(self, images, sites, mask):
#         # Initialize empty stats dictionary
#         stats = {}

#         try:
#             # Get the shape of the input images
#             b, k, h, w, c = images.shape
#             device = images.device
#             dtype = images.dtype

#             # Debug print to check shapes
#             # print(f"Input shapes - images: {images.shape}, sites: {sites.shape}, mask: {mask.shape}")

#             # Reshape images for processing
#             all_images = einops.rearrange(images, "b k h w c -> (b k) h w c")
#             # print(f"all_images shape: {all_images.shape}")

#             # Handle non-pooling case directly (simpler path)
#             if not self.Pooling:
#                 # Process all images
#                 visible_images = all_images
#                 visible_representations = self.resnet(visible_images)
                
#                 # Reshape back to [batch, frames, features]
#                 representations = einops.rearrange(visible_representations, "(b k) d -> b k d", b=b)
                
#                 # Add positional embeddings if enabled
#                 if self.use_positional_embeddings:
#                     sites_embeddings = self.site_embedding(sites)
#                     representations = representations + sites_embeddings
                
#                 # Take first frame or squeeze dimension if k=1
#                 if k > 1:
#                     patient_representation = representations[:, 0, :]  # Take first frame
#                 else:
#                     patient_representation = representations.squeeze(1)
                
#                 # Get predictions from classifier
#                 logits = self.classifier(patient_representation)
                
#                 # Squeeze last dimension if it's 1
#                 if logits.size(-1) == 1:
#                     logits = logits.squeeze(-1)
                
#                 return logits, stats
            
#             # Complex path for pooling case
#             else:
#                 # Extract indices of non-masked images
#                 flatten_mask = einops.rearrange(mask, "b k -> (b k)")
#                 # Get indices where mask is true
#                 flatten_indices = torch.where(flatten_mask > 0)[0]
                
#                 # Safety check: if no valid indices or all indices are invalid
#                 if flatten_indices.numel() == 0:
#                     # Use first image as fallback
#                     flatten_indices = torch.zeros(1, dtype=torch.long, device=device)
                
#                 # Safety check: validate indices are within bounds
#                 if flatten_indices.max() >= all_images.shape[0]:
#                     # Clamp indices to valid range
#                     flatten_indices = torch.clamp(flatten_indices, 0, all_images.shape[0] - 1)
                
#                 # Get visible images using safe indices
#                 visible_images = all_images[flatten_indices]
                
#                 # Process through ResNet
#                 visible_representations = self.resnet(visible_images)
                
#                 # Get representation dimension
#                 d = visible_representations.shape[-1]
                
#                 # Create zeros tensor and fill with representations at appropriate indices
#                 all_representations = torch.zeros((b * k, d), dtype=visible_representations.dtype, device=device)
#                 all_representations[flatten_indices] = visible_representations
                
#                 # Reshape back to [batch, frames, features]
#                 representations = einops.rearrange(all_representations, "(b k) d -> b k d", b=b)
                
#                 # Apply positional embeddings if enabled
#                 if self.use_positional_embeddings:
#                     sites_embeddings = self.site_embedding(sites)
#                     representations = representations + sites_embeddings
                
#                 # Use aggregation function to get fixed representation
#                 patient_representation = self.aggregation(representations, mask.float())
                
#                 # Handle Transformer special case
#                 if self.aggregation_type == Aggregation.TRANSFORMER:
#                     patient_representation, aggregation_stats = patient_representation
#                     stats.update(aggregation_stats)
                
#                 # Final prediction
#                 logits = self.classifier(patient_representation)
                
#                 # Squeeze last dimension if it's 1
#                 if logits.size(-1) == 1:
#                     logits = logits.squeeze(-1)
                
#                 return logits, stats
                
#         except RuntimeError as e:
#             if "CUBLAS_STATUS_NOT_INITIALIZED" in str(e):
#                 # Handle CUDA initialization error by moving tensors to CPU
#                 # This is a fallback mechanism for CUDA errors
#                 print("WARNING: CUDA error detected. Falling back to CPU processing.")
                
#                 # Move model to CPU if it's not already there
#                 device_backup = next(self.parameters()).device
#                 self.to('cpu')
                
#                 # Move inputs to CPU
#                 cpu_images = images.to('cpu')
#                 cpu_sites = sites.to('cpu')
#                 cpu_mask = mask.to('cpu')
                
#                 # Process on CPU
#                 b, k, h, w, c = cpu_images.shape
#                 all_images = einops.rearrange(cpu_images, "b k h w c -> (b k) h w c")
                
#                 # Simple path: just use first frame of each batch
#                 # This is a simplified fallback approach
#                 if k > 1:
#                     # Take first frame of each batch
#                     first_frames = all_images[::k]
#                     representations = self.resnet(first_frames)
#                 else:
#                     representations = self.resnet(all_images)
                
#                 # Process through classifier
#                 logits = self.classifier(representations)
                
#                 # Squeeze if needed
#                 if logits.size(-1) == 1:
#                     logits = logits.squeeze(-1)
                
#                 # Move results back to original device
#                 self.to(device_backup)
#                 logits = logits.to(device_backup)
                
#                 return logits, stats
#             else:
#                 # If it's another type of error, re-raise it
#                 raise

#     def save_gradients(self, grad):
#         self.gradients = grad

#     def get_activations(self):
#         return self.activations

#     def get_gradients(self):
#         return self.gradients

#     def forward_hook(self, module, input, output):
#         self.activations = output
#         output.register_hook(self.save_gradients)

#     def find_last_conv_layer(self):
#         # Find the last convolutional layer in the resnet
#         last_conv = None
#         for name, module in self.resnet.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 last_conv = module
#         return last_conv