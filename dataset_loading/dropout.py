import ast
import enum
from collections import Counter

import torch
import torch.nn as nn

from dataset_loading import datasetsingle
from utilities import utils

class DropOutType(str, enum.Enum):
    NONE = "none"
    INDEPENDENT = "independent"
    SITE = "site"
    KEEP_ONE = "keep_one"


class IndependentDropout(nn.Module):
    """Drop some images of the patients randomly."""

    def __init__(self, dropout_probability: float):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, images, sites, mask):
        if self.training:
            mask = torch.empty_like(mask).bernoulli_(1 - self.dropout_probability) * mask
        return images, sites, mask


class SiteDropout(nn.Module):
    DEFAULT_MASK = str([False] + [True] * (len(datasetsingle.SITE_MAPPING) - 1))

    def __init__(
        self,
        sites_counter: Counter,
        dropout_probability: float,
        keep_site_mask: str = DEFAULT_MASK,  # keep = True, drop = False
        disable_at_inference=True,
    ):
        super().__init__()
        self.disable_at_inference = disable_at_inference
        self.register_buffer("site_keep_probability", torch.zeros(len(datasetsingle.SITE_MAPPING)))
        self.keep_site_mask = ast.literal_eval(keep_site_mask)
        max_site_count = max(sites_counter.values())
        for k, v in sites_counter.items():
            keep_site = self.keep_site_mask[datasetsingle.SITE_MAPPING[k]]
            if keep_site:
                print(k)
                self.site_keep_probability[datasetsingle.SITE_MAPPING[k]] = (
                    1 - dropout_probability * v / max_site_count
                )

    def forward(self, images, sites, mask):
        if self.disable_at_inference and not self.training:
            return images, sites, mask
        proba_keep_image = self.site_keep_probability[sites].to(images.device)
        keep_image = torch.rand(sites.shape, device=images.device)
        keep_image = keep_image < proba_keep_image
        mask = mask.bool() & keep_image.bool()
        return images, sites, mask


class KeepOneSite(nn.Module):
    """Select only one image per patient."""

    def __init__(self, disable_at_inference=True):
        super().__init__()
        self.disable_at_inference = disable_at_inference

    def forward(self, images, sites, mask):
        if self.disable_at_inference and not self.training:
            return images, sites, mask

        batch_size = images.shape[0]
        num_images_per_patients = mask.sum(axis=1).long()

        # Pick a random image per patient.
        # Generates a large integer and take modulo the number of images to have a pseudo
        # uniform pick.
        selected_indices = (
            torch.randint(100000 * mask.shape[1], (batch_size,), device=images.device).long()
            % num_images_per_patients
        ).long()

        # Extract the selected elements
        batch_range = torch.arange(batch_size, device=images.device, dtype=torch.long)
        images = images[batch_range, selected_indices]
        sites = sites[batch_range, selected_indices]
        mask = mask[batch_range, selected_indices]

        # Add the lonely dim again
        images = images[:, None]
        sites = sites[:, None]
        mask = mask[:, None]

        return images, sites, mask


class KeepOneImagePerSite(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images, sites, mask):
        batch_size, max_images_per_patient = sites.shape
        num_sites = len(datasetsingle.SITE_MAPPING)

        sites_indices = torch.arange(num_sites, device=sites.device)

        # batch, num_sites, max_images_per_patient
        mask_per_site = sites[:, None, :] == sites_indices[None, :, None]

        # batch, num_sites
        count_sites = mask_per_site.sum(dim=-1)
        print("count_sites", count_sites)

        # batch_size, num_sites
        if self.training:
            selected_image_idx_per_site = (
                torch.randint(9999999, (batch_size, num_sites), device=sites.device).long()
                % count_sites.clip(min=1)  # clip to one to avoid modulo by 0.
            ).long()
        else:
            selected_image_idx_per_site = torch.zeros(
                (batch_size, num_sites), device=sites.device
            ).long()

        # batch, num_sites, max_images_per_patient
        image_order_per_site = utils.exclusive_cumsum(mask_per_site.long(), dim=-1)
        selected_image = (
            image_order_per_site == selected_image_idx_per_site[:, :, None]
        ) & mask_per_site
        new_mask = (selected_image.sum(dim=1) > 0).float() * mask

        return images, sites, new_mask


class RandomSite(nn.Module):
    def __init__(self, disable_at_inference=True):
        super().__init__()
        self.disable_at_inference = disable_at_inference

    def forward(self, images, sites, mask):
        if self.training or not self.disable_at_inference:
            sites = torch.randint(
                1,
                len(datasetsingle.SITE_MAPPING),
                size=sites.shape,
                dtype=sites.dtype,
                device=sites.device,
            )
        return images, sites, mask
