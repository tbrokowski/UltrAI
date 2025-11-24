import torch
import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self, features: int, pooling_type: str = "max", with_mlp: bool = False):
        super().__init__()
        self.pooling_type = pooling_type
        self.with_mlp = with_mlp
        if with_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(features, features * 4),
                nn.ReLU(),
                nn.Linear(features * 4, features),
            )
        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(features, features), nn.Tanh(), nn.Linear(features, 1)
            )

    def forward(self, representations, mask):
        if self.with_mlp:
            representations = self.mlp(representations)
        if self.pooling_type == "max":
            # Replace the masked value with a very small value to hide them from max.
            mask = mask.unsqueeze(-1)
            masked_value = representations.min() - 1.0
            representations = mask * representations + (1 - mask) * masked_value
            representations = representations.max(axis=1).values
        elif self.pooling_type == "mean":
            num_images_per_patient = mask.sum(axis=1)
            num_images_per_patient = torch.clip(num_images_per_patient, min=1)  # To avoid NaN
            representations = (
                torch.einsum("bsd,bs->bd", representations, mask) / num_images_per_patient[:, None]
            )
        elif self.pooling_type == "attention":

            a = self.attention(representations)
            a = torch.einsum("bsd,bs->bsd", a, mask)
            a = a.flatten(1)
            a_max = torch.max(a, dim=1, keepdim=True)[0]
            a_exp = torch.exp(a - a_max)
            a_exp = a_exp * (a != 0).float()  # this step masks
            a_softmax = a_exp / (torch.sum(a_exp, dim=1, keepdim=True) + 1e-10)

            representations = torch.einsum("bsd,bs->bd", representations, a_softmax)
        else:
            raise ValueError(f"Unknown pooling type '{self.pooling_type}'.'")
        return representations
