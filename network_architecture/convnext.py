import torch
import torch.nn as nn
from torchvision import models
class ConvNeXtTiny(nn.Module):
    def __init__(self, *, freeze=False, pretrained_path=None, pretrained=True):
        super().__init__()

        if pretrained_path is not None and pretrained:
            raise ValueError(
                "Loading a pretrained ConvNeXt should either be from torch.vision (pretrained=True) "
                "or from a checkpoint (pretrained_path) but not both."
            )

        # if pretrained, loads ConvNeXt Tiny pretrained on ImageNet
        self.convnext = models.convnext_tiny(pretrained=pretrained)

        # load pre-trained ConvNeXt from path
        if pretrained_path:
            model_dict = self.convnext.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in torch.load(pretrained_path).items() if k in model_dict
            }
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.convnext.load_state_dict(model_dict)

        # remove final classification layer
        self.convnext.classifier = nn.Identity()

        if freeze:
            for p in self.convnext.parameters():
                p.requires_grad = False

    def forward(self, images):
        representations = self.convnext(images)
        # Flatten the representations
        representations = representations.view(representations.size(0), -1)
        return representations