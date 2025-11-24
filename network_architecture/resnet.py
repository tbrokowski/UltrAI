
import torch
import torch.nn as nn
from torchvision import models
import timm

class ResNet(nn.Module):
    def __init__(self, model_name='resnet18', freeze=False, pretrained_path=None, pretrained=True):
        super().__init__()

        if pretrained_path is not None and pretrained:
            raise ValueError(
                "Loading a pretrained ResNet should either be from torchvision (pretrained=True) "
                "or from a checkpoint (pretrained_path) but not both."
            )

        # Select the appropriate model based on the model_name
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnext50_32x4d':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
        elif model_name == 'inceptionresnetv2':
            self.resnet = timm.create_model('inception_resnet_v2', pretrained=True)
        else:
            raise ValueError(f"Model name {model_name} is not supported. Choose from 'resnet18', 'resnet50', or 'resnext50_32x4d'.")

        # load pre-trained ResNet from path
        if pretrained_path:
            model_dict = self.resnet.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in torch.load(pretrained_path).items() if k in model_dict
            }
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.resnet.load_state_dict(model_dict)

        # remove final classification layer
        self.resnet.fc = nn.Identity()

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = True

            # Unfreeze the final layer (the last block before the classification layer)
            # for param in self.resnet.layer4.parameters():
            #     param.requires_grad = True

    def forward(self, images):
        representations = self.resnet(images)
        return representations