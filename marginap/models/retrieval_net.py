import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm


def get_backbone(name):
    if name == 'resnet18':
        logging.info("using resnet18")
        out_dim = 512
        backbone = models.resnet18(pretrained='imagenet')
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif name == 'resnet50':
        logging.info("using resnet50")
        out_dim = 2048
        backbone = models.resnet50(pretrained='imagenet')
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif name == 'swav':
        logging.info("using swav pretrained resnet50")
        out_dim = 2048
        backbone = torch.hub.load('facebookresearch/swav', 'resnet50')
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif name == 'resnet50_swsl':
        logging.info("using swsl pretrained resnet50")
        out_dim = 2048
        backbone = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        backbone = nn.Sequential(*list(backbone.children())[:-1])
    elif name == 'vit':
        logging.info("using ViT")
        out_dim = 768
        backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
        backbone.reset_classifier(-1)
    elif name == 'vit_deit':
        logging.info("using DeiT")
        out_dim = 384
        backbone = timm.create_model('vit_deit_small_patch16_224', pretrained=True)
        backbone.reset_classifier(-1)
    elif name == 'vit_deit_distilled':
        logging.info("using DeiT distilled")
        out_dim = 384
        backbone = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        backbone.reset_classifier(-1)
        backbone.head_dist = None
        backbone.dist_token = None
        backbone.pre_logits = nn.Identity()
    elif name == 'tnt':
        logging.info("using TnT")
        out_dim = 384
        backbone = timm.create_model('tnt_s_patch16_224', pretrained=True)
        backbone.reset_classifier(-1)

    return (backbone, out_dim)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        norm_features=False,
        without_fc=False,
    ):
        super().__init__()
        self.norm_features = norm_features
        self.without_fc = without_fc

        self.backbone, out_features = get_backbone(backbone_name)

        if self.norm_features:
            logging.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = nn.Linear(out_features, embed_dim)
        else:
            self.fc = nn.Identity()
            logging.info("Not using a linear projection layer")

    def forward(self, X):
        X = self.backbone(X)
        if isinstance(X, (tuple, list)):
            X = X[0]

        X = X.view(X.size(0), -1)
        X = self.standardize(X)
        X = self.fc(X)
        X = F.normalize(X, p=2, dim=1)
        return X
