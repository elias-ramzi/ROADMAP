import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

import roadmap.utils as lib

from .create_projection_head import create_projection_head


def get_backbone(name, pretrained=True):
    if name == 'resnet18':
        lib.LOGGER.info("using ResNet-18")
        out_dim = 512
        backbone = models.resnet18(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet50':
        lib.LOGGER.info("using ResNet-50")
        out_dim = 2048
        backbone = models.resnet50(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'resnet101':
        lib.LOGGER.info("using ResNet-101")
        out_dim = 2048
        backbone = models.resnet101(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    elif name == 'vit':
        lib.LOGGER.info("using ViT-S")
        out_dim = 768
        backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit':
        lib.LOGGER.info("using DeiT-S")
        out_dim = 384
        try:
            backbone = timm.create_model('vit_deit_small_patch16_224', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_distilled':
        lib.LOGGER.info("using DeiT-S distilled")
        try:
            deit = timm.create_model('vit_deit_small_patch16_224')
            deit_distilled = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_small_patch16_224')
            deit_distilled = timm.create_model('deit_small_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 384
        pooling = nn.Identity()
    elif name == 'vit_deit_base':
        lib.LOGGER.info("using DeiT-B")
        out_dim = 768
        try:
            backbone = timm.create_model('vit_deit_base_patch16_224', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_base_distilled':
        lib.LOGGER.info("using DeiT-B distilled")
        try:
            deit = timm.create_model('deit_base_patch16_224')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_base_patch16_224')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()
    elif name == 'vit_deit_base_384':
        lib.LOGGER.info("using DeiT-B 384")
        out_dim = 768
        try:
            backbone = timm.create_model('vit_deit_base_patch16_384', pretrained=pretrained)
        except RuntimeError:
            backbone = timm.create_model('deit_base_patch16_384', pretrained=pretrained)
        backbone.reset_classifier(-1)
        pooling = nn.Identity()
    elif name == 'vit_deit_base_384_distilled':
        lib.LOGGER.info("using DeiT-B 384 distilled")
        try:
            deit = timm.create_model('deit_base_patch16_384')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_384', pretrained=pretrained)
        except RuntimeError:
            deit = timm.create_model('deit_base_patch16_384')
            deit_distilled = timm.create_model('deit_base_distilled_patch16_384', pretrained=pretrained)
        deit_distilled.pos_embed = torch.nn.Parameter(torch.cat((deit_distilled.pos_embed[:, :1], deit_distilled.pos_embed[:, 2:]), dim=1))
        deit.load_state_dict(deit_distilled.state_dict(), strict=False)
        backbone = deit
        backbone.reset_classifier(-1)
        out_dim = 768
        pooling = nn.Identity()
    else:
        raise ValueError(f"{name} is not recognized")

    return (backbone, pooling, out_dim)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        norm_features=False,
        without_fc=False,
        with_autocast=False,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=True,
    ):
        super().__init__()

        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)

        assert isinstance(without_fc, bool)
        assert isinstance(norm_features, bool)
        assert isinstance(with_autocast, bool)
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")

        self.backbone, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained)
        if pooling == 'default':
            self.pooling = default_pooling
        elif pooling == 'none':
            self.pooling = nn.Identity()
        elif pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        lib.LOGGER.info(f"Pooling is {self.pooling}")

        if self.norm_features:
            lib.LOGGER.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = create_projection_head(out_features, embed_dim, projection_normalization_layer)
            lib.LOGGER.info(f"Projection head : \n{self.fc}")
        else:
            self.fc = nn.Identity()
            lib.LOGGER.info("Not using a linear projection layer")

    def forward(self, X):
        with torch.cuda.amp.autocast(enabled=self.with_autocast):
            X = self.backbone(X)
            X = self.pooling(X)

            X = X.view(X.size(0), -1)
            X = self.standardize(X)
            X = self.fc(X)
            X = F.normalize(X, p=2, dim=1)
            return X
