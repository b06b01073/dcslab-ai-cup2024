import torch.nn as nn
import torch
from . import DenseNet

from torchvision.models.swin_transformer import swin_b, Swin_B_Weights


__all__ = ['se_resnet_ibn_a', 'resnet101_ibn_a', 'resnext101_ibn_a', 'densenet169_ibn_a', 'swin_reid']

model_urls = {
    'resnet101_ibn_a': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/resnet.pth',
    'resnext101_ibn_a': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/resnext.pth',
    'se_resnet101_ibn_a': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/seresnet.pth',
    'densenet169_ibn_a': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/densenet.pth',
    'swin_reid': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/swin.pth',

    'resnet101_ibn_a_use_test': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/resnet_use_test.pth',
    'resnext101_ibn_a_use_test': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/resnext_use_test.pth',
    'se_resnet101_ibn_a_use_test': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/seresnet_use_test.pth',
    'densenet169_ibn_a_use_test': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/densenet_use_test.pth',
    'swin_reid_use_test': 'https://github.com/b06b01073/dcslab-ai-cup2024/releases/download/v1-centloss/swin_use_test.pth',
}


class SwinReID(nn.Module):
    def __init__(self, num_classes, embedding_dim=2048, imagenet_weight=True):
        super().__init__()

        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if imagenet_weight else None)

        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)


    def forward(self, x):
        f_t = self.swin(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    


class IBN_A(nn.Module):
    def __init__(self, backbone, pretrained=True, num_classes=3421, embedding_dim=2048):
        super().__init__()
        self.backbone = get_backbone(backbone)

        # the expected embedding space is \mathbb{R}^{2048}. resnet, seresnet, resnext satisfy this automatically
        if backbone == 'densenet':
            self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, embedding_dim)
        else:
            self.backbone.fc = nn.Identity() # pretend the last layer does not exist


        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)


    def forward(self, x):
        f_t = self.backbone(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    

def get_backbone(backbone):
    
    assert backbone in ['resnet', 'resnext', 'seresnet', 'densenet'], "no such backbone, we only support ['resnet', 'resnext', 'seresnet', 'densenet']"

    if backbone == 'resnet':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=False)
    
    if backbone == 'resnext':
        return torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=False)

    if backbone == 'seresnet':
        return torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=False)
    
    if backbone == 'densenet':
        return DenseNet.densenet169_ibn_a(pretrained=False)



def load_from_url(model, url):
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=torch.device('cpu')))

    return model

def swin_reid(use_test=False):
    num_classes, url = (4295, model_urls['swin_reid_use_test']) if use_test else (3421, model_urls['swin_reid'])
    model = SwinReID(num_classes=num_classes)
    model = load_from_url(model, url)

    return model


def resnet101_ibn_a(use_test=False):
    num_classes, url = (4295, model_urls['resnet101_ibn_a_use_test']) if use_test else (3421, model_urls['resnet101_ibn_a'])
    model = IBN_A(backbone='resnet', pretrained=False, num_classes=num_classes)
    model = load_from_url(model, url)

    return model


def resnext101_ibn_a(use_test=False):
    num_classes, url = (4295, model_urls['resnext101_ibn_a_use_test']) if use_test else (3421, model_urls['resnext101_ibn_a'])
    model = IBN_A(backbone='resnext', pretrained=False, num_classes=num_classes)
    model = load_from_url(model, url)

    return model


def se_resnet101_ibn_a(use_test=False):
    num_classes, url = (4295, model_urls['se_resnet101_ibn_a_use_test']) if use_test else (3421, model_urls['se_resnet101_ibn_a'])
    model = IBN_A(backbone='seresnet', pretrained=False, num_classes=num_classes)
    model = load_from_url(model, url)

    return model


def densenet169_ibn_a(use_test=False):
    num_classes, url = (4295, model_urls['densenet169_ibn_a_use_test']) if use_test else (3421, model_urls['densenet169_ibn_a'])

    model = IBN_A(backbone='densenet', pretrained=False, num_classes=num_classes)
    model = load_from_url(model, url)

    return model