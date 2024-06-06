# do not rename MDL as model to avoid naming confliction with the model.py from remote repo

import torch.nn as nn
import torch

import os
import torch.nn.functional as F
from collections import OrderedDict
import DenseNet
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# this class is kept for compatibility issue run3, run4 and rerun use this class 
class Resnet101IbnA(nn.Module):
    def __init__(self, num_classes=576):
        from warnings import warn
        warn('Deprecated warning: You should only use this class if you want to load the model trained in older commits. You should use `make_model(backbone, num_classes)` to build the model in newer version.')

        super().__init__()
        self.resnet101_ibn_a = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        
        embedding_dim = self.resnet101_ibn_a.fc.in_features
        
        self.resnet101_ibn_a.fc = nn.Identity() # pretend the last layer does not exist



        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.resnet101_ibn_a(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out



class SwinReID(nn.Module):
    def __init__(self, num_classes, embedding_dim=2048, imagenet_weight=True):
        super().__init__()

        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if imagenet_weight else None)

        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.swin(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out



class IBN_A(nn.Module):
    def __init__(self, backbone='pretrained', num_classes=576, embedding_dim=2048):
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

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.backbone(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    



def get_backbone(backbone):
    print(f'using {backbone} as backbone')
    if backbone == 'resnet':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=False)
    
    if backbone == 'resnext':
        return torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=False)

    if backbone == 'seresnet':
        return torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=False)

    if backbone == 'densenet':
        return DenseNet.densenet169_ibn_a(pretrained=False)
    
    if backbone == 'resnet34':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=False)



# def load_pretrained(model_path, type='resnext'):
#     model = make_model('densenet', num_classes=576)
#     model.load_state_dict(torch.load(model_path))
#     return model


def make_ibn_model(backbone, num_classes, embedding_dim=2048):
    return IBN_A(backbone, num_classes, embedding_dim)


def make_finetuned_model(backbone, num_classes, weights=None, cent_loss=True):
    ''' load the pretrained model (if `weights` is provided) and modify the structure of last nn.Linear layer 

        Args:
            backbone (str): specify the backbone for the model, it need to be one of ['resnet', 'resnext', 'seresnet', 'densenet']
            weights (str): if weight is provided, this function will load the pretrained weights from veri776-pretrain repo, visit that repo if you want to see what options are provided
            num_classes (int): the number of class for the last fully connected layer

        Returns: 
            return the model after modifying the output layer to have `num_classes` out_features 

    '''

    assert backbone in ['resnet', 'resnext', 'seresnet', 'densenet', 'swin']

    if weights:
        net = torch.hub.load('b06b01073/veri776-pretrain', weights, cent_loss=cent_loss)
    elif backbone == 'swin':
        net = SwinReID(num_classes=576)
    else:
        net = IBN_A(backbone)

    net.classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=False)
    net.classifier.apply(weights_init_classifier)

    return net


if __name__ == '__main__':


    load_pretrained('../veri776ReID/dense.pth')
    # print(model)