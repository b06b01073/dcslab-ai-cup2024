import torch.nn as nn
import torch

import os
import torch.nn.functional as F
from collections import OrderedDict
import DenseNet
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights


__all__ = ['make_model', 'IBN_A', 'resnet101_ibn_a', 'resnext101_ibn_a', 'densenet169_ibn_a', 'se_resnet101_ibn_a', 'swin_reid', 'resnet34_ibn_a']

model_urls = {
    'densenet169_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/IBN_densenet.pth',
    'se_resnet101_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/IBN_seresnet.pth',
    'resnext101_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/IBN_resnext.pth',
    'resnet101_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/IBN_resnet.pth',
    'swin_reid': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/SwinReID.pth',
    'resnet34_ibn_a': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v3-hubconf/IBN_resnet34.pth',

    'densenet169_ibn_a_cent': 'https://github.com/b06b01073/veri776-pretrain/releases/download/center/densenet_cent.pth',
    'se_resnet101_ibn_a_cent': 'https://github.com/b06b01073/veri776-pretrain/releases/download/center/seresnet_cent.pth',
    'resnext101_ibn_a_cent': 'https://github.com/b06b01073/veri776-pretrain/releases/download/center/resnext_cent.pth',
    'resnet101_ibn_a_cent': 'https://github.com/b06b01073/veri776-pretrain/releases/download/center/resnet_cent.pth',
    'swin_reid_cent': 'https://github.com/b06b01073/veri776-pretrain/releases/download/center/swin_cent.pth',


    'densenet_169_ibn_a_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/IBN_densenet_cos.pth',
    'se_resnet101_ibn_a_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/IBN_seresnet_cos.pth',
    'resnext101_ibn_a_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/IBN_resnext_cos.pth',
    'resnet101_ibn_a_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/IBN_resnet_cos.pth',
    'swin_reid_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/SwinReID_cos.pth',
    'resnet34_ibn_a_finetuned': 'https://github.com/b06b01073/veri776-pretrain/releases/download/v4-fine-tuned/IBN_resnet34_cos.pth'
}

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



class IBN_A(nn.Module):
    def __init__(self, backbone, pretrained=True, num_classes=576, embedding_dim=2048):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained=pretrained)

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
    



def get_backbone(backbone, pretrained):
    
    assert backbone in ['resnet', 'resnext', 'seresnet', 'densenet', 'resnet34'], "no such backbone, we only support ['resnet', 'resnext', 'seresnet', 'densenet', 'resnet34]"

    if backbone == 'resnet':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=pretrained)
    
    if backbone == 'resnext':
        return torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=pretrained)

    if backbone == 'seresnet':
        return torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=pretrained)

    if backbone == 'resnet34':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=pretrained)


    if backbone == 'densenet':
        return DenseNet.densenet169_ibn_a(pretrained=pretrained)
    

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



def make_model(backbone, num_classes, embedding_dim=2048):
    print(f'using {backbone} as backbone')

    if backbone == 'swin':
        return SwinReID(num_classes)


    return IBN_A(backbone, num_classes, embedding_dim=embedding_dim)



def densenet169_ibn_a(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = IBN_A(backbone='densenet', pretrained=False)
    
    if cent_loss:
        print('using pretrained model with center loss')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet169_ibn_a_cent']))
    else:
        if fine_tuned:
            print('using fine tuned model')
            model.classifier = nn.Linear(in_features=2048, out_features=3440, bias=False)
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet_169_ibn_a_finetuned'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['densenet169_ibn_a']))

    if print_net:
        print(model)


    return model



def se_resnet101_ibn_a(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = IBN_A(backbone='seresnet', pretrained=False)
    if cent_loss:
        print('using pretrained model with center loss')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['se_resnet101_ibn_a']))
    else:
        if fine_tuned:
            print('using fine tuned model')
            model.classifier = nn.Linear(in_features=2048, out_features=3440, bias=False)
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['se_resnet101_ibn_a_finetuned'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['se_resnet101_ibn_a']))


        

    if print_net:
        print(model)

    return model


def resnext101_ibn_a(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = IBN_A(backbone='resnext', pretrained=False)
        
    if cent_loss:
        print('using pretrained model with center loss')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnext101_ibn_a_cent']))
    else:
        if fine_tuned:
            print('using fine tuned model')
            model.classifier = nn.Linear(in_features=2048, out_features=3440, bias=False)
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnext101_ibn_a_finetuned'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnext101_ibn_a']))

    if print_net:
        print(model)

    return model


def resnet101_ibn_a(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = IBN_A(backbone='resnet', pretrained=False)

    if cent_loss:
        print('using pretrained model with center loss')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a_cent']))
    else:
        if fine_tuned:
            print('using fine tuned model')
            model.classifier = nn.Linear(in_features=2048, out_features=3440, bias=False)
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a_finetuned'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a']))

        if print_net:
            print(model)

    return model


def resnet34_ibn_a(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = IBN_A(backbone='resnet34', pretrained=False, embedding_dim=512)

    if fine_tuned:
        model.classifier = nn.Linear(in_features=512, out_features=3421, bias=False)
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_a_finetuned']))
    else:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet34_ibn_a']))


    if print_net:
        print(model)

    return model


def swin_reid(print_net=False, fine_tuned=False, device='cpu', cent_loss=False):
    model = SwinReID(num_classes=576)
    
    if cent_loss:
        print('using pretrained model with center loss')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['swin_reid_cent']))
    else:
        if fine_tuned:
            print('using fine tuned model')
            model.classifier = nn.Linear(in_features=2048, out_features=3440, bias=False)
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['swin_reid_finetuned'], map_location=torch.device(device)))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['swin_reid']))

    if print_net:
        print(model)

    return model