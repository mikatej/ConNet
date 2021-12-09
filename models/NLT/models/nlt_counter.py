import  torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import vgg16
from .resnet_50 import ResNet50
from .decoder import decoder,upsampler
from collections import OrderedDict
from torch.hub import load_state_dict_from_url


class NLT_Counter(nn.Module):
    def __init__(self, mode=None, backbone='vgg16'):
        super().__init__()
        self.mode = mode

        if self.mode == 'nlt':
            if backbone == 'vgg16':
                print('backbone is vgg16')
                self.encoder = vgg16(nlt=True)
                self.decoder = decoder(feature_channel=512,nlt=True)

            elif backbone == 'ResNet50':
                print('backbone is ResNet50')
                self.encoder = ResNet50(nlt=True)
                state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")
                self.encoder.load_state_dict(state_dict, strict=False)
                self.decoder = decoder(feature_channel=1024, nlt=True)

        else:
            if backbone == 'vgg16':
                self.encoder = vgg16(pretrained=True, nlt=False)
                self.decoder = decoder(feature_channel=512)

            elif backbone == 'ResNet50':
                self.encoder =  ResNet50(nlt=False)
                state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")
                self.encoder.load_state_dict(state_dict, strict=False)
                self.decoder =  decoder(feature_channel=1024, nlt=False)


    def forward(self, inp):

        return self.decoder(self.encoder(inp))

