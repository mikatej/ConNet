import torch.nn as nn
import torch
from torchvision import models
from models.CSRNet.utils import save_net,load_net
from collections import OrderedDict

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        # self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend_feat = [[(3,3), (3,8), (8,64)], [(64,19), (19,17), (17,64)],
        'M', [(64,20), (20,25), (25,128)], [(128,26), (26,22), (22,128)],
        'M', [(128,37), (37,34), (34,256)], [(256,43), (43,43), (43,256)], [(256,40), (40,32), (32,256)],
        'M', [(256,59), (59,62), (62,512)], [(512,71), (71,68), (68,512)], [(512,65), (65,60), (60,512)]]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1

    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            i = len(layers)
            if type(v) == list:
                conv2d = nn.Sequential(OrderedDict([
                    (str(i) + '-0', nn.Conv2d(v[0][0], v[0][1], kernel_size=(1,1), stride=(1,1), bias=False)),
                    (str(i) + '-1', nn.Conv2d(v[1][0], v[1][1], kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)),
                    (str(i) + '-2'.format(i), nn.Conv2d(v[2][0], v[2][1], kernel_size=(1,1), stride=(1,1)))
                ]))

                v = v[2][1]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
