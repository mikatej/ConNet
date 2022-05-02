'''
multi-level attention refine net
'''
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.MARUNet.utils import *
from torchvision.models import vgg16
from torch.hub import load_state_dict_from_url

channel_nums = [[32, 64, 128, 256, 512],  # half
                [21, 43, 86, 170, 340],  # third
                [16, 32, 64, 128, 256],  # quarter
                [13, 26, 51, 102, 205],  # fifth
                ]

class MARNet(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=False, NL='relu', objective='dmp', sp=False, se=False, block='', save_feature=False, ratio=4, transform=True):
        super(MARNet, self).__init__()

        channel = channel_nums[ratio-2]
        self.transform = transform
        
        self.downsample = downsample
        self.bn = bn
        self.NL = NL
        self.objective = objective
        self.sf = save_feature

        self.front0_0 = make_layers([channel[0]], in_channels=3, batch_norm=bn, NL=self.NL)
        if self.transform:
            self.front0_transform = feature_transform(channel[0], 64)
        self.front0_1 = make_layers([channel[0]], in_channels=channel[0], batch_norm=bn, NL=self.NL)
        # self.front0 = make_layers([channel[0]], in_channels=3, batch_norm=bn, NL=self.NL)

        self.pool1 = pool_layers()
        if self.transform:
            self.front1_transform = feature_transform(channel[0], 64)
        self.front1 =make_layers([channel[1]], in_channels=channel[0], batch_norm=bn, NL=self.NL)
        # self.front1 = make_layers(['M', channel[1], channel[1]], in_channels=channel[0], batch_norm=bn, NL=self.NL)

        self.pool2 = pool_layers()
        if self.transform:
            self.front2_transform = feature_transform(channel[1], 128)
        self.front2 = make_layers([channel[2], channel[2]], in_channels=channel[1], batch_norm=bn, NL=self.NL)
        # self.front2 = make_layers(['M', channel[2], channel[2], channel[2]], in_channels=channel[1], batch_norm=bn, NL=self.NL)

        self.pool3 = pool_layers()
        if self.transform:
            self.front3_transform = feature_transform(channel[2], 256)
        self.front3 = make_layers([channel[3], channel[3]], in_channels=channel[2], batch_norm=bn, NL=self.NL)
        # self.front3 = make_layers(['M', channel[3], channel[3], channel[3]], in_channels=channel[2], batch_norm=bn, NL=self.NL)

        self.pool4 = pool_layers()
        if self.transform:
            self.front4_transform = feature_transform(channel[3], 512)
        self.front4 = make_layers([channel[3], channel[3]], in_channels=channel[3], batch_norm=bn, NL=self.NL)
        # self.front4 = make_layers(['M', channel[3], channel[3], channel[3]], in_channels=channel[3], batch_norm=bn, NL=self.NL)
        
        self.brg_0 = make_layers([channel[3]], in_channels=channel[3], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.brg_transform = feature_transform(channel[3], 512)
        # self.brg = make_layers([channel[3]], in_channels=channel[3], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        self.back4 = make_layers([channel[3]], in_channels=channel[4], dilation=True, batch_norm=bn, NL=self.NL, se=se)
       
        self.back3 = make_layers([channel[2]], in_channels=channel[4], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.back3_transform = feature_transform(channel[2], 256)
        # self.back3 = make_layers([channel[2],], in_channels=channel[4], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        self.back2 = make_layers([channel[1]], in_channels=channel[3], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back1 = make_layers([channel[0]], in_channels=channel[2], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        self.back0 = make_layers([channel[0]], in_channels=channel[1], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        # objective is density map(dmp) and (binary) attention map(amp)
        print('objective dmp+amp!')
        self.amp_conv4_0 = make_layers([channel[2]], in_channels=channel[3], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.amp_conv4_transform = feature_transform(channel[2], 256)
        # self.amp_conv4 = make_layers([channel[2]], in_channels=channel[3], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        self.amp_conv3 = make_layers([channel[2]], in_channels=channel[2], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.amp_conv3_transform = feature_transform(channel[2], 256)

        self.amp_conv2_0 = make_layers([channel[1]], in_channels=channel[2], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.amp_conv2_transform = feature_transform(channel[1], 128)
        # self.amp_conv2 = make_layers([channel[1]], in_channels=channel[2], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        
        self.amp_conv1 = make_layers([channel[1]], in_channels=channel[1], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.amp_conv1_transform = feature_transform(channel[1], 128)

        self.amp_conv0 = make_layers([channel[0]], in_channels=channel[1], dilation=True, batch_norm=bn, NL=self.NL, se=se)
        if self.transform:
            self.amp_conv0_transform = feature_transform(channel[0], 64)
        
        self.amp_outconv4 = nn.Conv2d(channel[2], 1, kernel_size=3,padding=1)
        self.amp_outconv3 = nn.Conv2d(channel[2], 1, kernel_size=3,padding=1)
        self.amp_outconv2 = nn.Conv2d(channel[1], 1, kernel_size=3,padding=1)
        self.amp_outconv1 = nn.Conv2d(channel[1], 1, kernel_size=3,padding=1)
        self.amp_outconv0 = nn.Conv2d(channel[0], 1, kernel_size=3,padding=1)
        self.sgm = nn.Sigmoid()
        
        self.outconvb = nn.Conv2d(channel[3],1,3,padding=1)
        self.outconv4 = nn.Conv2d(channel[3],1,3,padding=1)
        self.outconv3 = nn.Conv2d(channel[2],1,3,padding=1)
        self.outconv2 = nn.Conv2d(channel[1],1,3,padding=1)
        self.outconv1 = nn.Conv2d(channel[0],1,3,padding=1)
        self.output_layer = nn.Conv2d(channel[0], 1, kernel_size=1)
        self.load_model = load_model
        # self._init_weights()

        self._initialize_weights()
        self.features = []


    def forward(self, x_in):
        self.features = []

        # x0 = self.front0(x_in)#1 size, 64
        # x1 = self.front1(x0)#1/2 size, 128
        # x2 = self.front2(x1)#1/4 size, 256
        # x3 = self.front3(x2)#1/8 size, 512
        # x4 = self.front4(x3)#1/16 size, 512
        # x_brg = self.brg(x4)#1/16 size, 512
        
        x0 = self.front0_0(x_in)
        if self.transform:
            self.features.append(self.front0_transform(x0))
        x0 = self.front0_1(x0)

        x1 = self.pool1(x0)
        if self.transform:
            self.features.append(self.front1_transform(x1))
        x1 = self.front1(x1)

        x2 = self.pool2(x1)
        if self.transform:
            self.features.append(self.front2_transform(x2))
        x2 = self.front2(x2)

        x3 = self.pool3(x2)
        if self.transform:
            self.features.append(self.front3_transform(x3))
        x3 = self.front3(x3)

        x4 = self.pool4(x3)
        if self.transform:
            self.features.append(self.front4_transform(x4))
        x4 = self.front4(x4)

        x_brg = self.brg_0(x4)
        if self.transform:
            self.features.append(self.brg_transform(x_brg))

        
        

        #calculate attention maps
        amp_d4 = self.amp_conv4_0(x_brg)#1/16,256
        if self.transform:
            self.features.append(self.amp_conv4_transform(amp_d4))        
        amp4 = self.sgm(self.amp_outconv4(amp_d4))#1/16,1
        
        amp_d3 = F.interpolate(amp_d4, x3.shape[2:], mode='bilinear')#1/8,256
        amp_d3 = self.amp_conv3(amp_d3)#1/8,256
        if self.transform:
            self.features.append(self.amp_conv3_transform(amp_d3))
        amp3 = self.sgm(self.amp_outconv3(amp_d3))#1/8,1
        
        amp_d2 = F.interpolate(amp_d3, x2.shape[2:], mode='bilinear')#1/4,256
        amp_d2 = self.amp_conv2_0(amp_d2)#1/4,128
        if self.transform:
            self.features.append(self.amp_conv2_transform(amp_d2))
        amp2 = self.sgm(self.amp_outconv2(amp_d2))#1/4,1
        
        amp_d1 = F.interpolate(amp_d2, x1.shape[2:], mode='bilinear')#1/2,128
        amp_d1 = self.amp_conv1(amp_d1)#1/2,128
        if self.transform:
            self.features.append(self.amp_conv1_transform(amp_d1))
        amp1 = self.sgm(self.amp_outconv1(amp_d1))#1/2,1
        
        amp_d0 = F.interpolate(amp_d1, x0.shape[2:], mode='bilinear')#1,128
        amp_d0 = self.amp_conv0(amp_d0)#1,64
        if self.transform:
            self.features.append(self.amp_conv0_transform(amp_d0))
        amp0 = self.sgm(self.amp_outconv0(amp_d0))#1,1
        del amp_d4, amp_d3, amp_d2, amp_d1, amp_d0
        
        xb4 = torch.cat([x_brg, x4], 1)#1/16, 1024
        if self.sf:
            self.xb4_before = xb4
        xb4 = xb4 * amp4 # v2
        if self.sf:
            self.xb4_after = xb4
        xb4 = self.back4(xb4) #1/16 size, 512
        
        #calculate density maps
        xb3 = F.interpolate(xb4, size=[x_in.shape[2]//8, x_in.shape[3]//8], mode='bilinear') #1/8 size, 512]
        
        xb3 = torch.cat([x3, xb3], dim=1) #1/8 size, 1024
        if self.sf:
            self.xb3_before = xb3
        xb3 = xb3 * amp3 # v2
        if self.sf:
            self.xb3_after = xb3
        xb3 = self.back3(xb3) #1/8 size, 256
        self.features.append(self.back3_transform(xb3))
        
        xb2 = F.interpolate(xb3, size=[x_in.shape[2]//4, x_in.shape[3]//4], mode='bilinear') #1/4 size, 256
        
        xb2 = torch.cat([x2, xb2], dim=1) #1/4 size, 512
        if self.sf:
            self.xb2_before = xb2
        xb2 = xb2 * amp2 # v2
        if self.sf:
            self.xb2_after = xb2
        xb2 = self.back2(xb2) #1/4 size, 128
        
        xb1 = F.interpolate(xb2, size=[x_in.shape[2]//2, x_in.shape[3]//2], mode='bilinear') #1/2 size, 128
        xb1 = torch.cat([x1, xb1], dim=1)#1/2, 256
        if self.sf:
            self.xb1_before = xb1
        xb1 = xb1 * amp1 # v2
        if self.sf:
            self.xb1_after = xb1
        xb1 = self.back1(xb1)#1/2, 256
        
        xb0 = F.interpolate(xb1, size=x_in.shape[2:], mode='bilinear') #1 size, 64
        xb0 = torch.cat([x0, xb0], 1)#1,128
        if self.sf:
            self.xb0_before = xb0
        xb0 = xb0 * amp0
        if self.sf:
            self.xb0_after = xb0
        xb0 = self.back0(xb0)#1, 64
        
        x_brg = F.interpolate(x_brg, size=x_in.shape[2:], mode='bilinear')
        db = self.outconvb(x_brg)
        xb4 = F.interpolate(xb4, size=x_in.shape[2:], mode='bilinear')
        d4 = self.outconv4(xb4)
        xb3 = F.interpolate(xb3, size=x_in.shape[2:], mode='bilinear')
        d3 = self.outconv3(xb3)
        xb2 = F.interpolate(xb2, size=x_in.shape[2:], mode='bilinear')
        d2 = self.outconv2(xb2)
        xb1 = F.interpolate(xb1, size=x_in.shape[2:], mode='bilinear')
        d1 = self.outconv1(xb1)
        
        dmp = self.output_layer(xb0)
        
        self.features.append(torch.abs(dmp))

        if self.training is True:
            return self.features, [torch.abs(dmp), torch.abs(d1), torch.abs(d2), torch.abs(d3), torch.abs(d4), torch.abs(db), amp4, amp3, amp2, amp1, amp0]
        return torch.abs(dmp), torch.abs(d1), torch.abs(d2), torch.abs(d3), torch.abs(d4), torch.abs(db), amp4, amp3, amp2, amp1, amp0
        

    def _random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights(self):
        if not self.load_model:
            pretrained_dict = dict()
            model_dict = self.state_dict()
            # path = 'models/MARUNet/pretrained/vgg16-397923af.pth'
            # pretrained_model = torch.load(path)
            pretrained_model = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")

            self._random_init_weights()
            # load the pretrained vgg16 parameters
            
            for i, (k, v) in enumerate(pretrained_model.items()):
                #print(i, k)
                  
                if i < 4:
                    layer_id = 0
                    module_id = k.split('.')[-2]
                elif i < 8:
                    layer_id = 1
                    module_id = int(k.split('.')[-2]) - 4
                elif i < 14:
                    layer_id = 2
                    module_id = int(k.split('.')[-2]) - 9
                elif i < 20:
                    layer_id = 3
                    module_id = int(k.split('.')[-2]) - 16
                elif i < 26:
                    layer_id = 4
                    module_id = int(k.split('.')[-2]) - 23
                else:
                    break
                k = 'front' + str(layer_id) + '.' + str(module_id) + '.' + k.split('.')[-1]
                
                if k in model_dict and model_dict[k].size() == v.size():
                    print(k, ' parameters loaded!')
                    pretrained_dict[k] = v
            
            print('weights loaded!')
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model,' loaded!')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def conv_layers(inp, oup, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    return nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
