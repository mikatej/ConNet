import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(stu_map, tea_map):
    similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum()
    return loss


def cal_dense_fsp(features):
    fsp = []
    for groups in features:
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                x = groups[i]
                y = groups[j]

                norm1 = nn.InstanceNorm2d(x.shape[1])
                norm2 = nn.InstanceNorm2d(y.shape[1])
                x = norm1(x)
                y = norm2(y)
                res = gram(x, y)
                fsp.append(res)
    return fsp


def gram(x, y):
    n = x.shape[0]
    c1 = x.shape[1]
    c2 = y.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    x = x.view(n, c1, -1, 1)[0, :, :, 0]
    y = y.view(n, c2, -1, 1)[0, :, :, 0]
    y = y.transpose(0, 1)
    # print (x.shape)
    # print (y.shape)
    # print()
    z = torch.mm(x, y) / (w*h)
    return z

def upsample_process(features, scale=None, ceil_mode=True, x = 60):
    new_features = []

    for i in range(len(features)):
        _, _, x2, _ = features[i].shape

        if (x == x2):
            # print(features[i].shape)
            new_features.append(features[i])
            continue

        ratio = x/x2

        if (ratio > 1):
            scale = nn.Upsample(scale_factor=ratio, mode='bilinear', align_corners=True)
        else:
            # down_ratio = pow(2, x2/x/2)
            ratio = int(x2/x)
            scale = nn.MaxPool2d(kernel_size=ratio, stride=ratio, ceil_mode=ceil_mode)
        new_features.append(scale(features[i]).cuda())
        # print(features[i].shape, ' > ', new_features[-1].shape)

    # print()
    return new_features

def scale_process(features, scale=[3, 2, 1], ceil_mode=True, min_x = 30):
    # process features for multi-scale dense fsp
    new_features = []
    for i in range(len(features)):
        if min_x == None:
            if i >= len(scale):
                # print(features[i].shape)
                new_features.append(features[i])
                continue
            down_ratio = pow(2, scale[i])
        else:
            _, _, x2, _ = features[i].shape
            if (x2 == min_x):
                # print(features[i].shape)
                new_features.append(features[i])
                continue
            down_ratio = int(x2/min_x)

        pool = nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        new_features.append(pool(features[i]))
        # print(features[i].shape, ' > ', new_features[-1].shape)

    # print()
    return new_features
