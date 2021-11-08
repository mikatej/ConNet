import torch
import os
from torchvision.models import (alexnet, googlenet,
                                vgg16, vgg16_bn, vgg19, vgg19_bn,
                                resnet18, resnet34,
                                resnet50, resnet101, resnet152,
                                densenet121, densenet169, densenet201)
import torch.nn as nn


def init_weights(model, classifier_only=False):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and not classifier_only:
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')

        elif isinstance(module, nn.BatchNorm2d) and not classifier_only:
            nn.init.constant_(module.weight, val=1)
            nn.init.constant_(module.bias, val=0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.constant_(module.bias, val=0)


def load_pretrained_model(model, model_save_path, pretrained_model):
    """
    loads a pre-trained model from a .pth file
    """
    model.load_state_dict(torch.load(os.path.join(
        model_save_path, '{}.pth'.format(pretrained_model))))


def get_model(model_config,
              imagenet_pretrain,
              model_save_path,
              input_channels,
              class_count):

    model = None

    if model_config == 'AlexNet':
        model = alexnet(pretrained=imagenet_pretrain, progress=True)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=class_count)

    elif model_config == 'GoogleNet':
        model = googlenet(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=1024,
                             out_features=class_count)

    elif model_config == 'VGG16':
        model = vgg16(pretrained=imagenet_pretrain, progress=True)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=class_count)

    elif model_config == 'VGG16_BN':
        model = vgg16_bn(pretrained=imagenet_pretrain, progress=True)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=class_count)

    elif model_config == 'VGG19':
        model = vgg19(pretrained=imagenet_pretrain, progress=True)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=class_count)

    elif model_config == 'VGG19_BN':
        model = vgg19_bn(pretrained=imagenet_pretrain, progress=True)
        model.classifier[6] = nn.Linear(in_features=4096,
                                        out_features=class_count)

    elif model_config == 'ResNet18':
        model = resnet18(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=512,
                             out_features=class_count)
    elif model_config == 'ResNet34':
        model = resnet34(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=512,
                             out_features=class_count)
    elif model_config == 'ResNet50':
        model = resnet50(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=2048,
                             out_features=class_count)
    elif model_config == 'ResNet101':
        model = resnet101(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=2048,
                             out_features=class_count)

    elif model_config == 'ResNet152':
        model = resnet152(pretrained=imagenet_pretrain, progress=True)
        model.fc = nn.Linear(in_features=2048,
                             out_features=class_count)

    elif model_config == 'DenseNet121':
        model = densenet121(pretrained=imagenet_pretrain, progress=True)
        model.classifier = nn.Linear(in_features=1024,
                                     out_features=class_count)
    elif model_config == 'DenseNet169':
        model = densenet169(pretrained=imagenet_pretrain, progress=True)
        model.classifier = nn.Linear(in_features=1664,
                                     out_features=class_count)

    elif model_config == 'DenseNet201':
        model = densenet201(pretrained=imagenet_pretrain, progress=True)
        model.classifier = nn.Linear(in_features=1920,
                                     out_features=class_count)

    if imagenet_pretrain is not True:
        init_weights(model)
    else:
        init_weights(model, classifier_only=True)

    return model
