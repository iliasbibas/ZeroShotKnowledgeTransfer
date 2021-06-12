import os

from models.lenet import *
from models.wresnet import *
from models.mobilenetv2 import *
from models.vgg import *
from models.googlenet import *
from models.inception import *
from models.densenet import *


def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None):
    if dataset in ['SVHN', 'CIFAR10']:
        n_classes = 10
        if model_name == 'LeNet':
            model = LeNet32(n_classes=n_classes)
        elif model_name == 'WRN-16-1':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-16-2':
            model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        elif model_name == 'WRN-40-1':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
        elif model_name == 'WRN-40-2':
            model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)
        else:
            raise NotImplementedError

        if pretrained:
            model_path = os.path.join(pretrained_models_path, dataset, model_name, "last.pth.tar")
            print('Loading Model from {}'.format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

    else:
        raise NotImplementedError

    return model

def select_model_ours(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None):

    if pretrained_models_path == None:
        raise Exception("Missing pretrained path.")

    if dataset != "CIFAR10":
        raise NotImplementedError

    if model_name == "MobileNetv2":
        model = mobilenet_v2(pretrained, path=pretrained_models_path)
    elif model_name == "vgg11_bn":
        model = vgg11_bn(pretrained, path=pretrained_models_path)
    elif model_name == "vgg13_bn":
        model = vgg13_bn(pretrained, path=pretrained_models_path)
    elif model_name == "vgg19_bn":
        model = vgg19_bn(pretrained, path=pretrained_models_path)
    elif model_name == "googlenet":
        model = googlenet(pretrained, path=pretrained_models_path)
    elif model_name == "inception":
        model = inception_v3(pretrained, path=pretrained_models_path)
    elif model_name == "densenet121":
        model = densenet121(pretrained, path=pretrained_models_path)
    else:
        raise NotImplementedError

    return model

if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import time

    x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, *act = model(x)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))
