"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import torch.nn as nn
import torchvision.transforms as transforms
import math

from model.BNN_LRTlayers import BBBLinear, BBBConv2d, FlattenLayer, ModuleWrapper

__all__ = ["PreResNet110", "PreResNet56", "PreResNet8", "PreResNet83", "PreResNet164"]

ac_type = 'softplus'
# ac_type = 'relu'


class ShortCut(ModuleWrapper):
    def __init__(self, in_planes, planes, stride=1):
        super(ShortCut, self).__init__()
        self.conv = BBBConv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        # self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, reuse_eps=False):
        # return self.bn(self.conv(x, reuse_eps=reuse_eps))
        return self.conv(x, reuse_eps=reuse_eps)

class LayerList(ModuleWrapper):
    def __init__(self, model_lst=[]):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(model_lst)

    def forward(self, x, reuse_eps=False):
        for i, l in enumerate(self.layers):
            x = l(x, reuse_eps=reuse_eps)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    # return nn.Conv2d(
    #     in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    # )
    return BBBConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Softplus()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, reuse_eps=False):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out, reuse_eps=reuse_eps)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out, reuse_eps=reuse_eps)

        if self.downsample is not None:
            residual = self.downsample(x, reuse_eps=reuse_eps)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = BBBConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(
        #     planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        # )
        self.conv2 = BBBConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3 = BBBConv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Softplus()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, reuse_eps=False):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out, reuse_eps=reuse_eps)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out, reuse_eps=reuse_eps)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out, reuse_eps=reuse_eps)

        if self.downsample is not None:
            residual = self.downsample(x, reuse_eps=reuse_eps)

        out += residual

        return out


class PreResNet(nn.Module):
    def __init__(self, num_classes=10, depth=110):
        super(PreResNet, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, "depth should be 9n+2"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, "depth should be 6n+2"
            n = (depth - 2) // 6
            block = BasicBlock

        self.inplanes = 16
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.conv1 = BBBConv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Softplus()
        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = BBBLinear(64 * block.expansion, num_classes)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2.0 / n))
            # el
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(
            #         self.inplanes,
            #         planes * block.expansion,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     )
            # )
            downsample = ShortCut(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return LayerList(layers)

    def forward(self, x, reuse_eps=False):
        x = self.conv1(x, reuse_eps=reuse_eps)

        x = self.layer1(x, reuse_eps=reuse_eps)  # 32x32
        x = self.layer2(x, reuse_eps=reuse_eps)  # 16x16
        x = self.layer3(x, reuse_eps=reuse_eps)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, reuse_eps=reuse_eps)
        kl = 0.0
        count_kl = 1
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                _kl = module.kl_loss()
                # if torch.isinf(_kl):
                #     import pdb; pdb.set_trace()
                kl = kl + _kl
                count_kl += 1
        kl = kl / count_kl

        return x, kl


class PreResNet164:
    base = PreResNet
    args = list()
    kwargs = {"depth": 164}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def PreResNet164_model(num_classes=10):
    return PreResNet(num_classes=num_classes, depth=164)


class PreResNet110:
    base = PreResNet
    args = list()
    kwargs = {"depth": 110}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class PreResNet83:
    base = PreResNet
    args = list()
    kwargs = {"depth": 83}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class PreResNet56:
    base = PreResNet
    args = list()
    kwargs = {"depth": 56}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class PreResNet8:
    base = PreResNet
    args = list()
    kwargs = {"depth": 8}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
