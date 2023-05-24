'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.BNN_LRTlayers import BBBLinear, BBBConv2d, FlattenLayer, ModuleWrapper
from model.BNNLayers import BBBLinear, BBBConv2d, FlattenLayer, ModuleWrapper


ac_type = 'softplus'
# ac_type = 'relu'


class ShortCut(ModuleWrapper):
    def __init__(self, in_planes, planes, stride=1):
        super(ShortCut, self).__init__()
        self.conv = BBBConv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, reuse_eps=False):
        return self.bn(self.conv(x, reuse_eps=reuse_eps))


class BasicBlock(ModuleWrapper):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = BBBConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BBBConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = LayerList()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ShortCut(in_planes, self.expansion*planes, stride)
            # self.shortcut = nn.Sequential(
            #     BBBConv2d(in_planes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion*planes)
            # )
        if ac_type == "softplus":
            act_fn = nn.Softplus
        else:
            act_fn = nn.ReLU

        self.act1 = act_fn()
        self.act2 = act_fn()

    def forward(self, x, reuse_eps=False):
        out = self.act1(self.bn1(self.conv1(x, reuse_eps=reuse_eps)))
        out = self.bn2(self.conv2(out, reuse_eps=reuse_eps))
        out += self.shortcut(x, reuse_eps=reuse_eps)
        out = self.act2(out)
        return out


class Bottleneck(ModuleWrapper):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = BBBConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BBBConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BBBConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = LayerList()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ShortCut(in_planes, self.expansion*planes, stride)
            # self.shortcut = nn.Sequential(
            #     BBBConv2d(in_planes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion*planes)
            # )
        if ac_type == "softplus":
            act_fn = nn.Softplus
        else:
            act_fn = nn.ReLU

        self.act1 = act_fn()
        self.act2 = act_fn()
        self.act3 = act_fn()

    def forward(self, x, reuse_eps=False):
        out = self.act1(self.bn1(self.conv1(x, reuse_eps=reuse_eps)))
        out = self.act2(self.bn2(self.conv2(out, reuse_eps=reuse_eps)))
        out = self.bn3(self.conv3(out, reuse_eps=reuse_eps))
        out += self.shortcut(x, reuse_eps=reuse_eps)
        out = self.act3(out)
        return out


class LayerList(ModuleWrapper):
    def __init__(self, model_lst=[]):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(model_lst)

    def forward(self, x, reuse_eps=False):
        for i, l in enumerate(self.layers):
            x = l(x, reuse_eps=reuse_eps)
        return x


class ResNet(ModuleWrapper):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = BBBConv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = BBBLinear(512*block.expansion, num_classes)

        if ac_type == "softplus":
            act_fn = nn.Softplus
        else:
            act_fn = nn.ReLU

        self.act = act_fn()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)
        return LayerList(layers)

    def forward(self, x, reuse_eps=False):
        out = self.act(self.bn1(self.conv1(x, reuse_eps=reuse_eps)))
        out = self.layer1(out, reuse_eps=reuse_eps)
        out = self.layer2(out, reuse_eps=reuse_eps)
        out = self.layer3(out, reuse_eps=reuse_eps)
        out = self.layer4(out, reuse_eps=reuse_eps)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, reuse_eps=reuse_eps)
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
        return out, kl

    def get_kl(self):
        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        return kl


def ResNet10(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet12(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 2, 1], num_classes=num_classes)

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
