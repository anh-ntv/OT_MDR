import torch
import torch.nn as nn
import sys, os
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.append(parentdir)

# from model.datasets import get_normalize_layer

from model.lenet import LeNet
from model.resnet import ResNet18, ResNet50, ResNet10
from model.preact_resnet import PreActResNet18, PreActResNet50
from model.vgg import Vgg16
from model.googlenet import GoogLeNet
from model.efficientnet import EfficientNetB0
from model.mobilenet import MobileNet
from model.wideresnet import WideResNet

from model.resnet_tiny_imagenet import ResNet18_tiny_imagenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ens_dict = {
    'WRNx2': ['wideresnet', 'wideresnet'],
    'WRNx3': ['wideresnet', 'wideresnet', 'wideresnet'],
    'WRNx4': ['wideresnet', 'wideresnet', 'wideresnet', 'wideresnet'],

    'Lex3': ['lenet', 'lenet', 'lenet'],

    'Mox3': ['mobilenet', 'mobilenet', 'mobilenet'],
    'Mox4': ['mobilenet', 'mobilenet', 'mobilenet', 'mobilenet'],
    'Mox5': ['mobilenet', 'mobilenet', 'mobilenet', 'mobilenet', 'mobilenet'],

    'Effx3': ['efficientnet', 'efficientnet', 'efficientnet'],
    'Effx4': ['efficientnet', 'efficientnet', 'efficientnet', 'efficientnet'],
    'Effx5': ['efficientnet', 'efficientnet', 'efficientnet', 'efficientnet', 'efficientnet'],

    'R10x2': ['resnet10', 'resnet10'],
    'R10x3': ['resnet10', 'resnet10', 'resnet10'],
    'R10x4': ['resnet10', 'resnet10', 'resnet10', 'resnet10'],
    'R10x5': ['resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10'],
    'R10x6': ['resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10'],
    'R10x7': ['resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10'],
    'R10x8': ['resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10', 'resnet10'],

    'R18x2': ['resnet18', 'resnet18'],
    'R18x3': ['resnet18', 'resnet18', 'resnet18'],
    'R18x4': ['resnet18', 'resnet18', 'resnet18', 'resnet18'],

    'resvggwide': ['resnet18', 'vgg16', 'wideresnet'],
    'reslemoo': ['resnet18', 'lenet', 'mobilenet'],
    'resvggle': ['resnet18', 'vgg16', 'lenet'],
    'moovggeff': ['mobilenet', 'vgg16', 'efficientnet'],

    'r10r18eff': ['resnet10', 'resnet18', 'efficientnet'],
    'r18mooeff': ['resnet18', 'mobilenet', 'efficientnet'],
    'r18effwide': ['resnet18', 'efficientnet', 'wideresnet'],
    'r10r18wide': ['resnet10', 'resnet18', 'wideresnet'],
    'r10r18vggwide': ['resnet10', 'resnet18', 'vgg16', 'wideresnet'],
    'r18mooeffvggwide': ['resnet18', 'mobilenet', 'efficientnet', 'vgg16', 'wideresnet'],

    'resnet10': ['resnet10'],
    'resnet12': ['resnet12'],
    'resnet18': ['resnet18'],
    'resnet50': ['resnet50'],
    'wideresnet': ['wideresnet'],
    'vgg16': ['vgg16'],
    'lenet': ['lenet'],
    'mobilenet': ['mobilenet'],
    'efficientnet': ['efficientnet'],

}


def ensemble_preds(logits, mode='average_prob', return_probs=False):
    # assert(mode == 'average_prob')
    if mode == 'average_prob':
        output = 0
        probs = []
        for logit in logits:
            prob = torch.softmax(logit, dim=-1)
            output += prob
            probs.append(prob)

        output /= len(logits)
        output = torch.clamp(output, min=1e-40)  # Important, to avoid NaN
        if return_probs:
            return output, probs
        else:
            return output

    elif mode == 'average_logit':
        # average all logits but still return prob 
        output = 0
        probs = []
        for logit in logits:
            output += logit
            prob = torch.softmax(logit, dim=-1)
            probs.append(prob)

        output /= len(logits)
        output = torch.softmax(output, dim=-1)
        output = torch.clamp(output, min=1e-40)  # important, to avoid NaN
        if return_probs:
            return output, probs
        else:
            return output

    else:
        raise ValueError


class EnsembleWrap(nn.Module):
    def __init__(self, models, mode='average_prob'):
        super(EnsembleWrap, self).__init__()
        self.models = models
        self.mode = mode

    def forward(self, x, return_probs=False):
        logits = []
        for model in self.models:
            logit = model(x)
            logits.append(logit)

        output, preds = ensemble_preds(logits, self.mode, return_probs=True)

        if return_probs:
            return output, preds
        else:
            return output

    def parameters(self):
        # Assign parameter of ensemble to optimizer 
        param = list(self.models[0].parameters())
        for model in self.models[1:]:
            param.extend(list(model.parameters()))

        return param


def get_ensemble(arch: str, dataset: str):
    models = []

    assert (arch in ens_dict)

    for m in ens_dict[arch]:
        submodel = get_architecture(m, dataset)
        submodel = nn.DataParallel(submodel)
        submodel = submodel.cuda()
        models.append(submodel)

    return models


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
	  and dividing by the dataset standard deviation.

	  In order to certify radii in original coordinates rather than standardized coordinates, we
	  add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
	  layer of the classifier rather than as a part of preprocessing as is typical.
	  """

    def __init__(self, means, sds):
        """
		:param means: the channel means
		:param sds: the channel standard deviations
		"""
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(input.device)
        # print(input)
        return (input - means) / sds


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
_CIFAR100_STDDEV = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

_MNIST_MEAN = [0.5, ]
_MNIST_STDDEV = [0.5, ]

_TINY_IMAGENET_MEAN = [0.485, 0.456, 0.406]  # [0.4802, 0.4481, 0.3975]
_TINY_IMAGENET_STDDEV = [0.229, 0.224, 0.225]  # [0.2302, 0.2265, 0.2262]


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    elif dataset == 'cifar100':
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)
    elif dataset == 'tiny_imagenet':
        return NormalizeLayer(_TINY_IMAGENET_MEAN, _TINY_IMAGENET_STDDEV)


def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    # assert(dataset == 'cifar10')
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200

    if arch == 'lenet':
        model = LeNet(num_classes=num_classes)
    elif arch == 'resnet10':
        model = ResNet10(num_classes=num_classes)
    elif arch == 'resnet18':
        if dataset == 'tiny_imagenet':
            model = ResNet18_tiny_imagenet(num_classes=num_classes)
        else:
            model = ResNet18(num_classes=num_classes)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif arch == 'preactresnet18':
        model = PreActResNet18(num_classes=num_classes)
    elif arch == 'preactresnet50':
        model = PreActResNet50(num_classes=num_classes)
    elif arch == 'vgg16':
        model = Vgg16(num_classes=num_classes)
    elif arch == 'googlenet':
        model = GoogLeNet(num_classes=num_classes)
    elif arch == 'efficientnet':
        model = EfficientNetB0(num_classes=num_classes)
    elif arch == 'mobilenet':
        model = MobileNet(num_classes=num_classes)
    elif arch == 'wideresnet':
        model = WideResNet(depth=16, num_classes=num_classes, widen_factor=8)

    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model).to(device)
