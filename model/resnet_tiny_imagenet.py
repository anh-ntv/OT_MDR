import torchvision.models as models
import torch.nn as nn

def ResNet18_tiny_imagenet(num_classes=200):
    """
    Ref: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/ResNet18_224_finetune.ipynb
    """
    #Load Resnet18
    model_ft = models.resnet18(pretrained=False)
    #Finetune Final few layers to adjust for tiny imagenet input
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

