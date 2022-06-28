import torch
from torchvision.models import resnet50,vgg19
import torch.nn as nn


x  = torch.rand(1,3,160,2560)

model = vgg19()
print(model.features(x).size())