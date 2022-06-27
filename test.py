import torch
from torchvision.models import resnet50
import torch.nn as nn


model = resnet50(pretrained=False)
newmodel = nn.Sequential(*(list(model.children())[:-2]))
print(newmodel)

x = torch.rand(1,3,160,2560)

print(newmodel(x).size())