import torch
from torchvision.models import resnet50,vgg19
import torch.nn as nn
from data.dataset import label_to_text

x = [1,2,3,4,5,5,7,0,0,0,0]

print(label_to_text(x))