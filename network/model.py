import torch
from data.dataset import num_letters
from torchvision.models import vgg16,resnet50
from torch.nn import functional as F
import torch.nn as nn


class VietOCRVGG16(nn.Module):
    def __init__(self, finetune=False):
        super(VietOCRVGG16, self).__init__()
        resnet = resnet50(pretrained=False)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))

        for param in self.backbone.parameters():
            param.requires_grad = finetune
        self.fc1 = nn.Linear(10240, 2048)
        self.fc2 = nn.Linear(2048,512)
        self.bi_lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(512, num_letters)
        self.loss_fn = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x, _ = self.bi_lstm(x)
        x = self.fc3(x)
        x = F.log_softmax(x, 2)

        if target != None and target_length != None:
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return x, None

