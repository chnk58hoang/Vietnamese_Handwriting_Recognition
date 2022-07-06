import torch
from torchvision.models import vgg16, resnet50
from torch.nn import functional as F
import torch.nn as nn


class VietOCRVGG16(nn.Module):
    def __init__(self, num_letters, finetune=False):
        super(VietOCRVGG16, self).__init__()
        self.backbone = vgg16(pretrained=True)

       
        for param in self.backbone.parameters():
            param.requires_grad = finetune

        self.fc1 = nn.Linear(2560, 512)
        self.bi_lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(1024, num_letters)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.backbone.features(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.fc1(x))
        x, _ = self.bi_lstm(x)

        if target != None and target_length != None:
            x = self.log_softmax(self.fc3(x))
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return self.softmax(self.fc3(x)), None