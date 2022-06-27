import torch

from network.cnn_blocks import CNN_Block
from network.bidirectional_lstm import Bidirectional_LSTM
from data.dataset import num_letters
from torchvision.models import vgg16
from torch.nn import functional as F
import torch.nn as nn


class VietOCRModel(nn.Module):
    def __init__(self):
        super(VietOCRModel, self).__init__()
        self.conv_net = self.create_convnet()
        self.bi_lstm = Bidirectional_LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_letters)
        self.loss_fn = nn.CTCLoss(blank=0)


    def create_convnet(self):
        conv_layers = []
        conv_layers += [CNN_Block(in_channels=3, out_channels=32, kernel_size=3, stride=1)]
        conv_layers += [CNN_Block(in_channels=32, out_channels=64, kernel_size=3, stride=2)]
        conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)]
        conv_layers += [CNN_Block(in_channels=64, out_channels=128, kernel_size=3, stride=2)]
        conv_layers += [CNN_Block(in_channels=128, out_channels=256, kernel_size=3, stride=2)]
        conv_layers += [CNN_Block(in_channels=256, out_channels=512, kernel_size=3, stride=2)]

        return nn.Sequential(*conv_layers)

    def forward(self, x, target=None, target_length=None):
        x = self.conv_net(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bi_lstm(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 2)

        if target != None and target_length != None:
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return x, None


class VietOCRVGG16(nn.Module):
    def __init__(self, finetune=False):
        super(VietOCRVGG16, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = finetune
        self.fc1 = nn.Linear(2560, 512)
        self.bi_lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(512, num_letters)
        self.loss_fn = nn.CTCLoss(blank=0)

    def forward(self, x, target=None, target_length=None):
        x = self.vgg16.features(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x, _ = self.bi_lstm(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 2)

        if target != None and target_length != None:
            x = x.permute(1, 0, 2)
            input_length = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long)
            loss = self.loss_fn(x, target, input_length, target_length)
            return x, loss

        return x, None

