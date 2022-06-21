import torch
import torch.nn as nn
from dataset import label_dict

class GreedyDecoder(nn.Module):
    def __init__(self, labels, blank=0):
        super(GreedyDecoder, self).__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, probs):
        results = []
        indices = torch.argmax(probs, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        for index in indices:
            index = [i for i in index if i != self.blank]
            joined = "".join([self.labels[i] for i in index])
            results.append(joined)

        return results

