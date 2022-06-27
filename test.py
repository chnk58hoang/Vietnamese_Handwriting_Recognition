import torch

from data.dataset import label_to_text,text_to_label,label_dict

x  = torch.tensor([1])
print(int(x))