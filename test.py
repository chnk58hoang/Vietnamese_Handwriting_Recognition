import torch

from data.dataset import label_to_text,text_to_label,label_dict

x = [1,2,3,4,5,6]
print(label_to_text(x))