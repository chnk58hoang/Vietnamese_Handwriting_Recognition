import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
from PIL import Image

letters = " #'%()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
num_letters = len(letters) + 1

label_dict = {letters.index(c) + 1 :c  for c in letters}
keys_list = list(label_dict.keys())
values_list = list(label_dict.values())

valid_transform = A.Compose([
    A.Normalize()])


def label_to_text(label):
    return "".join([label_dict[int(c)] for c in label])


def text_to_label(text):
    return [keys_list[values_list.index(c)] for c in text]


class VietOCR(Dataset):
    def __init__(self, image_path, image_list, label_path, img_w, img_h, phase):
        self.image_path = image_path
        self.label_path = label_path
        self.all_images = image_list
        self.all_labels = json.load(open(label_path)) if label_path != None else None
        self.img_w = img_w
        self.img_h = img_h
        self.phase = phase

        if self.phase == 'train':
            self.transform = A.Compose([
                A.GaussNoise(var_limit=(100, 200)),
                A.Cutout(num_holes=15, max_h_size=20, max_w_size=20),
                A.ElasticTransform(alpha_affine=0.5, alpha=1, sigma=0),
                A.Normalize(), A.Resize(self.img_h, self.img_w)])
        else:
            self.transform = A.Compose([
                A.Normalize(), A.Resize(self.img_h, self.img_w)])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # image
        image_dir = os.path.join(self.image_path, self.all_images[index])
        image = Image.open(image_dir).convert("RGB")
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2,0,1)

        # label

        label = self.all_labels[self.all_images[index]]
        label = text_to_label(label)

        label_length = len(label)

        return image, torch.tensor(label), torch.tensor(label_length)


def my_collate_fn(batch):
    (imgs,labels,label_lens) = zip(*batch)
    all_imgs = torch.stack([img for img in imgs],dim=0)
    all_labels = pad_sequence([torch.tensor(label) for label in labels],batch_first=True)
    all_lens = torch.tensor([length for length in label_lens])
    return all_imgs,all_labels,all_lens