import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
from PIL import Image
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('vocab/m.model')


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
                A.GaussNoise(var_limit=(100, 200), mean=30),
                A.Cutout(num_holes=15, max_h_size=20, max_w_size=20, fill_value=(255, 255, 255)),
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
        image = image.permute(2, 0, 1)

        # label

        label = self.all_labels[self.all_images[index]]
        label = sp.encode_as_ids(label)

        label_length = len(label)

        return image, label, label_length


def my_collate_fn(batch):
    (images, labels, label_lengths) = zip(*batch)

    all_imgs = torch.stack([img for img in images], dim=0)
    all_labels = [torch.tensor(label) for label in labels]

    all_labels = pad_sequence(all_labels, batch_first=True, padding_value=0)

    all_lengths = torch.tensor([length for length in label_lengths])

    return all_imgs, all_labels, all_lengths
