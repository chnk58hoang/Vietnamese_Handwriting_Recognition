import torch
import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

letters = " #'%()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
num_letters = len(letters) + 1

label_dict = {c : letters.index(c) + 1 for c in letters}
keys_list = list(label_dict.keys())
values_list = list(label_dict.values())


def text_to_label(text):
    return [label_dict[c] for c in text]


def label_to_text(labels):

    return "".join([keys_list[values_list.index(label)] for label in labels])




class VietOCR(Dataset):
    def __init__(self, image_path, label_path, img_w, img_h):
        self.image_path = image_path
        self.label_path = label_path
        self.all_images = os.listdir(image_path)
        self.all_labels = json.load(open(label_path)) if label_path != None else None
        self.img_w = img_w
        self.img_h = img_h
        self.transform = transforms.Compose(
            [transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # image
        image_dir = os.path.join(self.image_path, self.all_images[index])
        image = Image.open(image_dir).convert("RGB")
        image = image.resize((self.img_w,self.img_h),resample=Image.BILINEAR)
        image = self.transform(image)

        # label

        label = self.all_labels[self.all_images[index]]
        label = text_to_label(label)

        label_length = len(label)

        return image, label, label_length


def my_collate_fn(batch):
    labels = []
    imgs = []
    label_lengths = []
    max_label_length = 0
    for sample in batch:
        max_label_length = max(max_label_length, sample[2])

    for sample in batch:
        label = sample[1]
        label += [1] * (max_label_length - sample[2])

        labels.append(torch.tensor(label))
        imgs.append(sample[0])
        label_lengths.append(sample[2])

    imgs = torch.stack(imgs, dim=0)
    label_lengths = torch.tensor(label_lengths)
    labels = torch.stack(labels, dim=0)

    return imgs, labels, label_lengths
