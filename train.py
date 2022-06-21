from dataset import VietOCR, my_collate_fn
from model import VietOCRVGG16
from engine import train_model, valid_model
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import argparse


def split_dataset(dataset, ratio):
    len_dataset1 = int(len(dataset) * ratio)
    len_dataset2 = len(dataset) - len_dataset1

    dataset1, dataset2 = random_split(dataset, lengths=[len_dataset1, len_dataset2])

    return dataset1, dataset2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--ft", type=bool, default=False)
    args = parser.parse_args()

    # Define device

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define model
    model = VietOCRVGG16(finetune=args.ft)

    # Define dataset and dataloader
    all_dataset = VietOCR(image_path=args.img_path, label_path=args.label_path, img_h=160, img_w=2560)

    train_val_dataset, test_dataset = split_dataset(all_dataset, ratio=0.95)

    train_dataset, valid_dataset = split_dataset(train_val_dataset, ratio=0.85)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    # Define optimizer,scheduler
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.005)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(args.epoch):
        train_loss = train_model(model, device, train_dataset, train_dataloader, optimizer)
        print(f'Training loss:{train_loss}')
        val_loss = valid_model(model, device, valid_dataset, valid_dataloader)
        print(f'Validation loss:{val_loss}')
        lr_scheduler.step(val_loss)
