import os
from decoder.decoder import GreedySearchDecoder, BeamSearchDecoder
from data.dataset import VietOCR, my_collate_fn, num_letters, label_dict
from network.model import VietOCRVGG16
from engine import train_model, valid_model, inference, TrainController
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_path", default="archive/processed_data", type=str)
    parser.add_argument("--label_path", default="archive/finetune_labels.json", type=str)
    parser.add_argument("--ft", type=bool, default=False)
    parser.add_argument("--mode", type=str, default='greedy')

    args = parser.parse_args()

    # Define device

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define network
    model = VietOCRVGG16(num_letters=num_letters, finetune=args.ft).to(device)

    # Define dataset and dataloader

    img_list = os.listdir(args.img_path)
    shuffle(img_list)

    train_img_list = img_list[:int(len(img_list) * 0.8)]
    valid_img_list = img_list[int(len(img_list) * 0.8):]

    train_dataset = VietOCR(image_path=args.img_path, image_list=train_img_list, label_path=args.label_path, img_w=2560,
                            img_h=160, phase='train')
    valid_dataset = VietOCR(image_path=args.img_path, image_list=valid_img_list, label_path=args.label_path, img_w=2560,
                            img_h=160, phase='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    # Define optimizer,scheduler
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.005)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Define decoder

    if args.mode == 'greedy':
        decoder = GreedySearchDecoder(labels=label_dict)
    elif args.mode == 'beam':
        decoder = BeamSearchDecoder(labels=label_dict)

    # Define train controller
    train_controller = TrainController(lr_scheduler)

    for epoch in range(args.epoch):
        print(f'Epoch: {epoch + 1}/{args.epoch}')
        train_loss = train_model(model, device, train_dataset, train_dataloader, optimizer)
        print(f'Training loss: {train_loss}')
        val_loss = valid_model(model, device, valid_dataset, valid_dataloader)
        print(f'Validation loss: {val_loss}')
        inference(model, device, valid_dataset, args.batch_size, decoder)
        train_controller(val_loss, epoch, model, optimizer)
        if train_controller.stop:
            print('Early stopped')
            break
