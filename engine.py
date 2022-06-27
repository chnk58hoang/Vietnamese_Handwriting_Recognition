import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from decoder.decoder import GreedySearchDecoder, BeamSearchDecoder
from data.dataset import label_dict, label_to_text, my_collate_fn
import editdistance


def train_model(model, device, dataset, dataloader, optimizer):
    model = model.to(device)
    model.train()
    train_loss = 0.0
    for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        optimizer.zero_grad()

        images = data[0].to(device)
        targets = data[1].to(device)
        target_lengths = data[2].to(device)

        _, loss = model(images, targets, target_lengths)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


def valid_model(model, device, dataset, dataloader):
    model = model.to(device)
    model.eval()
    valid_loss = 0.0
    for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        images = data[0].to(device)
        targets = data[1].to(device)
        target_lengths = data[2].to(device)

        _, loss = model(images, targets, target_lengths)
        valid_loss += loss.item()

    return valid_loss / len(dataloader)


def inference(model, device, dataset, mode):
    subset_indices = torch.randint(size=(3,), low=0, high=len(dataset))

    subset = Subset(dataset, indices=subset_indices)
    dataloader = DataLoader(subset, batch_size=3, collate_fn=my_collate_fn)

    if mode == 'greedy':
        decoder = GreedySearchDecoder(labels=label_dict)
    elif mode == 'beam':
        decoder = BeamSearchDecoder(labels=label_dict)

    all_preds = []
    all_labels = []

    for batch, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)

        log_probs, _ = model(images)

        decoded_seqs = decoder(log_probs)

        print('Predictions' + '-' * 50)
        for seq in decoded_seqs:
            all_preds.append(seq)
            print(seq)

        print('Labels' + '-' * 50)

        for label in labels:
            all_labels.append(label_to_text(label))
            print(label)

    mean_norm_ed = 0.0
    for i in range(len(all_preds)):
        mean_norm_ed += editdistance.eval(all_preds[i], all_labels[i])
        mean_norm_ed /= len(all_labels[i])
    mean_norm_ed /= len(all_labels)
    print(f'Mean Normalized Edit Distance{mean_norm_ed}')
