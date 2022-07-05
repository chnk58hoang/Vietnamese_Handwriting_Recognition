import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from data.dataset import label_to_text
import editdistance


class Trainer():
    def __init__(self, lr_scheduler, patience=5, save_path='checkpoints/best_model.pth', best_val_loss=float('inf')):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = best_val_loss
        self.counter = 0
        self.min_delta = 1e-2
        self.stop = False

    def __call__(self, current_valid_loss, model, epoch, optimizer):
        if self.best_val_loss - current_valid_loss > self.min_delta:
            print(f'Validation loss improved from {self.best_val_loss} to {current_valid_loss}!')
            self.best_val_loss = current_valid_loss
            self.counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.save_path)

        else:
            self.counter += 1
            print(
                f'Validation loss did not improve from {self.best_val_loss}! Counter {self.counter} of {self.patience}.')
            if self.counter < self.patience:
                self.lr_scheduler.step(current_valid_loss)

            else:
                self.stop = True


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
    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            images = data[0].to(device)
            targets = data[1].to(device)
            target_lengths = data[2].to(device)

            _, loss = model(images, targets, target_lengths)
            valid_loss += loss.item()

        return valid_loss / len(dataloader)


def inference(model, device, dataset, mode,decoder):
    model.eval()
    subset_indices = torch.randint(size=(3,), low=0, high=len(dataset))

    subset = Subset(dataset, indices=subset_indices)
    dataloader = DataLoader(subset, batch_size=1)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            images = data[0].to(device)
            labels = data[1].to(device)

            log_probs, _ = model(images)

            decoded_seqs = decoder(log_probs)

            for seq in decoded_seqs:
                all_preds.append(seq)

            for label in labels:
                all_labels.append(label_to_text(label))

        mean_norm_ed = 0.0
        for i in range(len(all_labels)):
            print("Prediction: {0:70} Label: {1}".format(all_preds[i],
                                                             all_labels[i]))
            mean_norm_ed += editdistance.eval(all_preds[i], all_labels[i])
            mean_norm_ed /= len(all_labels[i])
        mean_norm_ed /= len(all_labels)
        print(f'Mean Normalized Edit Distance: {mean_norm_ed}')
