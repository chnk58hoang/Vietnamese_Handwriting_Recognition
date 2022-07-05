import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from data.dataset import my_collate_fn,label_to_text
import editdistance


class TrainController():
    def __init__(self, lr_scheduler, best_valid_loss=float('inf'), patience=5, save_path='checkpoints/best_model.pth',
                 min_delta=1e-2):
        self.best_valid_loss = best_valid_loss
        self.patience = patience
        self.counter = 0
        self.lr_scheduler = lr_scheduler
        self.save_path = save_path
        self.stop = False
        self.min_delta = min_delta

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if self.best_valid_loss - current_valid_loss >= self.min_delta:
            print(f"\nValidation loss improved from {self.best_valid_loss} to {current_valid_loss}")
            self.best_valid_loss = current_valid_loss
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.save_path)
            self.counter = 0

        else:
            self.counter += 1
            if self.counter < self.patience:
                print(
                    f"\nValidation loss did not improve from {self.best_valid_loss}. Early stop counter {self.counter} of {self.patience}")
                self.lr_scheduler.step(current_valid_loss)

            else:
                self.stop = True


def train_model(model, device, dataset, dataloader, optimizer):
    model.train()
    train_loss = 0.0
    counter = 0
    for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
        optimizer.zero_grad()
        images = data[0].to(device)
        targets = data[1].to(device)
        target_lengths = data[2].to(device)

        _, loss = model(images, targets, target_lengths)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss / counter


def valid_model(model, device, dataset, dataloader):
    model.eval()
    valid_loss = 0.0
    counter = 0
    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            images = data[0].to(device)
            targets = data[1].to(device)
            target_lengths = data[2].to(device)

            _, loss = model(images, targets, target_lengths)
            valid_loss += loss.item()

        return valid_loss / counter


def inference(model, device, dataset, batch_size, probs_decoder):
    model.eval()
    subset_indices = torch.randint(size=(3,), low=0, high=len(dataset))

    subset = Subset(dataset, indices=subset_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=my_collate_fn)

    mean_norm_ed = 0.0

    for data in dataloader:
        images = data[0].to(device)
        targets = data[1].to(device)
        target_lengths = data[2].to(device)

        probs,_ = model(images)
        results = probs_decoder(probs)

        for i in range(len(results)):
            print("Predictions:")
            print(results[i])
            print("Labels")
            text = label_to_text(targets[i])
            print(text)
            mean_norm_ed += editdistance.eval(results[i], text) / target_lengths[i]

        mean_norm_ed /= dataloader.batch_size

    print(f"Mean Normalized Editdistance: {mean_norm_ed}")
