from tqdm import tqdm


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


def inference(model,device,dataset):
    pass
