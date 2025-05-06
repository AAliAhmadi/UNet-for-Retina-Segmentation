#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from .model import build_unet
from .data import DriveDataset, load_data, augment_data
from .utils import DiceLoss, epoch_time, create_dir, seeding

# Hyperparameters (consider moving these to a config file later)
H = 512
W = 512
size = (H, W)
batch_size = 5
num_epochs = 50
lr = 1e-4
checkpoint_path = '/content/drive/MyDrive/new_retinal_data1/checkpoint.pth'
data_path = '/content/drive/MyDrive/small-retina/'
output_dir = '/content/drive/MyDrive/new_retinal_data'

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss

if __name__ == '__main__':
    seeding(42)
    create_dir(os.path.join(output_dir, 'train', 'image'))
    create_dir(os.path.join(output_dir, 'train', 'mask'))
    create_dir(os.path.join(output_dir, 'test', 'image'))
    create_dir(os.path.join(output_dir, 'test', 'mask'))

    (train_x_paths, train_y_paths), (test_x_paths, test_y_paths) = load_data(data_path)
    augment_data(train_x_paths, train_y_paths, os.path.join(output_dir, 'train'), augment=True)
    augment_data(test_x_paths, test_y_paths, os.path.join(output_dir, 'test'), augment=False)

    train_image_files = sorted(glob(os.path.join(output_dir, 'train', 'image', '*')))
    train_mask_files = sorted(glob(os.path.join(output_dir, 'train', 'mask', '*')))
    valid_image_files = sorted(
        glob(os.path.join(output_dir, 'test', 'image', '*')))
    valid_mask_files = sorted(
        glob(os.path.join(output_dir, 'test', 'mask', '*')))

    train_dataset = DriveDataset(train_image_files, train_mask_files)
    valid_dataset = DriveDataset(valid_image_files, valid_mask_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = DiceLoss()  # if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            data_str = f'Valid Loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}'
            print(data_str)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        epoch_mins, epoch_secs = epoch_time(epoch_start_time, time.time())
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}\n \tVal Loss: {valid_loss:.3f}')

    print(f'Total training time: {(time.time() - start_time) / 60:.2f} min')

    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Training Loss', c='blue')
    plt.plot(np.arange(1, num_epochs + 1), valid_losses, label='Validation Loss', c='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.show()

