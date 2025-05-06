#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from glob import glob
import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
from .utils import create_dir  # Import from the same package

def load_data(path):
    train_x = sorted(glob(os.path.join(path, 'training', 'images', '*.tif')))
    train_y = sorted(glob(os.path.join(path, 'training', '1st_manual', '*.gif')))
    test_x = sorted(glob(os.path.join(path, 'test', 'images', '*.tif')))
    test_y = sorted(glob(os.path.join(path, 'test', '1st_manual', '*.gif')))
    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x.split('/')[-1].split('.')[0]

        x_img = cv2.imread(x, cv2.IMREAD_COLOR)
        y_mask = imageio.mimread(y)[0]
        print(x_img.shape, y_mask.shape)

        X = [x_img]
        Y = [y_mask]

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x_img, mask=y_mask)
            X.append(augmented['image'])
            Y.append(augmented['mask'])

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x_img, mask=y_mask)
            X.append(augmented['image'])
            Y.append(augmented['mask'])

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x_img, mask=y_mask)
            X.append(augmented['image'])
            Y.append(augmented['mask'])

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f'{name}_{index}.png'
            tmp_mask_name = f'{name}_{index}.png'

            image_path = os.path.join(save_path, 'image', tmp_image_name)
            mask_path = os.path.join(save_path, 'mask', tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            index += 1

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        print(len(images_path))

    def __getitem__(self, index):
        # Reading images
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image / 255.0  # (512,512,3)
        image = np.transpose(image, (2, 0, 1))  # (3,512,512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # Reading masks
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # (512,512)
        mask = np.expand_dims(mask, axis=0)  # (1,512,512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    # Example usage of data loading and augmentation
    data_path = '/content/drive/MyDrive/small-retina/'
    (train_x_paths, train_y_paths), (test_x_paths, test_y_paths) = load_data(data_path)
    print(f'Train image paths: {len(train_x_paths)}, mask paths: {len(train_y_paths)}')
    print(f'Test image paths: {len(test_x_paths)}, mask paths: {len(test_y_paths)}')

    output_dir = '/content/drive/MyDrive/new_retinal_data'
    create_dir(os.path.join(output_dir, 'train', 'image'))
    create_dir(os.path.join(output_dir, 'train', 'mask'))
    create_dir(os.path.join(output_dir, 'test', 'image'))
    create_dir(os.path.join(output_dir, 'test', 'mask'))

    augment_data(train_x_paths, train_y_paths, os.path.join(output_dir, 'train'), augment=True)
    augment_data(test_x_paths, test_y_paths, os.path.join(output_dir, 'test'), augment=False)

    train_image_files = sorted(glob(os.path.join(output_dir, 'train', 'image', '*')))
    train_mask_files = sorted(glob(os.path.join(output_dir, 'train', 'mask', '*')))
    valid_image_files = sorted(glob(os.path.join(output_dir, 'test', 'image', '*')))
    valid_mask_files = sorted(glob(os.path.join(output_dir, 'test', 'mask', '*')))

    print(f'Augmented training images: {len(train_image_files)}, masks: {len(train_mask_files)}')
    print(f'Validation images: {len(valid_image_files)}, masks: {len(valid_mask_files)}')

    train_dataset = DriveDataset(train_image_files, train_mask_files)
    valid_dataset = DriveDataset(valid_image_files, valid_mask_files)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5, shuffle=False, num_workers=2)

    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(valid_loader)}')

