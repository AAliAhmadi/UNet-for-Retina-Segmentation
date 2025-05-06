#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from .model import build_unet
from .utils import mask_parse

# Hyperparameters (should match training)
H = 512
W = 512
size = (H, W)
checkpoint_path = '/content/drive/MyDrive/new_retinal_data1/checkpoint.pth'
data_path = '/content/drive/MyDrive/new_retinal_data'
results_dir = os.path.join(data_path, 'results')
os.makedirs(results_dir, exist_ok=True)

def calculate_metrics(y_true, y_predict):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8).reshape(-1)

    y_predict = y_predict.detach().cpu().numpy()
    y_predict = y_predict > 0.5
    y_predict = y_predict.astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_predict)
    score_f1 = f1_score(y_true, y_predict)
    score_recall = recall_score(y_true, y_predict)
    score_precision = precision_score(y_true, y_predict)
    score_acc = accuracy_score(y_true, y_predict)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_x = sorted(glob(os.path.join(data_path, 'test', 'image', '*')))
    test_y = sorted(glob(os.path.join(data_path, 'test', 'mask', '*')))

    metrics_scores = []
    jaccard_scores = []
    f1_scores = []
    accuracy_scores = []
    recall_scores = []
    precision_scores = []

    with torch.no_grad(), tqdm(total=len(test_x)) as pbar:
        for i, (img_path, mask_path) in enumerate(zip(test_x, test_y)):
            name = img_path.split('/')[-1].split('.')[0]

            # Reading image
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            original_image = image.copy()
            x = np.transpose(image, (2, 0, 1))
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            x = x.astype(np.float32)
            x = torch.from_numpy(x).to(device)

            # Reading mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            y_true = mask / 255.0
            y_true = np.expand_dims(y_true, axis=0)
            y_true_tensor = torch.from_numpy(y_true).unsqueeze(0).to(device)

            # Prediction
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)

            # Calculate metrics
            metrics = calculate_metrics(y_true_tensor, y_pred)
            jaccard_scores.append(metrics[0])
            f1_scores.append(metrics[1])
            recall_scores.append(metrics[2])
            precision_scores.append(metrics[3])
            accuracy_scores.append(metrics[4])
            metrics_scores.append(metrics)

            # Save results
            original_mask = mask_parse(mask)
            predicted_mask = mask_parse((y_pred.cpu().numpy()[0] > 0.5).astype(np.uint8) * 255)
            line = np.ones((size[1], 10, 3)) * 128
            concatenated_images = np.concatenate([original_image, line, original_mask, line, predicted_mask], axis=1)
            cv2.imwrite(os.path.join(results_dir, f'{name}_result.png'), concatenated_images)

            pbar.update(1)

    mean_jaccard = np.mean(jaccard_scores)
    mean_f1 = np.mean(f1_scores)
    mean_recall = np.mean(recall_scores)
    mean_precision = np.mean(precision_scores)
    mean_accuracy = np.mean(accuracy_scores)

    print(f'Jaccard: {mean_jaccard:.4f} - F1: {mean_f1:.4f} - Recall: {mean_recall:.4f} - Precision: {mean_precision:.4f} - Accuracy: {mean_accuracy:.4f}')

    # Example of visualizing a single prediction (moved here for clarity)
    if test_x:
        sample_img_path = test_x[1]
        sample_mask_path = test_y[1]

        image = cv2.imread(sample_img_path, cv2.IMREAD_COLOR)
        x = np.transpose(image, (2, 0, 1)) / 255.0
        x = np.expand_dims(x, axis=0).astype(np.float32)
        x_tensor = torch.from_numpy(x).to(device)

        with torch.no_grad():
            output = model(x_tensor)
            out_np = torch.sigmoid(output).cpu().numpy()[0][0]  # Get the single channel output

        true_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title('True Segmentation')

        plt.subplot(1, 3, 3)
        plt.imshow(out_np, cmap='gray')
        plt.title('Model Prediction')

        plt.tight_layout()
        plt.show()

