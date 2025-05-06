# Retinal Vessel Segmentation using U-Net

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-%3E%3D1.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This project implements a U-Net deep learning model for the segmentation of blood vessels in retinal images. Accurate segmentation of retinal vessels is crucial for the diagnosis and monitoring of various ophthalmological diseases.

## Table of Contents

- [Introduction](#introduction)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Dataset](#dataset)
- [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Training the Model](#training-the-model)
    - [Testing/Inference](#testinginference)
- [Results](#results)
- [Hyperparameters](#hyperparameters)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction 🎉

This project leverages the U-Net architecture, a popular convolutional neural network designed for biomedical image segmentation. The model is trained on a dataset of retinal images with corresponding ground truth vessel segmentations. The goal is to accurately identify and delineate blood vessels in new, unseen retinal images.

## File Structure 🛠️
```
retinal_segmentation/
├── model.py          # Defines the U-Net model architecture.
├── utils.py          # Contains utility functions (seeding, directory creation, loss functions, etc.).
├── data.py           # Handles data loading, augmentation, and the custom Dataset class.
├── train.py          # Script for training the U-Net model.
├── test.py           # Script for evaluating the trained model on the test set.
└── init.py       # Makes 'retinal_segmentation' a Python package.

```


## Getting Started 🚀

### Prerequisites 👇

Before you begin, ensure you have the following installed:

- **Python 3.x** ([https://www.python.org/downloads/](https://www.python.org/downloads/))
- **PyTorch** (>= 1.0) ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
- **torchvision** (installed along with PyTorch)
- **NumPy** (`pip install numpy`)
- **Matplotlib** (`pip install matplotlib`)
- **OpenCV-Python** (`pip install opencv-python`)
- **Imageio** (`pip install imageio`)
- **Scikit-learn** (`pip install scikit-learn`)
- **Albumentations** (`pip install albumentations`)
- **Tqdm** (`pip install tqdm`)

It is highly recommended to have a CUDA-enabled GPU for faster training.


### Dataset 🌎

This project is designed to work with a dataset of retinal images and their corresponding vessel segmentations. The code in `data.py` assumes a specific directory structure (e.g., `/content/small-retina/`).

-   ** ✅ Prepare your dataset: ** Organize your retinal images and their ground truth masks into the following structure (or modify the `load_data` function in `data.py` to match your structure):

    ```
    your_dataset_path/
    ├── training/
    │   ├── images/
    │   │   ├── *.tif
    │   │   └── ...
    │   └── 1st_manual/
    │       ├── *.gif
    │       └── ...
    └── test/
        ├── images/
        │   ├── *.tif
        │   └── ...
        └── 1st_manual/
            ├── *.gif
            └── ...
    ```

-   **Update `data_path`:** Modify the `data_path` variable in `train.py` and `test.py` to point to the root directory of your dataset.

## Usage

### Data Preparation

The `data.py` script handles data loading and augmentation. The `augment_data` function performs horizontal flip, vertical flip, and rotation on the training data to increase the dataset size and improve model robustness.

-   To preprocess and augment the data, you can run the `data.py` script directly (though it's usually called by `train.py`):
    ```bash
    python retinal_segmentation/data.py
    ```
    This will create augmented data in a new directory (`/content/drive/MyDrive/new_retinal_data/` by default). **Make sure to adjust the `output_dir` variable in `data.py` if needed.**

###⚙️ Training the Model

The `train.py` script trains the U-Net model.

-   **✅ Run the training script:**
    ```bash
    python retinal_segmentation/train.py
    ```
    This script will:
    -   Load the prepared training and validation data.
    -   Initialize the U-Net model.
    -   Train the model for the specified number of epochs.
    -   Evaluate the model on the validation set after each epoch.
    -   Save the best model weights (based on validation loss) to the `checkpoint_path`.
    -   Generate a plot of the training and validation loss.

### ✅ Testing/Inference

The `test.py` script evaluates the trained model on the test set.

-   **Run the testing script:**
    ```bash
    python retinal_segmentation/test.py
    ```
    This script will:
    -   Load the trained model weights from the `checkpoint_path`.
    -   Load the test data.
    -   Perform inference on the test images.
    -   Calculate and print evaluation metrics (Jaccard, F1 score, Recall, Precision, Accuracy).
    -   Save visual results (original image, ground truth mask, predicted mask) in the `results_dir`.
    -   Display a sample visualization of a prediction.

## ✨ Results

After running `test.py`, you will find the evaluation metrics printed in the console. Additionally, visual results of the segmentation will be saved in the `results/` directory within your specified `data_path`.

## ✨ Hyperparameters

The following hyperparameters can be found and adjusted in the `train.py` script:

-   `H`, `W`: Image height and width (default: 512).
-   `batch_size`: Number of samples per training batch (default: 5).
-   `num_epochs`: Number of training epochs (default: 50).
-   `lr`: Learning rate for the Adam optimizer (default: 1e-4).
-   `checkpoint_path`: Path to save the best model weights (default: `/content/drive/MyDrive/new_retinal_data1/checkpoint.pth`).
-   `data_path`: Path to the root of your dataset (default: `/content/drive/MyDrive/small-retina/`).
-   `output_dir`: Path to save augmented data and results (default: `/content/drive/MyDrive/new_retinal_data`).

## 🌳 Contributing

Contributions to this project are welcome! If you have suggestions, bug reports, or would like to add new features, please:

1.  Fork the repository.
2.  Create a new branch for your changes (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Create a new Pull Request.

Please ensure your code follows the existing style and includes appropriate comments and documentation.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details. *(You should create a `LICENSE` file in the root directory with the MIT license text).*

## 🌳 Acknowledgments

-   The U-Net architecture was originally proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in the paper: *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
-   We acknowledge the use of libraries such as PyTorch, NumPy, OpenCV, Albumentations, and Scikit-learn, which greatly facilitated the development of this project.
-   *(Optionally, acknowledge the dataset(s) you used if they have specific citation requirements).*

---

Feel free to customize this `README.md` further with more details specific to your project, such as performance metrics on your dataset, visualizations of the model architecture, or links to relevant resources. Good luck with your project!
