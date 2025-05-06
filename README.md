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
    - [Installation](#installation)
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

## Introduction

This project leverages the U-Net architecture, a popular convolutional neural network designed for biomedical image segmentation. The model is trained on a dataset of retinal images with corresponding ground truth vessel segmentations. The goal is to accurately identify and delineate blood vessels in new, unseen retinal images.

## File Structure

