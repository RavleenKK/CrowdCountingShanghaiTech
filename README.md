# CSRNet with CBAM Attention for Crowd Counting

A deep learning-based crowd counting system using the ShanghaiTech Part A dataset. The project enhances a dilated CNN architecture by integrating CBAM attention to improve focus on relevant crowd regions and suppress background noise.

## Features

- Crowd counting from input images
- Density map generation
- VGG-16 based feature extraction
- Dilated convolutional backend
- CBAM channel and spatial attention
- MAE and RMSE based evaluation

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- h5py
- Matplotlib
- CUDA

## Dataset

ShanghaiTech Part A containing 482 images, with 300 training images and 182 testing images.

## Results

The proposed VGG-16 + Dilated CNN + CBAM model achieved an MAE of 51.10 on ShanghaiTech Part A.

## Future Scope

- Evaluation on additional crowd-counting datasets
- Improved domain generalization
- Lightweight models for edge deployment
