# CSRNet with CBAM Attention for Crowd Counting

A deep learning-based crowd counting system using the ShanghaiTech Part A dataset. The project improves a dilated CNN architecture by integrating CBAM attention to focus on relevant crowd regions and suppress background noise.

## Features

- Crowd counting from images
- Density map generation
- VGG-16 feature extraction
- Dilated convolutional backend
- CBAM channel and spatial attention
- MAE and RMSE evaluation

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- h5py
- CUDA

## Dataset

ShanghaiTech Part A with 482 images:
- 300 training images
- 182 testing images

## Results

The proposed VGG-16 + Dilated CNN + CBAM model achieved:

- MAE: **51.10**
- RMSE: **94.91**

The baseline DCNN reported an MAE of 52.6, while the proposed model improved the MAE to 51.10. :contentReference[oaicite:0]{index=0}

## Future Scope

- Evaluation on additional crowd-counting datasets
- Improved domain generalization
- Lightweight architecture for edge deployment
