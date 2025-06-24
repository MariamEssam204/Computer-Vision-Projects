# Computer Vision Projects Repository

## Overview
A comprehensive collection of computer vision projects spanning multiple domains including image captioning and classification, medical imaging, and robotics. Each project includes implementation code,
trained models (where applicable), and detailed documentation.

## 📂 Project Catalog

### 🖼️ Core Vision Tasks

#### Facial Recognition
| Project | Method | Architecture | Dataset |
|---------|--------|----------|-------------|
| [Facial-keypoints-detector](./facial-keypoints-detector-project) | CNN from the scratch | two convolutional layers with batch normalization and max pooling, followed by fully connected layers. It processes grayscale images to predict 136 output values (68 (x,y) keypoint pairs) with dropout regularization to prevent overfitting. | Collected images |


#### Image Captioning
| Project | Architecture | Description | Dataset |
|---------|-------------|--------|---------|
| [Image Caption](./Image-Caption-project) | ResNet50 + LSTM | CNN to encode the images to a feature map, then pass it to an embedding layer then to be decoded by LSTM | COCO |


### 🤖 Robotics Applications
| Project | Description | Technologies |
|---------|-------------|--------------|
| [1D Localization system](./Object-Detection-notebooks/Robot-Localization) |System that calculate the likehood of the location of a robot taking in the sensor's reading and motion direction. | python |
| [1D Kalman Filter](./Object-Detection-notebooks/Kalman-Filter) |A lightweight, well-tested implementation of a one-dimensional Kalman filter in Python. Built for educational clarity and robustness, with extensive validation against edge cases. | python |


### � Medical Imaging
| Project | Description | Technologies | Dataset |
|---------|-------------|--------------|---------|
| [Breast Cancer Detection](./INbreast-bresat-cancer-detection) | Mammogram classification | ResNet50 | CBIS-DDSM |



## 🛠️ Installation

```bash
git clone https://github.com/MariamEssam204/Computer-Vision-Projects.git
cd Computer-Vision-Projects

