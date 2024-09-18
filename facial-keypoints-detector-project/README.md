# Facial Keypoints Detector

## Overview

This project is a **Facial Keypoints Detector**, which locates key points on human faces from images. These keypoints can be used in various applications like facial recognition, emotion detection, and augmented reality. The model was developed as part of the **Udacity Computer Vision Nanodegree**.

## Project Features

- **Dataset**: Preprocessed facial images of size `224x224` with corresponding keypoints.
- **Model Architecture**: A convolutional neural network (CNN) designed for regression of facial keypoints.
- **Custom Image Augmentation**: Implemented transformation functions to ensure data diversity and robustness.
  
## Challenges Faced

### 1. **Model Complexity vs. Overfitting**
   Initially, a deep, highly complex model was implemented with multiple layers and an excessive number of parameters to ensure high performance. However, this approach quickly led to **overfitting**. Despite its accuracy on the training set, the model struggled with generalization, performing poorly on unseen validation data.

   - **Solution**: Regularization techniques like `dropout` and `batch normalization` were introduced to mitigate overfitting. Additionally, techniques such as **data augmentation** were applied to increase the diversity of training images and improve model generalization.

### 2. **Simpler Model vs. Underfitting**
   On the other hand, attempting to reduce the model complexity by limiting the number of layers or parameters led to **underfitting**. This resulted in poor performance on both training and validation sets, as the simpler model was unable to capture complex facial keypoint patterns.

   - **Solution**: Achieving a balance between **model complexity** and **training time** was critical. I experimented with different architectures, finally settling on a mid-sized CNN with a reasonable number of layers, allowing the model to extract rich feature representations without overfitting.

### 3. **Balancing Training and Validation Accuracy**
   The final challenge was ensuring a **suitable training-validation accuracy** tradeoff. The model required tuning of several hyperparameters, including learning rate and batch size. Too small of a learning rate resulted in slow convergence, while a large rate caused the model to overshoot minima, preventing convergence.

   - **Solution**: I adjusted **the learning rate** to finding an optimal point for convergence.

## Key Learnings

- **Model Complexity**: A deep model can perform well but is prone to overfitting. Regularization and data augmentation are crucial in preventing this.
- **Simpler Models**: Simplified models may underfit if they lack the capacity to capture complex patterns.
- **Hyperparameter Tuning**: The right balance between learning rate, batch size, and other hyperparameters is critical for model performance.

## Tools and Technologies

- **OpenCV**
- **PyTorch**
- **Matplotlib**
- 
## Future Work

-Experimenting with more sophisticated CNN architectures like **ResNet** or **EfficientNet** for better accuracy. <br>
-Exploring **transfer learning** to leverage pre-trained models for improved performance. <br>
-Incorporating landmark tracking to build a real-time keypoints detection system.
