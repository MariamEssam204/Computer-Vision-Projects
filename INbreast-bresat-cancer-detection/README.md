# Breast Cancer Detection from Mammograms (INbreast Dataset)

## Project Overview

This project focuses on developing a machine learning model to classify breast tumors as **benign** or **malignant** using the INbreast dataset of mammogram images in **DICOM (Digital Imaging and Communications in Medicine)** format. The goal is to support early breast cancer detection by using medical image analysis and predictive models, potentially aiding radiologists in making accurate diagnoses.

The project involves extracting important metadata from DICOM files, preprocessing the mammogram images, and connecting these data to clinical labels for classification.

---

## Dataset

The project utilizes the **INbreast dataset**, which consists of **411 DICOM mammogram images** accompanied by an Excel file containing vital clinical metadata. The metadata includes:

- **BI-RADS Assessment**: A standardized scale used by radiologists to rate the likelihood of breast cancer. Higher values (4-5) indicate higher malignancy risk.
- **ACR Breast Density**: A measure of breast tissue density, with denser tissue increasing the difficulty of tumor detection.
- **Lesion Type (Mass or Calcification)**: Describes the type of abnormality seen in the mammogram.
- **Laterality**: Indicates whether the mammogram is of the left or right breast.
- **View**: Mammogram imaging angles (e.g., CC - Cranio-Caudal, MLO - Mediolateral Oblique).
- **Mass Shape, Margins, and Density**: Describes the physical characteristics of masses in the mammogram.

---

## Workflow

1. **Data Loading**: 
   - Load the DICOM images from the dataset.
   - Load the Excel metadata file for corresponding labels and clinical information.
   
2. **Preprocessing**:
   - Randomly select DICOM images and extract their pixel data and metadata.
   - Convert the pixel array to a suitable format for image processing.
   - Connect each DICOM image with the corresponding metadata from the Excel file.

3. **Image Processing**:
   - Resize images and convert to grayscale (as needed for the model).
   - Extract image features such as texture, shape, and pixel intensity.

4. **Model Development**:
   - Merge metadata (e.g., BI-RADS, lesion type) with image data.
   - Develop a machine learning model to predict whether a tumor is **benign** or **malignant** depending on BI-RADS

---

## Challenges Faced

### 1. **DICOM Metadata Extraction**
   - Initially, extracting useful metadata such as `PixelSpacing` from DICOM files was problematic. The solution was to focus on only the necessary fields that pertain to diagnosis (e.g., BI-RADS score, lesion type).
   
### 2. **Merging Excel Data with DICOM Images**
   - The INbreast dataset contains an Excel sheet with clinical metadata and a separate folder of images. The challenge was to map each image to its corresponding data point in the Excel file for preprocessing and model training.
   
### 3. **Limited Dataset Size**
   - With only 411 images, the dataset size is a limiting factor for training deep learning models. Techniques like data augmentation (rotation, flipping) were considered to artificially increase the dataset size.

### 4. **Handling Missing Metadata**
   - Not all DICOM images had complete metadata (e.g., `PatientName`, `PixelSpacing`). Missing data was either imputed with median values or left out of the feature set when appropriate.


## References
-**INbreast Dataset**: INbreast Dataset Details <br>
-**BI-RADS Guidelines**: BI-RADS Classification


