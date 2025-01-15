# Alzheimer-Classification
Alzheimer classification using CNN model with Pytorch

## Dataset
Download Alzheimer MRI dataset from kaggle:
https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy

4 classes: Mild Impairment, Moderate Impairment, Non Impairment, Very Mild Impairment

Train: 2560 images for each class
Test:
- Mild Impairment = 179
- Moderate Impairment = 12
- Non Impairment = 640
- Very Mild Impairment = 448

File type: .jpg

## Data Preproccesing
- Resize: 256x256
- Center crop: 224x224
- Transform image to tensor
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Model Training
# MobileNetV2
# 

## Evaluation
- Accuracy
- Loss
- Confusion matrix