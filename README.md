# Alzheimer-Classification
Alzheimer classification using CNN model with Pytorch

**Notice**: This is a repository of a project that I worked with my teammates in subject DAP391m at FPT University, I re-up this project because the original project was set "private" by my teacher. Some of the models were trained by my teammate on their own files, therefore, the output cells of those models will be empty in this repository.

## Requirement
- Python >= 3.10
- Jupiter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- torch == 2.5.1, torchvision == 0.20.1, torchinfo == 1.8.0
- shap, lime
- opencv-python
- pillow
- joblib
- Chatterbot
- streamlit

## Alzheimer's Disease Stages Classification Using CNN Models
### Dataset
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

### Data Preprocessing
- Resize: 256x256
- Center crop: 224x224
- Transform image to tensor
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### CNN Models Training
Each model is trained separately. Before training a new model, define new 'device' and new 'model' to avoid errors.  
- ResNet50
- DenseNet121
- MobileNetV2
- MobileNetV3 small
- MobileNetV3 large
- EfficientNetB0

### Evaluation and Comparison
- Train and valid accuracy plot
- Train and valid loss plot
- Confusion matrix
- Classification report: accuracy, precision, recall, f1-score

### Explainable AI
Use Grad-CAM heatmaps and LIME visualization to understand the reasoning behind the model’s decisions.

## Alzheimer's Disease Diagnosis Using ML Models
### Dataset
Download Alzheimer dataset from kaggle:

https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

The dataset contains extensive health information for 2,149 patients. It includes demographic details, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, symptoms, and a diagnosis of Alzheimer's disease.

File type: .csv

### Data Preprocessing
- Check for missing values, outliers, and inconsistencies 
- Remove irrelevant columns

### Exploratory Data Analysis (EDA)
Analyze distribution patterns, identify the relationship between risk factors and Alzheimer's disease diagnosis.

### Model Selection
As our dataset is slightly unbalance (65% Alzheimer's and 35% Not Alzheimer's), we decided to use:
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

### Training
The dataset was split into two subsets: 80\% for training and 20\% for testing. The training process employed **cross-validation** to enhance model generalization and prevent overfitting. 

**GridSearchCV** with five-fold cross-validation was applied to the training set to systematically explore different hyperparameter combinations and identify the optimal configuration for each model.

Once the best parameters were determined for each algorithm, the tuned models were used to make predictions on the held-out test set, providing the final evaluation metrics to assess their effectiveness in diagnosing Alzheimer's disease.

### Evaluation and Comparison
- Test accuarcy plot
- AUC score plot
- ROC curve
- Confusion matrix
- Classification report: accuracy, precision, recall, f1-score

### Explainable AI
Use SHAP for global variable explanation and LIME for local variable explanation to understand the reasoning behind the model’s decisions.

## Streamlit Interface:
### Chatbot
A simple chatbot that can provide some information about Alzheirmer's disease.

### Alzheimer's Diagnosis
Allows users to enter data and return diagosis result with confidence score, displays the SHAP and LIME explanation.

### Alzheimer's Stages Classification
Allows users to upload an image and return classifcation result with confidence score, displays the Grad-CAM visualization and LIME visualization.