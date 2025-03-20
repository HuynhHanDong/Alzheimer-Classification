# Alzheimer-Classification
Alzheimer classification using CNN model with Pytorch

**Notice**: This is a repository of a project that I worked with my teammates in subject DAP391m at FPT University, I re-up this project because the original project was set "private" by my teacher. Some of the models were trained by my teammate on their own files, therefore, the output cells of those models will be empty in this repository.

# Alzheimer's disease classification using CNN models
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

### Evaluation and Visualization
- Train and valid accuracy plot
- Train and valid loss plot
- Confusion matrix
- Classification report: accuracy, precision, recall, f1-score

### Explainable AI
Use Grad-CAM heatmaps and LIME visualization to understand the reasoning behind the model’s decisions.

### Streamlit Interface:
Allows users to upload an image, shows classifcation result with confidence score, displays the Grad-CAM visualization and LIME visualization.

# Chatbot
(coming soon)