import torch
from torchvision import transforms
from PIL import Image
import os
import streamlit as st
from GradCAM import *
import Classifier

# Streamlit header
st.header('Alzheimer Classification CNN Model')

# Labels for the model's output
labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load('.\\MobileNetV2_pretrained.pth', map_location=device,weights_only=False)

# Define transforms to match training preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (adjust to your dataset)
])


# File uploader in Streamlit
uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    # Ensure the 'test_images' directory exists
    os.makedirs('test_images', exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join('test_images', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, width=224)

    image = Image.open(file_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Perform classification and display the result
    predicted_label, confidence = Classifier.classify_image(model, image, labels, transform, device)
    st.write(f"The Image belongs to {predicted_label} with a score of {confidence * 100:.2f}%")

    st.title("Grad-CAM Visualization")
    gradcam_heatmap = compute_gradCAM(model, image, transform, device)

    # Convert the Grad-CAM mask to the original image size
    gradcam_heatmap = cv2.resize(gradcam_heatmap, (image.width, image.height))

    # Overlay the Grad-CAM heatmap on the image
    image_np = np.array(image)
    overlayed_image = apply_colormap_on_image(image_np, gradcam_heatmap)

    # Display the results
    st.image(overlayed_image, caption="Grad-CAM Visualization", width=224)