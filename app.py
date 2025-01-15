import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import os
import streamlit as st

# Streamlit header
st.header('Alzheimer Classification CNN Model')

# Load a test image (replace 'path_to_image' with the actual path)
test_image = Image.open('path_to_image')

# Labels for the model's output
labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load('.\\MobileNetV2_pretrained.pth', map_location=device,weights_only=False)

def classify_image(model, image, class_names):
    """
    Classify a single image using a trained model.

    Parameters:
        model: Trained PyTorch model.
        image: Input image in PIL.Image format.
        class_names: List of class names.

    Returns:
        Predicted class label and confidence.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Define transforms to match training preprocessing
    transform = Compose([
        ToTensor(),  # Convert to PyTorch tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (adjust to your dataset)
    ])

    # Transform the input image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # Print the result
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")

    return class_names[predicted_class], confidence

# File uploader in Streamlit
uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    # Ensure the 'upload' directory exists
    os.makedirs('upload', exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join('upload', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, width=200)

    # Perform classification and display the result
    predicted_label, confidence = classify_image(model, test_image, class_names=labels)
    st.write(f"The Image belongs to {predicted_label} with a score of {confidence * 100:.2f}%")