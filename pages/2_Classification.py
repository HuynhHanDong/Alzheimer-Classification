import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import streamlit as st
import GradCAM 
import Classifier
from lime import lime_image
from skimage.segmentation import mark_boundaries

st.set_page_config(page_title='Classification')

# Streamlit header
st.header('Alzheimer Classification CNN Model')

# Labels for the model's output
labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load('.\\saved_models\\EfficienNetB0.pth', map_location=device,weights_only=False)

# Define transforms to match training preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (adjust to your dataset)
])

def classifier_fn(images):
    model.to(device) 
    model.eval()

    # Convert NumPy array (N, H, W, C) → PyTorch Tensor (N, C, H, W)
    images = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
    
    # Normalize images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformed_images = torch.stack([normalize(img) for img in images])

    with torch.no_grad():
        output = model(transformed_images)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    return probabilities.cpu().numpy()

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
    gradcam_heatmap = GradCAM.compute_gradCAM(model, image, transform, device)

    # Convert the Grad-CAM mask to the original image size
    gradcam_heatmap = cv2.resize(gradcam_heatmap, (image.width, image.height))

    # Overlay the Grad-CAM heatmap on the image
    image_np = np.array(image)
    overlayed_image = GradCAM.apply_colormap_on_image(image_np, gradcam_heatmap)

    # Display the results
    st.image(overlayed_image, caption="Grad-CAM Visualization", width=224)

    st.title("LIME Visualization")

    # Create a LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Define transforms for LIME
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    # Explain the model's prediction for the first image
    explanation = explainer.explain_instance(
        image=np.array(transf(image)),  # image as numpy array
        classifier_fn=classifier_fn, # classification function
        top_labels=4, # Number of top labels to explain
        hide_color=0, # Color to replace hidden parts, default is 0 (black)
        batch_size=1,
        num_samples=1000 # Number of perturbed samples
    )

    # Get explanation for the top predicted label
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, # Explain the top predicted class
        positive_only=True, # Highlight only positive regions
        num_features=10, # Show top 10 important regions
        hide_rest=False # Keep the rest of the image visible
    )

    # Define the highlight color
    highlight_color = [255, 255, 0]  # (yellow)
    # Define transparency level (0 = fully transparent, 1 = fully opaque)
    alpha = 0.5  # Adjust transparency level

    # Convert image to float for blending
    colored_image = temp.astype(float)

    # Apply color to positive mask areas
    for i in range(3):  # Iterate over R, G, B channels
        colored_image[:, :, i] = np.where(mask == 1, (1 - alpha) * temp[:, :, i] + alpha * highlight_color[i], temp[:, :, i])

    # Visualize the explanation
    st.image(mark_boundaries(temp / 255, mask), caption="LIME Visualization", width=224)
    st.image(colored_image / 255, caption="LIME Visualization (Highlighted)", width=224)
