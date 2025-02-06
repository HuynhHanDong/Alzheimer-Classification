import torch

def classify_image(model, image, class_names, transform, device):
    """
    Classify a single image using a trained model.

    Parameters:
        model: Trained PyTorch model.
        image: Input image in PIL.Image format.
        class_names: List of class names.
        tranform: define transform methods to the data
        device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Returns:
        Predicted class label and confidence.
    """
    model.to(device) 
    model.eval()

    # Transform the input image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return class_names[predicted_class], confidence