import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook the gradients of the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_idx):
        # Get the gradients and activation
        gradients = self.gradients.cpu().data.numpy()
        activation = self.activation.cpu().data.numpy()

        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(2, 3))  # Shape: [num_channels]

        # Weighted sum of the activation maps
        grad_cam = np.zeros(activation.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            grad_cam += w * activation[0, i]

        # Apply ReLU to the weighted activation maps
        grad_cam = np.maximum(grad_cam, 0)

        # Normalize Grad-CAM to range [0, 1]
        grad_cam -= np.min(grad_cam)
        grad_cam /= np.max(grad_cam)

        return grad_cam
    
def compute_gradCAM(model, image, transform, device):
    '''
    Parameters:
        model: Trained PyTorch model.
        image: Input image in PIL.Image format.
        tranform: define transform methods to the data
        device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    model.to(device) 

    # Transform the input image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Select the target layer for Grad-CAM
    target_layer = model.features[-1]  # The last convolutional layer
    grad_cam = GradCAM(model, target_layer)

    # Perform a forward pass
    output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    # Compute Grad-CAM
    model.zero_grad()
    output[0, class_idx].backward()
    cam = grad_cam.generate(class_idx)
    return cam

def apply_colormap_on_image(image, mask, alpha=0.5):
    """Overlay a heatmap on an image.

    Parameters: NDArray image
        image: Original image as a numpy array (H, W, 3).
        mask: Grad-CAM heatmap as a numpy array (H, W), normalized to [0, 1].
        alpha: Opacity of the heatmap overlay
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = np.float32(heatmap) * alpha + np.float32(image) * (1 - alpha)
    return np.uint8(overlayed_image)