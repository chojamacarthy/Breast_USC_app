
import streamlit as st
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
import pandas as pd
import numpy as np
import altair as alt
import cv2
import matplotlib.pyplot as plt

# Load selected model
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# Preporocessing function
def get_transform(model_name):
    if model_name == "Inception_V3":
        return transforms.Compose([
            transforms.Resize(size=(299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

# Predicting function
def predict_image(model, transform, image):
    image = image.convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1).squeeze()
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item() * 100
    
    return predicted_class_idx, confidence, probabilities

# Grad CAM function
class SaveFeatures:
    def __init__(self, module):
        self.module = module
        self.features = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)
        self.hook_grad = module.register_backward_hook(self.hook_grad_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def hook_grad_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # Gradient with respect to the output

    def close(self):
        self.hook.remove()
        self.hook_grad.remove()

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

    def __call__(self, x, class_idx=None):
        conv_output = SaveFeatures(self.target_layer)  # Hook to get feature maps and gradients
        model_output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(model_output)  # Default to the class with the highest score

        one_hot = torch.zeros((1, model_output.size()[-1]), dtype=torch.float32).to(x.device)
        one_hot[0][class_idx] = 1

        self.model.zero_grad()
        model_output.backward(gradient=one_hot, retain_graph=True)  # Backpropagate for the target class

        # Get the gradients and feature maps from the hook
        gradients = conv_output.gradients
        feature_maps = conv_output.features[0]

        # Perform global average pooling to get the weights
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Multiply feature maps by the pooled gradients
        for i in range(len(pooled_gradients)):
            feature_maps[i, :, :] *= pooled_gradients[i]

        # Average along the channel dimension to get the Grad-CAM heatmap
        heatmap = torch.mean(feature_maps, dim=0).cpu().detach().numpy()

        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap /= np.max(heatmap)
        return heatmap
    

# Function to overlay Grad-CAM heatmap on the image
def overlay_heatmap(heatmap, img):

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to uint8 format
    heatmap = np.uint8(255 * heatmap)
    
    # Apply color map (e.g., JET) to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert the original image to uint8 (if it's float32)
    if img.dtype != np.uint8:
        img = np.uint8(255 * img)
    
    # Ensure the original image has 3 channels (if it's grayscale, convert to RGB)
    if len(img.shape) == 2 or img.shape[2] == 1:  # If grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Overlay the heatmap onto the original image
    overlayed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    return overlayed_img

# Load your test image and preprocess
def process_image_for_gradcam(uploaded_image, model_name):
    
    img_array = np.array(uploaded_image)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    if model_name == "Inception_V3":
       target_size = (299, 299)  # Inception V3 requires 299x299 input size
    else:
        target_size = (224, 224) 
    # Resize to original size for overlay
    img_resized = cv2.resize(img_array, target_size)
    
    return img_resized

def get_target_layer(model, model_name):
    if model_name == "EfficientNet_V2_m":
        return model.features[8]
    elif model_name == "ResNet50":
        return model.layer4[2].conv3
    elif model_name == "GoogleNet":
        return model.inception5b.branch4[1].conv
    elif model_name == "Swin_v2_b":
        return model.features[7][1]
    elif model_name == "Inception_V3":
        return model.Mixed_7c.branch3x3dbl_3b.conv
    elif model_name == "ViT_B_16":
        return model.blocks[-1].norm1
    else:
        raise ValueError(f"Could not find appropriate target layer for {model_name}")

def generate_gradcam_visualization(model, model_name, image_tensor, pil_image, class_idx=None):
    """
    Generate Grad-CAM visualization for the given model and image.
    Returns the original image and overlayed Grad-CAM image.
    """
    try:
        # Get the target layer for the model
        target_layer = get_target_layer(model, model_name)
        
        # Initialize Grad-CAM
        grad_cam = GradCam(model, target_layer)
        
        # Process the original image for overlay
        original_img = process_image_for_gradcam(pil_image, model_name)
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam(image_tensor, class_idx)
        
        # Overlay heatmap on original image
        overlayed_img = overlay_heatmap(heatmap, original_img)
        
        return original_img, overlayed_img, heatmap
    
    except Exception as e:
        st.error(f"Error generating Grad-CAM visualization: {str(e)}")
        return None, None, None

def display_gradcam_results(original_img, overlayed_img):
    """
    Display Grad-CAM results in Streamlit with proper formatting.
    """
    if original_img is not None and overlayed_img is not None:
        st.markdown("##### Grad-CAM Visualisation")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Image")
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.write("Overlayed heatmap")
            st.image(overlayed_img, use_container_width=True)
        
        # Add explanation
        st.info("""
        **Grad-CAM interpretation:**
        - Red/Yellow areas: Regions that strongly influenced the prediction
        - Blue areas: Regions that had little influence
        - The heatmap shows where the model focused on to make its prediction.
        """)