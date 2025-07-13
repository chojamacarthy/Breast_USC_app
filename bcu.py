# Import libraries

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
from utils import SaveFeatures, GradCam, overlay_heatmap, process_image_for_gradcam, get_target_layer, generate_gradcam_visualization, display_gradcam_results, load_model, get_transform, predict_image

# Layout and web app title
st.title("A Breast Cancer Ultrasound Image Classifier developed by finetuning pretrained models")
st.write("Select a finetuned model, upload a breast ultrasound image of a benign or malignant tumour, select its actual class, and watch the model output its prediction.")
st.write("You can select from a number of modes depending on your use case.")
st.write("All models were trained on 4244 images from 7 different datasets on an NVIDIA A100 GPU.")

st.image("./training_images.png", caption = "A sample of a single batch of images used in training.", use_container_width=True)

# Choose app mode
st.sidebar.title("Choose App Mode")
mode = st.sidebar.selectbox("Select mode", [
    "Single Image Prediction, Single Model", 
    "Single Image Prediction, Model Comparison",
    "Single Image Prediction, Model Ensemble"
])

# Model options and paths
models_dir = 'models'

model_options = {
    "EfficientNet_V2_m": os.path.join(models_dir, "EfficientNet_model.pth"),
    "ResNet50": os.path.join(models_dir, "Resnet50_model.pth"),
    "GoogleNet": os.path.join(models_dir, "GoogleNet_model.pth"),
    "Swin_v2_b": os.path.join(models_dir, "swin_v2_model.pth"),
    "Inception_V3": os.path.join(models_dir, "Inception_v3_model.pth")
}

# Model sizes
st.write("Below is a table describing the model sizes and their perfromance metrics on the test dataset following finetuning and training. Use this a guide in selecting your model(s).")
model_sizes_df = pd.DataFrame({
    "Model": ["EfficientNet_V2_m", "ResNet50", "GoogleNet", "Swin_v2_b", "Inception_V3"],
    "Size (MB)": [50, 100, 80, 120, 200],
    "Parameters": [54_000_000, 25_000_000, 10_000_000, 30_000_000, 23_000_000],
    "Accuracy (%)": [85.0, 87.5, 84.0, 88.0, 90.0],
    "F1_Score": [0.84, 0.86, 0.82, 0.87, 0.89],
})

st.dataframe(model_sizes_df, use_container_width=True)

# Class names
class_names = ['Benign', 'Malignant']

# Mode: Single Image Prediction, Single Model
if mode == "Single Image Prediction, Single Model":
    # Model selector
    model_name = st.selectbox("Select a model", list(model_options.keys()))
    model_path = model_options[model_name]
    model = load_model(model_path)
    transform = get_transform(model_name)

    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=False)
    
    # Actual class selection
    actual_class = st.selectbox("Select the actual class", ['Benign', 'Malignant', 'Unknown'])

    if uploaded_image:
    
        # Display the uploaded image
        st.image(uploaded_image, caption=uploaded_image.name, width=300)
        
        # Load and process the image
        image = Image.open(uploaded_image)
        predicted_class_idx, confidence, probabilities = predict_image(model, transform, image)
        
        # Display prediction results
        st.write(f"**Prediction:** {class_names[predicted_class_idx]}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Comparison with actual class
        if actual_class != 'Unknown':
            actual_class_idx = class_names.index(actual_class)
            if predicted_class_idx == actual_class_idx:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
                st.write(f"**Expected:** {actual_class}")
                st.write(f"**Predicted:** {class_names[predicted_class_idx]}")

        original_img, overlayed_img, heatmap = generate_gradcam_visualization(
            model, 
            model_name,
            transform(image).unsqueeze(0) if image.mode == 'RGB' else transform(image.convert('RGB')).unsqueeze(0),
            image,
            class_idx=None
            )
                
        display_gradcam_results(
            original_img = original_img,
            overlayed_img = overlayed_img,
            )

# Mode: Single Image Prediction, Model Comparison

elif mode == "Single Image Prediction, Model Comparison":
    # Multi-select for models
    selected_models = st.multiselect("Select models to compare", list(model_options.keys()), default=list(model_options.keys())[:2])
    
    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=False)
    
    # Actual class selection
    actual_class = st.selectbox("Select the actual class", ['Benign', 'Malignant'])
    
    if uploaded_image and selected_models:
        # Display the uploaded image
        st.image(uploaded_image, caption=uploaded_image.name, width=300)
        
        # Load and process the image
        image = Image.open(uploaded_image)
        actual_class_idx = class_names.index(actual_class)
        
        st.header("Model Comparison Results")
        
        # Create columns for side-by-side comparison
        # cols = st.columns(len(selected_models))
        
        for model_name in selected_models:
            st.write("---------------------")
            st.subheader(f"**{model_name}**")
            
            model_path = model_options[model_name]
            model = load_model(model_path)
            transform = get_transform(model_name)
                
            predicted_class_idx, confidence, probabilities = predict_image(model, transform, image)
             
            st.write(f"Prediction: {class_names[predicted_class_idx]}")
            st.write(f"Confidence: {confidence:.2f}%")
                
            # Comparison with actual class
            if predicted_class_idx == actual_class_idx:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
                
            original_img, overlayed_img, heatmap = generate_gradcam_visualization(
                model, 
                model_name,
                transform(image).unsqueeze(0) if image.mode == 'RGB' else transform(image.convert('RGB')).unsqueeze(0),
                image,
                class_idx=None
                )
                
            display_gradcam_results(
                original_img = original_img,
                overlayed_img = overlayed_img,
                )

# Mode: Single Image Prediction, Model Ensemble
elif mode == "Single Image Prediction, Model Ensemble":
    # Multi-select for models
    selected_models = st.multiselect("Select models for ensemble", list(model_options.keys()), default=list(model_options.keys())[:2])
    st.write("***Please note that the ensemble prediction is based on majority voting among the selected models.***")
    st.write("***GRAD-CAM results are not available in this mode.***")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=False)
    
    # Actual class selection
    actual_class = st.selectbox("Select the actual class", ['Benign', 'Malignant'])
    
    if uploaded_image and selected_models:
        # Display the uploaded image
        st.image(uploaded_image, caption=uploaded_image.name, width=300)
        
        # Load and process the image
        image = Image.open(uploaded_image)
        actual_class_idx = class_names.index(actual_class)
        
        st.write("##### Ensemble Prediction")
        
        # Initialize variables for ensemble prediction
        class_votes = {0: 0, 1: 0}
        class_confidences = {0: [], 1: []}
        
        for model_name in selected_models:
            model_path = model_options[model_name]
            model = load_model(model_path)
            transform = get_transform(model_name)
            
            predicted_class_idx, confidence, _ = predict_image(model, transform, image)
            class_votes[predicted_class_idx] += 1
            class_confidences[predicted_class_idx].append(confidence)

        # Decide final prediction based on majority vote
        final_prediction_idx = max(class_votes, key=class_votes.get)
        final_prediction = class_names[final_prediction_idx]

        # Calculate average confidence of the predicted class
        confidences = class_confidences[final_prediction_idx]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Display ensemble results
        named_votes = {class_names[k]: v for k, v in class_votes.items()}
        votes_df = pd.DataFrame.from_dict(named_votes, orient='index', columns=['Votes'])
        votes_df.index.name = 'Class'
        st.write(votes_df)
        st.write(f"**Average Confidence of the predicted class:** {average_confidence:.2f}%")

        # Highlight ensemble result
        if average_confidence > 80:
            st.success("Averege ensemble confidence is above 80%")
        else:
            st.warning("Averege ensemble condifdence is below 80%. Consider reveiwing individual model predictions.")

        # Compare with actual class
        if final_prediction_idx == actual_class_idx:
            st.success("✅ Correct")
        else:
            st.error("❌ Incorrect")
