# Import libraries

import streamlit as st
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import os
import pandas as pd
import numpy as np
from my_utils import SaveFeatures, GradCam, overlay_heatmap, process_image_for_gradcam, get_target_layer, generate_gradcam_visualization, display_gradcam_results, load_model, get_transform, predict_image

# Layout and web app title
st.title("A Breast Cancer Ultrasound Image Classifier developed by finetuning pretrained models")
st.write("Select a finetuned model, upload a breast ultrasound image of a benign or malignant tumour, select its actual class, and watch the model output its prediction.")
st.write("You can select from a number of modes depending on your use case.")
st.write("All models were trained, validated and tested on 4244 images from 7 different datasets on an NVIDIA A100 GPU.")

st.image("./training_images.png", caption = "A sample of a single batch of images used in training.", use_container_width=True)

st.image("./model_accuracy.png", caption = "Model Accuracy Vs Model Size as a guide for model selection.", use_container_width=True)

# Choose app mode
st.sidebar.title("Choose Mode")

st.sidebar.write("Singe Model - Predicts the image class using a single selected model.")
st.sidebar.write("Model Comparison - Compares predictions from multiple selected models on a single image.")
st.sidebar.write("Model Ensemble - Combines predictions from multiple selected models using majority voting.")

mode = st.sidebar.selectbox("Select mode", [
    "Single Model", 
    "Model Comparison",
    "Model Ensemble"
])

# Model options and paths
models_dir = 'models'
model_options = {
    "EfficientNet_V2_m": os.path.join(models_dir, "EfficientNet_model.pth"),
    "ResNet50": os.path.join(models_dir, "Resnet50_model.pth"),
    "GoogleNet": os.path.join(models_dir, "GoogleNet_model.pth"),
    "Swin_v2_b": os.path.join(models_dir, "swin_v2_model.pth"),
    "Inception_V3": os.path.join(models_dir, "Inception_v3_model.pth"),
    "ViT_B_16": os.path.join(models_dir, "ViT_B_16_model.pth"),
}

# Model sizes
model_sizes_df = pd.DataFrame({
    "Model": ["EfficientNet_V2_m", "ResNet50", "GoogleNet", "Swin_v2_b", "Inception_V3", 'ViT_B_16'],
    "Size (MB)": [727.67, 376.82, 118.07, 760.94, 325.49, 333.89],
    "Number of Parameters (Million)": [53.6, 23.5, 6.1, 58.9, 25.1, 85.8],
    "Accuracy": [0.8165, 0.8075, 0.7835, 0.8165, 0.7985, 0.8045],
    "Precision": [0.8273, 0.8082, 0.7838, 0.8198, 0.8042, 0.8072],
    "Recall": [0.8165, 0.8075, 0.7835, 0.8165, 0.7985, 0.8045],
    "F1_Score": [0.8190, 0.8078, 0.7836, 0.8177, 0.8003, 0.8032],
})

st.dataframe(model_sizes_df, use_container_width=True)
st.write("Above is a chart and a table describing the model sizes and their performance metrics on the test dataset following finetuning and training. Use this a guide in selecting your model(s).")
st.write("***Please note that GRAD-CAM results are not available for the transformer based models (ViT_B_16, Swin_v2_b)***")


# Class names
class_names = ['Benign', 'Malignant']

# Mode: Single Image Prediction, Single Model
if mode == "Single Model":
    # Model selector
    model_name = st.sidebar.selectbox("Select a model", list(model_options.keys()))
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

elif mode == "Model Comparison":
    # Multi-select for models
    selected_models = st.sidebar.multiselect("Select models to compare", list(model_options.keys()), default=list(model_options.keys())[:2])
    
    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=False)
    
    # Actual class selection
    actual_class = st.selectbox("Select the actual class", ['Benign', 'Malignant', 'Unknown'])
    
    if uploaded_image and selected_models:
        # Display the uploaded image
        st.image(uploaded_image, caption=uploaded_image.name, width=300)
        
        # Load and process the image
        image = Image.open(uploaded_image)
        
        st.header("Model Comparison Results")
        
        # Create columns for side-by-side comparison
        # cols = st.columns(len(selected_models))
        
        results = [] 

        for model_name in selected_models:

            st.write("---------------------")
            st.subheader(f"**{model_name}**")
            
            model_path = model_options[model_name]
            model = load_model(model_path)
            transform = get_transform(model_name)
                
            predicted_class_idx, confidence, probabilities = predict_image(model, transform, image)
            predicted_label = class_names[predicted_class_idx]
             
            st.write(f"Prediction: {class_names[predicted_class_idx]}")
            st.write(f"Confidence: {confidence:.2f}%")
                
            # Comparison with actual class
            correctness = None
            if actual_class != 'Unknown':
                actual_class_idx = class_names.index(actual_class)
                correctness = predicted_class_idx == actual_class_idx
                if correctness:
                    st.success("✅ Correct")
                else:
                    st.error("❌ Incorrect")

            results.append({
                "Model": model_name,
                "Prediction": predicted_label,
                "Confidence (%)": round(confidence, 2),
                "Correct": correctness if correctness is not None else "Unknown"
            })
                
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
            
        df_results = pd.DataFrame(results)
        st.subheader("Model Comparison Results")
        st.dataframe(df_results)

# Mode: Single Image Prediction, Model Ensemble
elif mode == "Model Ensemble":
    # Multi-select for models
    selected_models = st.sidebar.multiselect("Select models for ensemble", list(model_options.keys()), default=list(model_options.keys())[:2])
    st.write("***Please note that the ensemble prediction is based on majority voting among the selected models.***")
    st.write("***GRAD-CAM results to show model attention are not available in this mode.***")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], accept_multiple_files=False)
    
    # Actual class selection
    actual_class = st.selectbox("Select the actual class", ['Benign', 'Malignant', 'Unknown'])
    
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
        st.write(f"**Average confidence of the predicted class from majority voting:** {average_confidence:.2f}%")

        # Highlight ensemble result
        if average_confidence > 80:
            st.success("Average ensemble confidence is above 80%")
        else:
            st.warning("Average ensemble condifdence is below 80%. Consider reviewing individual model predictions.")

        # Compare with actual class
        if actual_class != 'Unknown':
            if final_prediction_idx == actual_class_idx:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
