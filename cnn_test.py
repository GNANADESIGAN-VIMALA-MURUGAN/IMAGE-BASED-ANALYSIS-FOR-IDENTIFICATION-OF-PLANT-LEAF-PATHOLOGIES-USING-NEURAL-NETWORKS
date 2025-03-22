
import os
import sys
import cv2
import numpy as np
import time
from keras.models import load_model

# Define paths for the models
models = {
    "potato": "models/CNN_POTATO_E100.h5",
    "tomato": "models/CNN_TOMATO_E100.h5",
    "grape": "models/CNN_GRAPE_E100.h5",
}

# Label mappings for each model
label_mappings = {
    "potato": {
        0: 'Potato___early_blight',
        1: 'Potato___healthy',
        2: 'Potato___late_blight',
    },
    "tomato": {
        0: 'Tomato___early_blight',
        1: 'Tomato___healthy',
        2: 'Tomato___late_blight',
    },
    "grape": {
        0: 'Grape___black_measles',
        1: 'Grape___black_rot',
        2: 'Grape___healthy',
    },
}

# Function to load the selected model
def load_model_by_name(leaf_name):
    if leaf_name not in models:
        raise ValueError(f"Invalid leaf name: {leaf_name}. Choose from 'potato', 'tomato', or 'grape'.")
    
    model_path = models[leaf_name]
    print(f"Loading {leaf_name.capitalize()} disease prediction model...")
    start_time = time.time()
    model = load_model(model_path)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds.")
    return model, label_mappings[leaf_name]

# Preprocess the user-input image (binary data)
def preprocess_image(image_data, img_size=(150, 150)):
    try:
        # Convert binary data to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode the image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image data.")

        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

# Test the model with user input
def test_model(model, label_mapping, image_data):
    print("Processing image...")

    try:
        # Preprocess the image
        input_image = preprocess_image(image_data)
        
        # Measure prediction time
        start_time = time.time()
        prediction = model.predict(input_image)
        pred_time = time.time() - start_time
        
        # Get predicted label and confidence
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        result = {
            "predicted_class": label_mapping[predicted_class],
            "confidence": confidence,
            "prediction_time": pred_time,
        }
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

# Main function to handle predictions
def test1(leaf_name, image_data):
    try:
        # Load the model
        model, label_mapping = load_model_by_name(leaf_name)
        
        # Test the model
        result = test_model(model, label_mapping, image_data)
        
        return result
    except ValueError as ve:
        return {"error": str(ve)}