import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Parameters
img_height, img_width = 64, 64  # Same size used during training

# Models and their corresponding class names
models = {
    "potato": {
        "model": load_model('models/GRU_RNN_POTATO_E100.h5'),
        "classes": ['Potato___early_blight', 'Potato___healthy', 'Potato___late_blight']
    },
    "tomato": {
        "model": load_model('models/GRU_RNN_TOMATO_E100.h5'),
        "classes": ['Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight']
    },
    "grape": {
        "model": load_model('models/GRU_RNN_GRAPE_E100.h5'),
        "classes": ['Grape___black_measles', 'Grape___black_rot', 'Grape___healthy']
    }
}

# Default selected model
selected_model_name = "potato"
selected_model = models[selected_model_name]["model"]
selected_classes = models[selected_model_name]["classes"]

# Preprocess the image (binary data)
def preprocess_image(image_data, img_size=(64, 64)):
    """
    Preprocess the selected image for model prediction.
    - `image_data`: Binary image data.
    - `img_size`: Target size for resizing the image.
    Returns the preprocessed image.
    """
    try:
        # Convert binary data to a NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image data.")
        # Resize the image
        img = cv2.resize(img, img_size)
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")

# Make predictions
def predict_image(model, image_array, class_names):
    """
    Predict the class of the given image using the trained model.
    """
    start_time = time.time()
    predictions = model.predict(image_array)
    prediction_time = time.time() - start_time
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index] * 100
    predicted_class = class_names[predicted_class_index]
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "prediction_time": prediction_time
    }

# Main function to test the model with the selected image
def test_model(leaf_name, image_data):
    try:
        
        
        # Load the model and class names
        model = models[leaf_name]["model"]
        class_names = models[leaf_name]["classes"]
        
        # Preprocess the image
        image_array = preprocess_image(image_data)
        
        # Make a prediction
        result = predict_image(model, image_array, class_names)
        return result
    except Exception as e:
        return {"error": str(e)}