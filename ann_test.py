import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

# Define class names for each model
class_names_dict = {
    "potato": ['Potato___early_blight', 'Potato___healthy', 'Potato___late_blight'],
    "tomato": ['Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight'],
    "grape": ['Grape___black_measles', 'Grape___black_rot', 'Grape___healthy']
}

# Load the models
models = {
    "potato": tf.keras.models.load_model("models/ANN_MLP_POTATO_E100.h5"),
    "tomato": tf.keras.models.load_model("models/ANN_MLP_TOMATO_E100.h5"),
    "grape": tf.keras.models.load_model("models/ANN_MLP_GRAPE_E100.h5")
}

# Initially set the model to Potato
selected_model_name = "potato"
model = models[selected_model_name]
class_names = class_names_dict[selected_model_name]

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
        # Flatten the image to match the input shape expected by the model
        img = img.reshape(1, -1)
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
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "prediction_time": prediction_time
    }

# Main function to test the model with the selected image
def test_model(leaf_name, image_data):
    try:
       
        # Load the model and class names
        model = models[leaf_name]
        class_names = class_names_dict[leaf_name]
        
        # Preprocess the image
        image_array = preprocess_image(image_data)
        
        # Make a prediction
        result = predict_image(model, image_array, class_names)
        return result
    except Exception as e:
        return {"error": str(e)}