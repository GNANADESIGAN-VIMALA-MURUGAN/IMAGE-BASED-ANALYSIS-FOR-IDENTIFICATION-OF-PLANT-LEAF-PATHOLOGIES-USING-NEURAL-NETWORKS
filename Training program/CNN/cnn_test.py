import os
import cv2
import numpy as np
import time
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# Define paths for the three models
models = {
    "Potato": "CNN_POTATO_200.h5",
    "Tomato": "Improved_Tomato_Plant_Disease_Model.h5",
    "Grape": "Improved_Grape_Plant_Disease_Model.h5",
}

# Label mappings for each model
label_mappings = {
    "Potato": {
        0: 'Potato___early_blight',
        1: 'Potato___healthy',
        2: 'Potato___late_blight',
    },
    "Tomato": {
        0: 'Tomato___early_blight',
        1: 'Tomato___healthy',
        2: 'Tomato___late_blight',
    },
    "Grape": {
        0: 'Grape___black_measles',
        1: 'Grape___black_rot',
        2: 'Grape___healthy',
    },
}

# Initialize variables
selected_model = None
model = None
current_label_mapping = None

# Function to load the selected model
def load_selected_model(model_name):
    global model, current_label_mapping
    model_path = models[model_name]
    start_time = time.time()
    model = load_model(model_path)
    load_time = time.time() - start_time
    current_label_mapping = label_mappings[model_name]
    print(f"Model '{model_name}' loaded in {load_time:.2f} seconds.")

# Function to preprocess the user-input image
def preprocess_image(image_path, img_size=(150, 150)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to test the model with user input
def test_model(image_path):
    if not model:
        messagebox.showerror("Error", "Please select a model first!")
        return

    print(f"Processing image: {image_path}")
    
    # Preprocess the image
    input_image = preprocess_image(image_path)
    
    # Measure prediction time
    start_time = time.time()
    prediction = model.predict(input_image)
    pred_time = time.time() - start_time
    
    # Get predicted label and confidence
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"Prediction Time: {pred_time:.2f} seconds")
    print(f"Predicted Class: {current_label_mapping[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the results in a popup window
    show_result(image_path, current_label_mapping[predicted_class], confidence, pred_time)

# Function to display the result in a popup window
def show_result(image_path, predicted_label, confidence, pred_time):
    result_window = tk.Toplevel()
    result_window.title("Prediction Result")
    
    # Display the image
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    
    img_label = tk.Label(result_window, image=img_tk)
    img_label.image = img_tk
    img_label.pack(pady=10)
    
    # Display prediction details
    result_text = (
        f"Predicted Class: {predicted_label}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Prediction Time: {pred_time:.2f} seconds"
    )
    result_label = tk.Label(result_window, text=result_text, font=("Arial", 14))
    result_label.pack(pady=10)

# Function to open a file dialog to select an image
def browse_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        test_model(file_path)

# Function to handle model selection
def select_model(event):
    selected_model_name = model_combo.get()
    if selected_model_name:
        load_selected_model(selected_model_name)

# Main application window
root = tk.Tk()
root.title("Plant Disease Prediction")

# Dropdown menu to select the model
model_label = tk.Label(root, text="Select Model:", font=("Arial", 14))
model_label.pack(pady=10)

model_combo = ttk.Combobox(root, values=list(models.keys()), font=("Arial", 14))
model_combo.bind("<<ComboboxSelected>>", select_model)
model_combo.pack(pady=10)

# Create a button to browse and test an image
browse_button = tk.Button(
    root, text="Select Image", font=("Arial", 16),
    command=browse_image, bg="lightblue"
)
browse_button.pack(pady=20)

root.mainloop()
