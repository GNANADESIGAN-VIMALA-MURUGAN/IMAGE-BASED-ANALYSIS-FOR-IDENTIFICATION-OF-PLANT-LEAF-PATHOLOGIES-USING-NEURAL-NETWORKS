import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk

# Parameters
img_height, img_width = 64, 64  # Same size used during training

# Models and their corresponding class names
models = {
    "Potato": {
        "model": load_model('GRU_RNN_POTATO_E100.h5'),
        "classes": ['Potato___early_blight', 'Potato___healthy', 'Potato___late_blight']
    },
    "Tomato": {
        "model": load_model('GRU_RNN_TOMATO_E100.h5'),
        "classes": ['Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight']
    },
    "Grape": {
        "model": load_model('GRU_RNN_GRAPE_E100.h5'),
        "classes": ['Grape___black_measles', 'Grape___black_rot', 'Grape___healthy']
    }
}

# Default selected model
selected_model_name = "Potato"
selected_model = models[selected_model_name]["model"]
selected_classes = models[selected_model_name]["classes"]

def predict_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = selected_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]
    predicted_class = selected_classes[predicted_class_index]
    return predicted_class, confidence

# GUI Functionality
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))  # Resize for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f"Predicted: {predicted_class}\nConfidence: {confidence * 100:.2f}%", fg="blue")

def change_model(event):
    global selected_model, selected_classes, selected_model_name
    selected_model_name = model_selector.get()
    selected_model = models[selected_model_name]["model"]
    selected_classes = models[selected_model_name]["classes"]
    result_label.config(text="Model changed to " + selected_model_name)

# Initialize GUI
root = tk.Tk()
root.title("Leaf Disease Classification")
root.geometry("400x550")
root.configure(bg="white")

Label(root, text="Select a Model", font=("Arial", 12), bg="white").pack(pady=5)
model_selector = ttk.Combobox(root, values=list(models.keys()), state="readonly")
model_selector.set(selected_model_name)
model_selector.pack()
model_selector.bind("<<ComboboxSelected>>", change_model)

Label(root, text="Select an Image for Classification", font=("Arial", 12), bg="white").pack(pady=10)
Button(root, text="Upload Image", command=open_image, font=("Arial", 10), bg="lightgray").pack()

image_label = Label(root, bg="white")
image_label.pack(pady=10)

result_label = Label(root, text="Prediction will appear here", font=("Arial", 12), bg="white")
result_label.pack(pady=10)

root.mainloop()
