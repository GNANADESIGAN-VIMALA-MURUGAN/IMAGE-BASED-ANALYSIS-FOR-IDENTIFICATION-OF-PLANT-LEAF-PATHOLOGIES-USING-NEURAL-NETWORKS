# import time
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tkinter import Tk, filedialog

# # Load the model (Ensure to use the correct path to your saved model)
# model = tf.keras.models.load_model("final_model.h5")

# # Assuming class names were printed during training or defined manually
# class_names = ['Grape___black_measles', 'Grape___black_rot', 'Grape___healthy']  # Replace with actual class names

# # Preprocess the image
# def preprocess_image(image_path, img_size=(64, 64)):
#     """
#     Preprocess the selected image for model prediction.
#     - `image_path`: Path to the image file.
#     - `img_size`: Target size for resizing the image.
#     Returns the preprocessed image and the original image.
#     """
#     original_image = load_img(image_path)
#     resized_image = load_img(image_path, target_size=img_size)
#     image_array = img_to_array(resized_image) / 255.0  # Normalize to [0, 1]
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#     # Flatten the image to match the input shape expected by the model
#     image_array = image_array.reshape(image_array.shape[0], -1)

#     return image_array, original_image

# # Make predictions
# def predict_image(model, image_array, class_names):
#     """
#     Predict the class of the given image using the trained model.
#     """
#     predictions = model.predict(image_array)
#     predicted_class = class_names[np.argmax(predictions)]
#     predicted_probability = np.max(predictions)
    
#     return predicted_class, predicted_probability

# # Select an image file
# def select_image():
#     """
#     Opens a file dialog to select an image file.
#     """
#     root = Tk()
#     root.withdraw()  # Hides the root window
#     file_path = filedialog.askopenfilename(
#         title="Select an Image",
#         filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
#     )
#     return file_path

# # Main function to test the model with the selected image
# def main():
#     # Let the user select an image
#     image_path = select_image()
#     if not image_path:
#         print("No image selected. Exiting...")
#         return

#     # Preprocess the image
#     start_time = time.time()
#     image_array, original_image = preprocess_image(image_path)
    
#     # Make a prediction
#     predicted_class, predicted_probability = predict_image(model, image_array, class_names)
#     prediction_time = time.time() - start_time

#     # Print results
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Prediction Probability: {predicted_probability * 100:.2f}%")
#     print(f"Prediction Time: {prediction_time:.2f} seconds")

#     # Optionally show the image (you can skip this part if not needed)
#     original_image.show()

# # Run the main function
# if __name__ == "__main__":
#     main()
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog, Label, Button, messagebox, Canvas, PhotoImage, OptionMenu, StringVar
from PIL import Image, ImageTk

# Define class names for each model
class_names_dict = {
    "Potato": ['Potato___early_blight', 'Potato___healthy', 'Potato___late_blight'],
    "Tomato": ['Tomato___early_blight', 'Tomato___healthy', 'Tomato___late_blight'],
    "Grape": ['Grape___black_measles', 'Grape___black_rot', 'Grape___healthy']
}

# Load the models
models = {
    "Potato": tf.keras.models.load_model("ANN_MLP_POTATO_E100.h5"),
    "Tomato": tf.keras.models.load_model("ANN_MLP_TOMATO_E100.h5"),
    "Grape": tf.keras.models.load_model("ANN_MLP_GRAPE_E100.h5")
}

# Initially set the model to Potato
selected_model_name = "Potato"
model = models[selected_model_name]
class_names = class_names_dict[selected_model_name]

# Preprocess the image
def preprocess_image(image_path, img_size=(64, 64)):
    """
    Preprocess the selected image for model prediction.
    - `image_path`: Path to the image file.
    - `img_size`: Target size for resizing the image.
    Returns the preprocessed image and the original image.
    """
    original_image = load_img(image_path)
    resized_image = load_img(image_path, target_size=img_size)
    image_array = img_to_array(resized_image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Flatten the image to match the input shape expected by the model
    image_array = image_array.reshape(image_array.shape[0], -1)

    return image_array, original_image

# Make predictions
def predict_image(model, image_array, class_names):
    """
    Predict the class of the given image using the trained model.
    """
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    predicted_probability = np.max(predictions)
    
    return predicted_class, predicted_probability

# Select an image file
def select_image():
    """
    Opens a file dialog to select an image file.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    return file_path

# Update model based on user selection
def update_model(event):
    """
    Update the model and class names based on selected model.
    """
    global selected_model_name, model, class_names
    selected_model_name = model_selection.get()
    model = models[selected_model_name]
    class_names = class_names_dict[selected_model_name]

# Main function to test the model with the selected image
def main():
    # Let the user select an image
    image_path = select_image()
    if not image_path:
        messagebox.showinfo("Info", "No image selected. Exiting...")
        return

    # Preprocess the image
    start_time = time.time()
    image_array, original_image = preprocess_image(image_path)
    
    # Make a prediction
    predicted_class, predicted_probability = predict_image(model, image_array, class_names)
    prediction_time = time.time() - start_time

    # Display the image in the GUI
    original_image.thumbnail((300, 300))  # Resize the image to fit in the GUI
    img_tk = ImageTk.PhotoImage(original_image)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference to avoid garbage collection

    # Display the prediction results
    result_text = f"Predicted Class: {predicted_class}\nPrediction Probability: {predicted_probability * 100:.2f}%\nPrediction Time: {prediction_time:.2f} seconds"
    result_label.config(text=result_text)

# Create the main window
root = Tk()
root.title("Image Classification")
root.geometry("500x600")

# Create a dropdown to select the model
model_selection = StringVar(root)
model_selection.set(selected_model_name)  # Set default value
model_menu = OptionMenu(root, model_selection, "Potato", "Tomato", "Grape", command=update_model)
model_menu.pack(pady=10)

# Create a canvas to display the image
image_label = Label(root)
image_label.pack(pady=10)

# Create a label to display the prediction results
result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Create a button to select an image
select_button = Button(root, text="Select Image", command=main)
select_button.pack(pady=10)

# Run the application
root.mainloop()
