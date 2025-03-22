import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random
import time  # Import time for tracking training time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
import warnings

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

plt.style.use('ggplot')

# Dataset directory
dataDir = 'C:/Users/gnana/OneDrive/Desktop/IMAGE-BASED ANALYSIS FOR IDENTIFICATION OF PLANT LEAF PATHOLOGIES USING NEURAL NETWORKS/Dataset/Grape'

# Automatically fetch classes
classes = sorted([d for d in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, d))])
print(f"Detected Classes: {classes}")

imgPaths = []
labels = []

# Collect image paths and labels
for className in classes:
    classPath = os.path.join(dataDir, className)
    for img in os.listdir(classPath):
        imgPath = os.path.join(classPath, img)
        imgPaths.append(imgPath)
        labels.append(className)

# Create a dataframe
df = pd.DataFrame({'imgPath': imgPaths, 'label': labels})
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Map string labels to integers
label_mapping = {label: idx for idx, label in enumerate(classes)}
print(f"Label Mapping: {label_mapping}")

df['label'] = df['label'].replace(label_mapping).astype(int)

# Resize images and normalize
IMG_SIZE = (150, 150)
imgs = []
for imgPath in tqdm(df['imgPath'], total=len(df)):
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    imgs.append(img)

images = np.array(imgs) / 255.0  # Normalize
labels = np.array(df['label'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, shuffle=True)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Define CNN Model
with tf.device('/GPU:0'):
    kerasModel = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(len(label_mapping), activation='softmax')
    ])

    kerasModel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)  # Modified patience

    # Record start time
    start_time = time.time()

    # Train the Model
    history = kerasModel.fit(
        datagen.flow(X_train, y_train, batch_size=50),
        validation_data=(X_test, y_test),
        epochs=100,  # Train for 100 epochs without early stopping
        callbacks=[reduce_lr],  # Only use reduce_lr
        verbose=1
    )

    # Record end time
    end_time = time.time()
    training_time = end_time - start_time

    # Save the model
    kerasModel.save('CNN_GRAPE_E100.h5')
    print("Model saved as 'CNN_GRAPE_E100.h5'")

    y_pred = kerasModel.predict(X_test)

# Plot Accuracy and Loss
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Model Accuracy\nTraining Time: {training_time:.2f} secs\nFinal Train Accuracy: {history.history["accuracy"][-1]:.4f}, Final Validation Accuracy: {history.history["val_accuracy"][-1]:.4f}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss\nFinal Train Loss: {history.history["loss"][-1]:.4f}, Final Validation Loss: {history.history["val_loss"][-1]:.4f}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Confusion Matrix and Classification Report
new_y_pred = [np.argmax(x) for x in y_pred]
CM = confusion_matrix(y_test, new_y_pred)
sns.heatmap(CM, center=True, cmap='summer', annot=True, fmt='d')
plt.show()

ClassificationReport = classification_report(y_test, new_y_pred)
print('Classification Report:\n', ClassificationReport)
