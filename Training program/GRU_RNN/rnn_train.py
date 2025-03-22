import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import time

# Enable memory growth for GPU to utilize all available memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled GPU memory growth: {gpus}")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define parameters
img_height, img_width = 64, 64  # Resize images to 64x64
batch_size = 32
num_classes = 3  # Number of disease classes
epochs = 100
validation_split = 0.2  # 20% of data for validation

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split  # Split dataset into train/val
)

# Load training data from the single dataset folder
train_generator = train_datagen.flow_from_directory(
    'C:/Users/gnana/OneDrive/Desktop/IMAGE-BASED ANALYSIS FOR IDENTIFICATION OF PLANT LEAF PATHOLOGIES USING NEURAL NETWORKS/Dataset/Grape',  # Replace with your dataset path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Automatically uses 80% of the data (1 - validation_split)
)

# Load validation data from the same folder
validation_generator = train_datagen.flow_from_directory(
    'C:/Users/gnana/OneDrive/Desktop/archive/datag',  # Same path as above
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Automatically uses 20% of the data
)

# Build the model
model = models.Sequential()
model.add(layers.Reshape((img_height, img_width * 3), input_shape=(img_height, img_width, 3)))
model.add(layers.GRU(128, return_sequences=True))
model.add(layers.Dropout(0.3))
model.add(layers.GRU(64))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1) if epoch >= 10 else lr

# Train the model
start_time = time.time()
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[LearningRateScheduler(lr_scheduler)]
)

# Save and evaluate the model
training_time = time.time() - start_time
model.save('GRU_RNN_Grape_E100.h5')


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

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
