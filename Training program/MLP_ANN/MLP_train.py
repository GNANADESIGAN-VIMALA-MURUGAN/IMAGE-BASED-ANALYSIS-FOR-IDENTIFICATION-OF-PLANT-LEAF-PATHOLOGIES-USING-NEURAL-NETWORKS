import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback  # For progress bar

# Enhanced data loading with augmentation
def load_dataset(data_dir, img_size=(64, 64), batch_size=32, validation_split=0.2):
    # Create augmentation layers
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.GaussianNoise(0.01)
    ])

    # Load datasets and automatically detect class names
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int'
    )

    # Get automatically detected class names
    class_names = train_data.class_names
    print(f"Detected Classes: {class_names}")

    # Apply augmentation only to training data
    train_data = train_data.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

    # Optimize dataset performance
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, val_data, class_names

# Load dataset and automatically detect classes
data_dir = "C:/Users/gnana/OneDrive/Desktop/IMAGE-BASED ANALYSIS FOR IDENTIFICATION OF PLANT LEAF PATHOLOGIES USING NEURAL NETWORKS/Dataset/Grape"
img_size = (64, 64)
batch_size = 64
train_data, val_data, class_names = load_dataset(data_dir, img_size, batch_size)
num_classes = len(class_names)

# Save class names for later use in prediction
np.save("class_names.npy", class_names)

# Model creation function
def create_model():
    model = Sequential([
        Dense(2048, kernel_regularizer=l2(0.001), 
              input_shape=(img_size[0]*img_size[1]*3,)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        Dense(1024, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Data preprocessing
def preprocess_data(dataset):
    images, labels = [], []
    for img_batch, label_batch in dataset:
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images.reshape(images.shape[0], -1), labels

X_train, y_train = preprocess_data(train_data)
X_val, y_val = preprocess_data(val_data)

# Callbacks with progress bar
callbacks = [
    ModelCheckpoint('grape_best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    TqdmCallback(verbose=1)  # Add progress bar
]

# Training with progress bar
start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=0  # Set to 0 to avoid duplicate progress bars
)
training_time = time.time() - start_time

# Save final model
model.save('ANN_MLP_GRAPE_E100.h5')

# Evaluation
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"\nTraining Complete (Time: {training_time:.2f}s)")
print(f"Detected Classes: {class_names}")
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")


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