import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset paths
dataset_path = "/Users/pierreemmanuelgerard/Desktop/CODESPACE/FACIAL_EMOTION_RECOGNITION/dataset/"

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for testing

# Load training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode="categorical"
)

# Load test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode="categorical"
)

# Print class indices
print("Class labels (emotion categories):", train_generator.class_indices)