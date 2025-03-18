import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import os

# Define dataset paths
dataset_path = "/Users/pierreemmanuelgerard/Desktop/CODESPACE/FACIAL_EMOTION_RECOGNITION/dataset/"

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Check if dataset exists
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Dataset folders not found. Please check dataset_path.")

# Data Augmentation & Normalization
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

# Define CNN Model
model = Sequential([
    Input(shape=(48, 48, 3)),  # Define input shape explicitly
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,  # Adjust epochs as needed
    verbose=1
)

# Save trained model
model.save("emotion_recognition_model.h5")
print("Model training complete and saved as 'emotion_recognition_model.h5'")
# Evaluate model performance on test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"🔥 Test Accuracy: {test_acc:.2f}")