import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_recognition_model.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  # Extract face
        roi = cv2.resize(roi, (48, 48))  # Resize to model input size
        roi = roi / 255.0  # Normalize
        roi = np.expand_dims(roi, axis=0)  # Expand dimensions for prediction

        # Predict emotion
        prediction = model.predict(roi)
        emotion_index = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_index]

        # Display emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Show video frame
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()