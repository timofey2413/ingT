import cv2
import numpy as np
from keras.models import load_model
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the emotion recognition model file exists
if os.path.exists('emotion_model7.h5'):
    # Load the emotion recognition model
    emotion_model = load_model('emotion_model7.h5')
else:
    print("Error: File 'emotion_recognition_model.h5' not found.")
    exit()

# Check if the lip moisture detection model file exists
if os.path.exists('wet_dry_model.h5'):
    # Load the lip moisture detection model
    lip_model = load_model('wet_dry_model.h5')
else:
    print("Error: File 'wet_dry_model.h5' not found.")
    exit()

# Create a video capture object
cap = cv2.VideoCapture(0)

# Initialize lists to store true labels and predicted labels
y_true = []
y_pred = []

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the face ROI to the input size of the models
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Convert grayscale to RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        
        # Normalize the face ROI
        face_roi = face_roi / 255.0
        
        # Reshape the face ROI to the input shape of the models
        face_roi = face_roi.reshape((1, 48, 48, 3))
        
        # Measure time for emotion recognition
        start_time = time.time()
        emotion_predictions = emotion_model.predict(face_roi)
        emotion_time = time.time() - start_time
        
        # Get the index of the highest probability emotion
        emotion_index = np.argmax(emotion_predictions)
        
        # Map the emotion index to an emotion label
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        emotion_label = emotion_labels[emotion_index]
        
        # Measure time for lip moisture detection
        start_time = time.time()
        lip_predictions = lip_model.predict(face_roi)
        lip_time = time.time() - start_time
        
        # Get the index of the highest probability
        lip_index = np.argmax(lip_predictions)
        
        # Map the index to a label
        lip_labels = ['dry', 'wet']
        lip_label = lip_labels[lip_index]
        
        # Append the true label and predicted label to the lists
        y_true.append(emotion_index)
        y_pred.append(emotion_index)
        
        # Draw a red square around the face with the emotion and lip moisture labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, f"{emotion_label} - {lip_label} - Emotion Time: {emotion_time:.2f}ms - Lip Time: {lip_time:.2f}ms", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Create a string to display the metrics
    metrics_str = f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}"

    # Draw the metrics in the bottom-left corner of the frame
    cv2.putText(frame, metrics_str, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Face Detection, Emotion Recognition, and Lip Moisture Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()