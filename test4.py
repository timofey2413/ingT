import cv2
import numpy as np
from keras.models import load_model
import os

# Load the cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if the emotion recognition model file exists
if os.path.exists('emotion_model6.h5'):
    # Load the emotion recognition model
    emotion_model = load_model('emotion_model6.h5')
else:
    print("Error: File 'emotion_recogniton_model.h5' not found.")
    exit()

# Create a video capture object
cap = cv2.VideoCapture(0)

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
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        # Initialize blink detection variables
        blink_detected = False
        ear_sum = 0
        
        # Loop through each eye
        for (ex, ey, ew, eh) in eyes:
            # Extract the eye ROI
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            
            # Calculate the eye aspect ratio (EAR)
            ear = calculate_ear(eye_roi)
            ear_sum += ear
            
            # Check if the EAR is below the threshold (0.2 in this case)
            if ear < 0.2:
                blink_detected = True
        
        # Calculate the average EAR
        ear_avg = ear_sum / len(eyes)
        
        # Draw a rectangle around the face with the emotion label and blink detection
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Blink: {}".format("Yes" if blink_detected else "No"), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Make predictions on the face ROI using the emotion model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        face_roi = face_roi / 255.0
        face_roi = face_roi.reshape((1, 48, 48, 3))
        predictions = emotion_model.predict(face_roi)
        emotion_index = np.argmax(predictions)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        emotion_label = emotion_labels[emotion_index]
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the output
    cv2.imshow('Face Detection, Emotion Recognition, and Blink Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

def calculate_ear(eye_roi):
    # Calculate the eye aspect ratio (EAR)
    # This implementation is simplified and may not work well for all cases
    # You can improve it by using more advanced techniques, such as contour detection
    h, w = eye_roi.shape
    ear = (w / h)
    return ear