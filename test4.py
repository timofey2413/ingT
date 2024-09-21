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

# Check if the sweat detection model file exists
if os.path.exists('sweat_model.h5'):
    # Load the sweat detection model
    sweat_model = load_model('sweat_model.h5')
else:
    print("Error: File 'sweat_model.h5' not found.")
    exit()

# Create a video capture object
cap = cv2.VideoCapture(0)

def calculate_ear(eye_roi):
    # Calculate the eye aspect ratio (EAR)
    # This implementation is simplified and may not work well for all cases
    # You can improve it by using more advanced techniques, such as contour detection
    h, w = eye_roi.shape
    ear = (w / h)
    return ear

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
        
        # Check if there are any eyes detected
        if len(eyes) > 0:
            # Calculate the average EAR
            ear_avg = ear_sum / len(eyes)
        else:
            ear_avg = 0  # or some other default value
            blink_detected = False  # or some other default value
        
        # Make predictions on the face ROI using the emotion model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        face_roi = face_roi / 255.0
        face_roi = face_roi.reshape((1, 48, 48, 3))
        predictions = emotion_model.predict(face_roi)
        emotion_index = np.argmax(predictions)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        emotion_label = emotion_labels[emotion_index]

        # Add sweat detection
        sweat_roi = face_roi.copy()  # Use the same face ROI for sweat detection
        sweat_roi = sweat_roi / 255.0  # Normalize the input
        sweat_roi = sweat_roi.reshape((1, 48, 48, 3))
        sweat_predictions = sweat_model.predict(sweat_roi)
        sweat_probability = sweat_predictions[0][0]  # Get the sweat probability

        # Draw a rectangle around the face with the emotion label and blink detection
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Blink: {}".format("Yes" if blink_detected else "No"), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Emotion: {}".format(emotion_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Sweat: {:.2f}%".format(sweat_probability * 100), (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the output
    cv2.imshow('Face Detection, Emotion Recognition, Blink Detection, and Sweat Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()ы