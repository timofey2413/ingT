import cv2
import numpy as np
from keras.models import load_model
import os
import tkinter as tk
from PIL import Image, ImageTk
import time

# Get the current platform (Windows or macOS)
platform = os.name

# Set the folder path based on the platform
if platform == 'nt':  # Windows
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')
else:  # macOS
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')

# Load the cascade classifier for face detection
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the emotion recognition model file exists
emotion_model_path = os.path.join(folder_path, 'emotion_model7.h5')
if os.path.exists(emotion_model_path):
    # Load the emotion recognition model
    emotion_model = load_model(emotion_model_path)
else:
    exit()

# Check if the lip moisture detection model file exists
lip_model_path = os.path.join(folder_path, 'wet_dry_model.h5')
if os.path.exists(lip_model_path):
    # Load the lip moisture detection model
    lip_model = load_model(lip_model_path)
else:
    exit()

# Create a Tkinter window
window = tk.Tk()
window.title("Face Detection, Emotion Recognition, and Lip Moisture Detection")



# Create a canvas to display the video
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Get a list of available cameras
camera_indices = []
for i in range(10):  # Try up to 10 cameras
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        camera_indices.append(i)
        cap.release()

# Create a dropdown menu to select the camera
camera_var = tk.StringVar()
camera_var.set(camera_indices[0])  # Default to the first camera
camera_menu = tk.OptionMenu(window, camera_var, *camera_indices)
camera_menu.pack()

# Create labels to display the results
emotion_label_var = tk.StringVar()
lip_label_var = tk.StringVar()
stress_level_var = tk.StringVar()
threat_probability_var = tk.StringVar()
prediction_time_var = tk.StringVar()
accuracy_var = tk.StringVar()

label1 = tk.Label(window, textvariable=emotion_label_var)
label1.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)
label2 = tk.Label(window, textvariable=lip_label_var)
label2.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)
label3 = tk.Label(window, textvariable=stress_level_var)
label3.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)
label4 = tk.Label(window, textvariable=threat_probability_var)
label4.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)
label5 = tk.Label(window, textvariable=prediction_time_var)
label5.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)
label6 = tk.Label(window, textvariable=accuracy_var)
label6.pack(side=tk.LEFT, fill=tk.X, expand=False, pady=10)

# Add a label to the dropdown menu
camera_label = tk.Label(window, text="Камеры")
camera_label.pack()

# Create a video capture object
cap = cv2.VideoCapture(int(camera_var.get()))

start_time = time.time()

def update_frame():
    global start_time
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Resize the frame
    frame = cv2.resize(frame, (640, 480))
    
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
        
        # Measure the time before making predictions
        start_prediction_time = time.time()
        
        # Make predictions on the face ROI using the emotion model
        emotion_predictions = emotion_model.predict(face_roi)
        
        # Get the index of the highest probability emotion
        emotion_index = np.argmax(emotion_predictions)
        
        # Map the emotion index to an emotion label
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
        emotion_label = emotion_labels[emotion_index]
        
        # Calculate the threat probability based on the emotion label
        threat_probability = 0.0
        if emotion_label in ['Angry', 'Fear', 'Disgust']:
            threat_probability = 0.7
        elif emotion_label in ['Sad', 'Surprise']:
            threat_probability = 0.3
        else:
            threat_probability = 0.1

        # Calculate the stress level based on the emotion label
        stress_level = 0.0
        if emotion_label in ['Angry', 'Fear', 'Disgust']:
            stress_level = 0.8
        elif emotion_label in ['Sad', 'Surprise']:
            stress_level = 0.5
        else:
            stress_level = 0.2

        # Make predictions on the face ROI using the lip moisture detection model
        lip_predictions = lip_model.predict(face_roi)
        
        # Get the index of the highest probability
        lip_index = np.argmax(lip_predictions)
        
        # Map the index to a label
        lip_labels = ['dry', 'wet']
        lip_label = lip_labels[lip_index]
        
        # Measure the time after making predictions
        end_prediction_time = time.time()
        
        # Calculate the time elapsed for making predictions
        prediction_time = (end_prediction_time - start_prediction_time) * 1000  # Convert to milliseconds
        
        # Update the labels
        emotion_label_var.set(f"Emotion: {emotion_label}")
        lip_label_var.set(f"Lip Moisture: {lip_label}")
        stress_level_var.set(f"Stress Level: {stress_level:.2f}")
        threat_probability_var.set(f"Threat Probability: {threat_probability:.2f}")
        prediction_time_var.set(f"Prediction Time: {prediction_time:.2f} ms")
        accuracy_var.set(f"Accuracy: {emotion_label}")  # Assuming accuracy is the same as emotion label
        
        # Draw a red square around the face with the emotion, lip moisture, threat probability, stress level, and prediction time labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, f"{emotion_label} - {lip_label} - Threat: {threat_probability:.2f} - Stress: {stress_level:.2f} - Prediction Time: {prediction_time:.2f} ms", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Convert the frame to a Tkinter-compatible image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Get the current width and height of the image
    width, height = img.size

    # Calculate the new width and height while maintaining the aspect ratio
    new_width = 800
    new_height = int(height * (new_width / width))

    # Resize the image
    img = img.resize((new_width, new_height))

    img = ImageTk.PhotoImage(img)
    
    # Display the image on the canvas
    canvas.create_image(0, 0, image=img, anchor='nw')
    canvas.image = img
    
    # Update the window
    window.after(1, update_frame)

# Start the video capture
update_frame()

# Run the Tkinter event loop
window.mainloop()

# Release the video capture object
cap.release()
cv2.destroyAllWindows()