import cv2
import numpy as np
import requests

# Set the ESP-CAM's IP address
esp_cam_ip = 'http://192.168.1.100'  # Replace with your ESP-CAM's IP address

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Send a GET request to the ESP-CAM to capture an image
    response = requests.get(esp_cam_ip + '/capture')
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the image from the response
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw a red square around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Display the output
        cv2.imshow('Face Detection', frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cv2.destroyAllWindows()