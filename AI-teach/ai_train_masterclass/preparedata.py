import os
import cv2
import numpy as np
from mediapipe1 import get_face_landmarks

data_dir = 'archive1kk'
output = []

for emotion_i, emotion in enumerate(sorted(os.listdir(data_dir))):
    # Skip .DS_Store files
    if emotion == ".DS_Store":
        continue 
    
    emotion_path = os.path.join(data_dir, emotion)
    for image_path_ in os.listdir(emotion_path):
        # Skip .DS_Store files 
        if image_path_ == ".DS_Store":
            continue
        
        image_path = os.path.join(emotion_path, image_path_)
        image = cv2.imread(image_path)
        face_landmarks = get_face_landmarks(image)
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_i))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))