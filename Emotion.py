import cv2

from deepface import DeepFace

import matplotlib.pyplot as plt
test_img = cv2.imread("/home/tapok/Документы/Test_hlebalo/ingT/pushkin.jpg")

while True:
    
    plt.imshow(test_img[:,:,::-1])
    emo_detector = fer(mtcnn=True)
    captured_emotions = emo_detector.detect_emotions(test_img)
    dominant_emmotion, emotion_score = emo_detector.top_emotion(test_img)

    print(dominant_emmotion, emotion_score)
    #captured_emotions