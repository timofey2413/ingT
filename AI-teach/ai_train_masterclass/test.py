import pickle
import cv2
import numpy as np
from mediapipe1 import get_face_landmarks

# Определение эмоций
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Загрузка модели
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Открытие видеопотока
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть видеопоток.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр.")
        break
    
    # Получение лицевых ключевых точек
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
    
    if face_landmarks is not None and len(face_landmarks) > 0:
        face_landmarks = np.array(face_landmarks).reshape(1, -1)  # Преобразование в нужный формат
        
        # Прогнозирование эмоции
        output = model.predict(face_landmarks)
        emotion_index = np.argmax(output)

        if 0 <= emotion_index < len(emotions):
            cv2.putText(frame,
                        emotions[emotion_index],
                        org=(10, frame.shape[0] - 10),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=3,
                        color=(255, 0, 0),
                        thickness=5)
        else:
            print(f"Недопустимый индекс эмоции: {emotion_index}")
    else:
        print("Лицевые ключевые точки не найдены.")
    
    cv2.imshow("frame", frame)
    
    key_press = cv2.waitKey(30)
    if key_press == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()