import cv2
import mediapipe as mp

# Инициализация Mediapipe для распознавания лиц
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Инициализация видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить видео.")
        break

    # Преобразование цвета с BGR на RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   
    # Обработка изображения с помощью Mediapipe
    results = face_mesh.process(image_rgb)


    # Проверка, обнаружены ли ключевые точки лица
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Отображение результата
    cv2.imshow('Face Mesh', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()