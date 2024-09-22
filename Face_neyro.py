import cv2
import tensorflow as tf
import tensorflow_hub as hub
import time

# Загрузите модель FaceNet
model = hub.load("https://tfhub.dev/OpenAI/tf2-preview/facenet/r160x160/M1/1")

# Загрузите кодировки лиц из файла (или создайте их)
with open("data/face_encodings.pickle", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Начните работу с веб-камерой
video_capture = cv2.VideoCapture(0)
fps = video_capture.get(cv2.CAP_PROP_FPS)

while True:
    start_time = time.time()

    # Считывание кадра с веб-камеры
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Преобразование изображения в тензор TensorFlow
    image = tf.convert_to_tensor(rgb_frame, dtype=tf.float32)
    image = tf.image.resize(image, (160, 160))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    # Создание кодировки лица
    embedding = model(image)

    # Сравнение с известными лицами
    match = face_recognition.compare_faces(known_face_encodings, embedding.numpy()[0], tolerance=0.5)
    name = "Неизвестно"
    if True in match:
        first_match_index = match.index(True)
        name = known_face_names[first_match_index]

    # Отображение результата
    cv2.rectangle(frame, (0, 0), (150, 35), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (10, 30), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Распознавание лиц', frame)

    # Управление скоростью обработки
    end_time = time.time()
    time_taken = end_time - start_time
    time_to_wait = 1/fps - time_taken
    if time_to_wait > 0:
        time.sleep(time_to_wait)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Закрытие потока видео и окна
video_capture.release()
cv2.destroyAllWindows()