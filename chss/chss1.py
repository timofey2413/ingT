import cv2
import numpy as np
import matplotlib.pyplot as plt

# Определение области лица
def detect_face(frame):
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  if len(faces) > 0:
    (x, y, w, h) = faces[0]
    return frame[y:y+h, x:x+w]
  else:
    return None

# Обработка видеопотока с камеры
def process_camera():
  cap = cv2.VideoCapture(0)  # Используем камеру по умолчанию
  fps = cap.get(cv2.CAP_PROP_FPS)
  signal = []
  time_stamps = []
  count = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      face = detect_face(frame)
      if face is not None:
        # Извлечение цветовых каналов
        b, g, r = cv2.split(face)
        
        # Вычисление среднего значения красного канала
        mean_red = np.mean(r)
        
        # Добавление данных в сигналы
        signal.append(mean_red)
        time_stamps.append(count / fps)
        count += 1
      else:
        print("Лицо не обнаружено")
      cv2.imshow('Frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break
  cap.release()
  cv2.destroyAllWindows()
  return signal, time_stamps

# Запуск обработки видео с камеры
signal, time_stamps = process_camera()

# Отрисовка графика
plt.plot(time_stamps, signal)
plt.xlabel('Время (с)')
plt.ylabel('Среднее значение красного канала')
plt.title('Фотоплетизмограмма')
plt.show()