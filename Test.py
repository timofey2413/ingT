import cv2

image = cv2.imread("pushkin.jpg")
# преобразуем изображение к оттенкам серого
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# инициализировать распознаватель лиц (каскад Хаара по умолчанию)
face_cascade = cv2.CascadeClassifier("face.xml")
# обнаружение всех лиц на изображении
faces = face_cascade.detectMultiScale(image_gray)
# печатать количество найденных лиц
print(f"{len(faces)} лиц обнаружено на изображении.")
# для всех обнаруженных лиц рисуем синий квадрат
for x, y, width, height in faces:
   cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
   cv2.imwrite("pushkin_detected.jpg", image)