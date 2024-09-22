import cv2
from PIL import Image 
# создать новый объект камеру
cap = cv2.VideoCapture(0)
# инициализировать поиск лица (по умолчанию каскад Хаара)
face_cascade = cv2.CascadeClassifier("Face/face_bok2.xml")


    # чтение изображения с камеры
image = open("dataSet/train_00110_aligned.jpg")
# преобразование к оттенкам серого
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# обнаружение лиц на фотографии
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # для каждого обнаруженного лица нарисовать синий квадрат
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    cropped = image[y:y + height, x:x + width]
cv2.imshow("image", image)
cv2.imwrite("image.jpg", cropped)
