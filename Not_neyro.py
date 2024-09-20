import cv2

# создать новый объект камеру
cap = cv2.VideoCapture(0)
# инициализировать поиск лица (по умолчанию каскад Хаара)
face_cascade = cv2.CascadeClassifier("face_bok2.xml")

while True:
    # чтение изображения с камеры
    _, image = cap.read()
    # преобразование к оттенкам серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # обнаружение лиц на фотографии
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # для каждого обнаруженного лица нарисовать синий квадрат
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    #cropped = image[y:y + height, x:x + width]
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()