from insightface.app import FaceAnalysis
import cv2
app = FaceAnalysis(name="buffalo_sc",providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))  #подготовка нейросети
img = cv2.imread("/home/tapok/Документы/Test_hlebalo/ingT/Images/people.png") #считываем изображение
faces = app.get(img) #ищем лица на изображении и получаем информацию о них
for face in faces:
    print(face)
x, y, x2, y2 = face.bbox #получаем границы лица
cropped = img[int(y):int(y2), int(x):int(x2)] #вырезаем лицо из изображения
cv2.imshow('image', cropped) #показываем лицо
cv2.waitKey(0)