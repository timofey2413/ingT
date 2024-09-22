# подключаем библиотеку машинного зрения
import cv2
# библиотека для вызова системных функций
import os
import sqlite3 as sl
# получаем путь к этому скрипту
path = os.path.dirname(os.path.abspath(__file__))
# указываем, что мы будем искать лица по примитивам Хаара
detector = cv2.CascadeClassifier("face.xml")
# счётчик изображений
i=0
# расстояния от распознанного лица до рамки
con = sl.connect('test.db')
cursor = con.cursor()
# открываем базу
with con:
    # получаем количество таблиц с нужным нам именем
    data = con.execute("select count(*) from sqlite_master where type='table' and name='bd'")
    for row in data:
        # если таких таблиц нет
        if row[0] == 0:
            
            # создаём таблицу для товаров
            with con:
                con.execute("""
                    CREATE TABLE bd(
                        id PRIMARY KEY,
                        name TEXT
                    );
                """)

offset=50

# запрашиваем номер пользователя
id = input("Введите id пользователя: ")
name = input(str("Введите имя пользователя: "))
#cursor.execute("INSERT INTO 'bd' (id) VALUES (?)ON CONFLICT(id) DO UPDATE SET id=execute.id;", (id,))
#cursor.execute("INSERT INTO 'bd' ('name') VALUES (?)", (name,))
#cursor.execute("INSERT INTO 'bd' ('id') values(?)", (id,))
#cursor.execute('INSERT INTO bd (name) values(?)', (name))

# получаем доступ к камере
video=cv2.VideoCapture(0)
# запускаем цикл
while True:
    # берём видеопоток
    ret, im =video.read()
    # переводим всё в ч/б для простоты
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # настраиваем параметры распознавания и получаем лицо с камеры
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
    # обрабатываем лица
    for(x,y,w,h) in faces:
        # увеличиваем счётчик кадров
        i=i+1
        # записываем файл на диск
        cv2.imwrite("dataSet/face-"+str(id) +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        # формируем размеры окна для вывода лица
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        # показываем очередной кадр, который мы запомнили
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        # делаем паузу
        cv2.waitKey(100)
    # если у нас хватает кадров
    if i>30:
        # освобождаем камеру
        video.release()
        # удалаяем все созданные окна
        cv2.destroyAllWindows()
        # останавливаем цикл
        break