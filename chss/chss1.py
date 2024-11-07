import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class VideoCapture(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video and Graph")
        
        # Установка фиксированного размера окна
        self.setFixedSize(800, 600)

        # Layout
        self.layout = QVBoxLayout()
        
        # OpenCV Video
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        # Matplotlib Figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.signal = []
        self.time_stamps = []
        self.count = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

    def detect_face(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            return frame[y:y+h, x:x+w], (x, y, w, h)  # Возвращаем также координаты лица
        else:
            return None, None

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            face, coords = self.detect_face(frame)

            if face is not None:
                # Отображение прямоугольника вокруг лица
                (x, y, w, h) = coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Извлечение области щек (примерно)
                cheek_area = frame[y + int(h/2):y + h, x + int(w/4):x + int(3*w/4)]  # Извлекаем нижнюю часть лица
                # Измените координаты, если хотите более точно настроить область

                # Извлечение цветовых каналов
                b, g, r = cv2.split(cheek_area)

                # Вычисление среднего значения красного канала
                mean_red = np.mean(r)
                self.signal.append(mean_red)
                self.time_stamps.append(self.count / self.fps)
                self.count += 1

                # Обновление графика
                self.plot_graph()

            # Преобразование изображения для отображения
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def plot_graph(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.time_stamps, self.signal, color='r')
        ax.set_xlabel('Время (с)')
        ax.set_ylabel('Среднее значение красного канала')
        ax.set_title('Фотоплетизмограмма')
        self.canvas.draw()

    def closeEvent(self, event):
        self.cap.release()  # Освобождение ресурсов
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoCapture()
    window.show()
    sys.exit(app.exec_())