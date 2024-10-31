import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.open_button = QPushButton("Open Video", self)
        self.open_button.clicked.connect(self.open_video)
        self.layout.addWidget(self.open_button)

        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None

    def open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.timer.start(30)  # Запускаем таймер для обновления кадров

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Преобразуем BGR в RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
            else:
                self.timer.stop()
                self.cap.release()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
