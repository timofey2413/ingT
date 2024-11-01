import sys
import cv2
import numpy as np
from keras.models import load_model
import os
import time  # Добавлено
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtQml import QQmlApplicationEngine

# Get the current platform (Windows or macOS)
platform = os.name

# Set the stylesheet based on the platform
# with open('style.qss', 'r') as f:
#     app.setStyleSheet(f.read())

# Set the folder path based on the platform
if platform == 'nt':  # Windows
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')
else:  # macOS
    folder_path = os.path.join(os.path.expanduser('~'), 'Documents', 'ingt')
    
stylesheet_path = os.path.join(folder_path, 'style.qss')
# Load the cascade classifier for face detection
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Check if the emotion recognition model file exists
emotion_model_path = os.path.join(folder_path, 'emotion_model7.h5')
if os.path.exists(emotion_model_path):
    emotion_model = load_model(emotion_model_path)
else:
    exit()

# Check if the lip moisture detection model file exists
lip_model_path = os.path.join(folder_path, 'wet_dry_model.h5')
if os.path.exists(lip_model_path):
    lip_model = load_model(lip_model_path)
else:
    exit()

def load_stylesheet(filename):
    try:
        with open(filename, 'r') as file:  # Используем 'file' как контекстную переменную
            return file.read()
    except FileNotFoundError:
        print(f"Файл стилей '{filename}' не найден.")
        return ""  # Возвращаем пустую строку, если файл не найден
    except Exception as e:
        print(f"Ошибка при загрузке файла стилей: {e}")
        return ""  # Возвращаем пустую строку в случае других ошибок

class FaceEmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection, Emotion Recognition, and Lip Moisture Detection")

        # Create layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create a label for displaying the video
        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        # Create a dropdown menu to select the camera
        self.camera_combo = QComboBox()
        self.layout.addWidget(self.camera_combo)

        # Create labels to display the results
        self.emotion_label = QLabel("Emotion: ")
        self.layout.addWidget(self.emotion_label)
        self.lip_label = QLabel("Lip Moisture: ")
        self.layout.addWidget(self.lip_label)
        self.stress_level_label = QLabel("Stress Level: ")
        self.layout.addWidget(self.stress_level_label)
        self.threat_probability_label = QLabel("Threat Probability: ")
        self.layout.addWidget(self.threat_probability_label)
        self.prediction_time_label = QLabel("Prediction Time: ")
        self.layout.addWidget(self.prediction_time_label)
        self.accuracy_label = QLabel("Accuracy: ")
        self.layout.addWidget(self.accuracy_label)

        # Get a list of available cameras
        self.camera_indices = []
        for i in range(10):  # Try up to 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_indices.append(i)
                cap.release()

        # Populate the camera dropdown
        self.camera_combo.addItems(map(str, self.camera_indices))
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        # Create a video capture object
        self.cap = cv2.VideoCapture(int(self.camera_combo.currentText()))

        # Set up a timer to update the video frame
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def change_camera(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(int(self.camera_combo.currentText()))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face
        for (x, y, w, h) in faces:
            # Extract the face region of interest (ROI)
            face_roi = gray[y:y + h, x:x + w]

            # Resize the face ROI to the input size of the models
            face_roi = cv2.resize(face_roi, (48, 48))

            # Convert grayscale to RGB
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

            # Normalize the face ROI
            face_roi = face_roi / 255.0

            # Reshape the face ROI to the input shape of the models
            face_roi = face_roi.reshape((1, 48, 48, 3))

            # Measure the time before making predictions
            start_prediction_time = time.time()

            # Make predictions on the face ROI using the emotion model
            emotion_predictions = emotion_model.predict(face_roi)

            # Get the index of the highest probability emotion
            emotion_index = np.argmax(emotion_predictions)

            # Map the emotion index to an emotion label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Contempt']
            emotion_label = emotion_labels[emotion_index]

            # Calculate the threat probability based on the emotion label
            threat_probability = 0.0
            if emotion_label in ['Angry', 'Fear', 'Disgust']:
                threat_probability = 0.7
            elif emotion_label in ['Sad', 'Surprise']:
                threat_probability = 0.3
            else:
                threat_probability = 0.1

            # Calculate the stress level based on the emotion label
            stress_level = 0.0
            if emotion_label in ['Angry', 'Fear', 'Disgust']:
                stress_level = 0.8
            elif emotion_label in ['Sad', 'Surprise']:
                stress_level = 0.5
            else:
                stress_level = 0.2

            # Make predictions on the face ROI using the lip moisture detection model
            lip_predictions = lip_model.predict(face_roi)

            # Get the index of the highest probability
            lip_index = np.argmax(lip_predictions)

            # Map the index to a label
            lip_labels = ['dry', 'wet']
            lip_label = lip_labels[lip_index]

            # Measure the time after making predictions
            end_prediction_time = time.time()

            # Calculate the time elapsed for making predictions
            prediction_time = (end_prediction_time - start_prediction_time) * 1000  # Convert to milliseconds

            # Update the labels
            self.emotion_label.setText(f"Emotion: {emotion_label}")
            self.lip_label.setText(f"Lip Moisture: {lip_label}")
            self.stress_level_label.setText(f"Stress Level: {stress_level:.2f}")
            self.threat_probability_label.setText(f"Threat Probability: {threat_probability:.2f}")
            self.prediction_time_label.setText(f"Prediction Time: {prediction_time:.2f} ms")
            self.accuracy_label.setText(f"Accuracy: {emotion_label}")  # Assuming accuracy is the same as emotion label

            # Draw a red square around the face with the emotion, lip moisture, threat probability, stress level, and prediction time labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"{emotion_label} - {lip_label} - Threat: {threat_probability:.2f} - Stress: {stress_level:.2f} - Prediction Time: {prediction_time:.2f} ms", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Convert the frame to a QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Display the image on the label
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stylesheet = load_stylesheet(stylesheet_path)  # Загружаем стили из нового пути
    if stylesheet:
        app.setStyleSheet(stylesheet)  # Устанавливаем стиль только если он был загружен успешно
    window = FaceEmotionApp()
    window.show()
    sys.exit(app.exec_())