import tensorflow as tf
import tensorflow_hub as hub

# Загрузите модель FaceNet (или EfficientNet)
model = hub.load("https://tfhub.dev/OpenAI/tf2-preview/facenet/r160x160/M1/1")

# Обработка изображения
image = tf.io.read_file("dataSet/pushkin.jpg")
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, (160, 160))
image = tf.cast(image, tf.float32) / 255.0

# Создание кодировки лица
embedding = model(tf.expand_dims(image, axis=0))