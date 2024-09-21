import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Specify the path to the train folder
train_folder = 'archive/train'

# Get the list of emotion folders
emotion_folders = [f for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))]

# Add sweat folder to the list
emotion_folders.append('sweat')

# Create a dictionary to map emotion folders to integer labels
emotion_labels = {folder: i for i, folder in enumerate(emotion_folders)}

# Initialize lists to store image paths and labels
image_paths = []
labels = []
sweat_labels = []

# Loop through each emotion folder
for emotion_folder in emotion_folders:
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(os.path.join(train_folder, emotion_folder)) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Loop through each image file
    for image_file in image_files:
        # Get the full path to the image file
        image_path = os.path.join(train_folder, emotion_folder, image_file)
        
        # Get the label (emotion) from the folder name and map it to an integer label
        label = emotion_labels[emotion_folder]
        
        # Append the image path and label to the lists
        image_paths.append(image_path)
        labels.append(label)
        
        # Add sweat label (1 if sweat, 0 if not)
        if emotion_folder == 'sweat':
            sweat_labels.append(1)
        else:
            sweat_labels.append(0)

# Load and preprocess the images
X = []
for image_path in image_paths:
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure the image is loaded with three color channels
    image = image.resize((48, 48))  # Resize to 48x48
    image = np.array(image) / 255.0  # Normalize pixel values
    X.append(image)

X = np.array(X)

# Convert labels to categorical
y = to_categorical(labels)

# Convert sweat labels to categorical
y_sweat = np.array(sweat_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, y_train_sweat, y_test_sweat = train_test_split(X, y, y_sweat, test_size=0.05, random_state=42)

# Define the model architecture
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotion_folders) - 1, activation='softmax'))  # Emotion recognition output
model.add(Dense(1, activation='sigmoid'))  # Sweat detection output

# Compile the model
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'binary_crossentropy'], metrics=['accuracy', 'accuracy'])

# Train the model
model.fit(X_train, [y_train, y_train_sweat], epochs=10, batch_size=32, validation_data=(X_test, [y_test, y_test_sweat]))

# Save the trained model
model.save('emotion_model_sweat.h5')