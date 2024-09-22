import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Specify the path to the train folder
train_folder = 'customarchive'

# Get the list of subfolders (dry and wet)
subfolders = [f for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))]

# Create a dictionary to map subfolders to labels
labels = {'dry': 0, 'wet': 1}

# Initialize lists to store image paths and labels
image_paths = []
labels_list = []

# Loop through each subfolder
for subfolder in subfolders:
    # Get the list of image files in the subfolder
    image_files = [f for f in os.listdir(os.path.join(train_folder, subfolder)) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Loop through each image file
    for image_file in image_files:
        # Get the full path to the image file
        image_path = os.path.join(train_folder, subfolder, image_file)
        
        # Get the label from the subfolder name
        label = labels[subfolder]
        
        # Append the image path and label to the lists
        image_paths.append(image_path)
        labels_list.append(label)

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
y = to_categorical(labels_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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
model.add(Conv2D(256, (3, 3), activation='relu'))  # added another layer
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # increased units
model.add(Dense(2, activation='softmax'))  # output layer with 2 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=30, validation_data=(X_test, y_test))  # increased epochs and batch size

# Save the trained model
model.save('wet_dry_model2.h5')