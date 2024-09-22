import cv2
import os

# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier("Face/face_bok2.xml")

# Set the folder path and the output folder path
folder_path = "dry"
output_folder_path = "customarchive/dry"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Check if the image is read successfully
        if image is None:
            print(f"Error: Unable to read image file {image_path}")
            continue

        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        # Loop through each detected face
        for x, y, width, height in faces:
            # Increase the size of the cropped area to include the whole head
            x -= width // 4
            y -= height // 2
            width += width // 2
            height += height // 2

            # Ensure the cropped area is within the image boundaries
            x = max(0, x)
            y = max(0, y)
            width = min(width, image.shape[1] - x)
            height = min(height, image.shape[0] - y)

            # Crop the face
            cropped_face = image[y:y + height, x:x + width]

            # Save the cropped face with the same filename as the original image
            output_filename = filename
            output_path = os.path.join(output_folder_path, output_filename)
            cv2.imwrite(output_path, cropped_face)