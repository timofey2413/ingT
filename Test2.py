import cv2
from fer import FER

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to RGB (FER expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create an instance of the FER detector
    emo_detector = FER(mtcnn=True)
    
    # Detect emotions in the frame
    captured_emotions = emo_detector.detect_emotions(frame_rgb)
    
    # Get the dominant emotion and its score
    dominant_emotion, emotion_score = emo_detector.top_emotion(frame_rgb)
    
    # Print the detected emotions and dominant emotion
    print("Captured emotions:", captured_emotions)
    print("Dominant emotion:", dominant_emotion, "with score:", emotion_score)
    
    # Display the frame with the detected emotions
    cv2.imshow("Detected Emotions", frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()