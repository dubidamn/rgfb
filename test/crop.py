import cv2
import os

# Function to create the output directory if it doesn't exist
def create_output_directory(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

# Function to save the cropped face images with specified dimensions
def save_cropped_faces(faces, frame, output_dir, frame_index, target_size=(80, 45)):
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_img)
        if len(eyes) >= 2:  # Check if at least two eyes are detected
            # Resize the face image to the target size
            resized_face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
            cropped_face_path = os.path.join(output_dir, f'face_{frame_index}_{i}.jpg')
            cv2.imwrite(cropped_face_path, resized_face_img)

# Define the path for the Haar cascade files
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

# Load the cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Define your video path and output directory for your local machine
video_path = "/Users/budianto/Desktop/S1R1.mp4"  # Path to the video file on your desktop
output_dir = "/Users/budianto/Desktop/Captures/"  # Path to the directory where the captures will be saved

# Create the output directory
create_output_directory(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open the video file")

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there are no frames left

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Save the cropped faces with the specified dimensions
    save_cropped_faces(faces, frame, output_dir, frame_index)
    
    frame_index += 1

# Release the video capture object
cap.release()

# Provide the number of faces cropped
total_faces_cropped = frame_index
print(f"Total faces cropped: {total_faces_cropped}")
