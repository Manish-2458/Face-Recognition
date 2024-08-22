import face_recognition
import cv2
import numpy as np
import os

# Path to the directory containing known faces
KNOWN_FACES_DIR = 'known_faces'

# Load known faces
known_faces = []
known_roll_numbers = []

# Loop through each person’s directory in the known faces directory
for roll_number in os.listdir(KNOWN_FACES_DIR):
    roll_dir = os.path.join(KNOWN_FACES_DIR, roll_number)
    
    # Loop through each image in the person's directory
    for filename in os.listdir(roll_dir):
        image_path = os.path.join(roll_dir, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Get face encoding
        encodings = face_recognition.face_encodings(image)
        
        if encodings:  # Check if any face encodings were found
            encoding = encodings[0]  # Get the first face encoding
            known_faces.append(encoding)
            known_roll_numbers.append(roll_number)
        else:
            print(f"No face found in {image_path}")


# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.4)
        name = "Unknown"

        # Use the first match found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_roll_numbers[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for Esc key
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
