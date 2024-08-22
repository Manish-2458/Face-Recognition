import face_recognition
import os

# Directories for known and test images
KNOWN_FACES_DIR = 'known_faces'
TEST_FACES_DIR = 'test_faces'

# Lists to hold the encodings and names of known faces
known_faces = []
known_roll_numbers = []

# Load known faces
for roll_number in os.listdir(KNOWN_FACES_DIR):
    roll_dir = os.path.join(KNOWN_FACES_DIR, roll_number)
    for filename in os.listdir(roll_dir):
        image_path = os.path.join(roll_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_roll_numbers.append(roll_number)

# Counters for accuracy metrics
TP = 0  # True Positives
TN = 0  # True Negatives
FP = 0  # False Positives
FN = 0  # False Negatives

# Evaluate test images
for filename in os.listdir(TEST_FACES_DIR):
    image_path = os.path.join(TEST_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if face_encodings:
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.4)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_roll_numbers[first_match_index]

            # Assume the filename contains the correct roll number
            true_name = filename.split('_')[0]
            if name == true_name:
                TP += 1
            else:
                FP += 1
        else:
            # If the model says "Unknown" but it should have recognized someone
            true_name = filename.split('_')[0]
            if true_name == "Unknown":
                TN += 1
            else:
                FN += 1
    else:
        # If no face is detected, count as a False Negative
        FN += 1

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f'Accuracy: {accuracy:.2f}')
print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
