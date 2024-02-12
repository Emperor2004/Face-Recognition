import cv2
import numpy
import face_recognition
from face_data import FACE_DATA


known_encodings = []
known_names = []

for image_path in FACE_DATA:
    img = face_recognition.load_image_file(image_path)
    img_encoding = face_recognition.face_encodings(img)[0]  # Assuming one face per image
    known_encodings.append(img_encoding)
    # Extract name from the image path (assuming the file name is the name of the person)
    name = image_path.split('/')[-1].split('.')[0]
    known_names.append(name)

# Open the camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Unable to open the camera.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # Compare the encoded face with known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, assign the name
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name below the face
        cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
