import cv2
import os
import time

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Place the XML file in the same directory as the script

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face

# Create a directory to store images
output_directory = 'collected_images'
os.makedirs(output_directory, exist_ok=True)

# Print the current working directory
print("Current working directory:", os.getcwd())

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Set the delay in seconds (e.g., 1 second)
image_capture_delay = 0.1

# Collect samples of your face from webcam input
while True:
    ret, frame = cap.read()
    extracted_face = face_extractor(frame)
    if extracted_face is not None:
        count += 1
        face = cv2.resize(extracted_face, (400, 400))

        if face is not None:  # Check if the face is not None
            # Save the file in the specified directory with a unique name
            file_name_path = os.path.join(output_directory, str(count) + '.jpg')
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Empty face detected, not saving")

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter key
        break

    # Add a delay to stabilize the camera
    time.sleep(image_capture_delay)

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
