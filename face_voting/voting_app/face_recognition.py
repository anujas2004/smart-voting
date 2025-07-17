# import cv2
# import numpy as np
# import pickle
# import os
# from sklearn.neighbors import KNeighborsClassifier
# import pyttsx3

# facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def register_face(aadhar_number):
#     if not os.path.exists('data/'):
#         os.makedirs('data/')

#     video = cv2.VideoCapture(0)
#     faces_data = []
#     i = 0
#     frames_total = 50
#     capture_after_frame = 2

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w]
#             resized_img = cv2.resize(crop_img, (50,50))

#             if len(faces_data) <= frames_total and i % capture_after_frame == 0:
#                 faces_data.append(resized_img)

#             i += 1
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

#         cv2.imshow('Register Face', frame)
#         k = cv2.waitKey(1)
#         if k == ord('q') or len(faces_data) >= frames_total:
#             break

#     video.release()
#     cv2.destroyAllWindows()

#     faces_data = np.asarray(faces_data).reshape((frames_total, -1))

#     with open(f'data/{aadhar_number}.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)

#     return True

# def recognize_face():
#     video = cv2.VideoCapture(0)
#     face_data_list = []
#     labels = []

#     for file in os.listdir('data/'):
#         if file.endswith('.pkl'):
#             aadhar_number = file.split('.')[0]
#             with open(f'data/{file}', 'rb') as f:
#                 faces = pickle.load(f)
#                 for face in faces:
#                     face_data_list.append(face)
#                     labels.append(aadhar_number)

#     if not face_data_list:
#         print("No registered faces found.")
#         return None

#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(face_data_list, labels)

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)

#         # Check if no faces were detected
#         if len(faces) == 0:
#             cv2.imshow('Face Recognition', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             continue

#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w]
#             resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

#             result = knn.predict(resized_img)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 2)
#             cv2.putText(frame, result[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             cv2.imshow('Face Recognition', frame)
            
#             # Check if face is recognized
#             if result[0] in labels:
#                 video.release()
#                 cv2.destroyAllWindows()
#                 return result[0]
#             else:
#                 # If not recognized, generate audio feedback and show error page
#                 engine = pyttsx3.init()
#                 engine.say("You are not registered.")
#                 engine.runAndWait()
                
#                 # Here you would redirect or change view to show the error message
#                 print("User not registered. Redirecting to error page.")
#                 return None

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()
#     return None
import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyttsx3

# Load the face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    # Convert to grayscale if not already
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard size
    face_img = cv2.resize(face_img, (50, 50))
    
    # Normalize pixel values
    face_img = face_img.astype(np.float32) / 255.0
    
    # Apply histogram equalization for better contrast
    face_img = cv2.equalizeHist(face_img.astype(np.uint8))
    face_img = face_img.astype(np.float32) / 255.0
    
    return face_img

def register_face(aadhar_number):
    # Create the data directory if it doesn't exist
    if not os.path.exists('data/'):
        os.makedirs('data/')

    # Initialize the video capture object
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video capture")
        return False

    # Initialize the face data list
    faces_data = []
    labels = []

    # Set the number of frames to capture
    frames_total = 100  # Increased number of frames
    capture_after_frame = 1  # Capture every frame

    # Initialize the frame counter
    i = 0
    print("Starting face registration. Please look at the camera and move your face slowly.")

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame with improved parameters
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            crop_img = frame[y:y+h, x:x+w]

            # Preprocess the face
            processed_face = preprocess_face(crop_img)

            # Add the face to the face data list
            if len(faces_data) <= frames_total:
                faces_data.append(processed_face.flatten())
                labels.append(aadhar_number)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

        # Display the frame
        cv2.imshow('Register Face', frame)

        # Check for the 'q' key press
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) >= frames_total:
            break

        # Increment the frame counter
        i += 1

    # Release the video capture object
    video.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    if len(faces_data) < 20:  # Minimum number of samples required
        print("Error: Not enough face samples captured. Please try again.")
        return False

    # Save the face data to a pickle file
    try:
        with open(f'data/{aadhar_number}.pkl', 'wb') as f:
            pickle.dump((faces_data, labels), f)
        print(f"Successfully registered {len(faces_data)} face samples for Aadhar: {aadhar_number}")
        return True
    except Exception as e:
        print(f"Error saving face data: {str(e)}")
        return False

def recognize_face():
    # Initialize the video capture object
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video capture")
        return None

    # Initialize the face data list and labels
    face_data_list = []
    labels = []

    # Load the face data from the pickle files
    print("Loading registered faces...")
    try:
        for file in os.listdir('data/'):
            if file.endswith('.pkl'):
                print(f"Loading face data from {file}")
                with open(f'data/{file}', 'rb') as f:
                    faces, face_labels = pickle.load(f)
                    face_data_list.extend(faces)
                    labels.extend(face_labels)
    except Exception as e:
        print(f"Error loading face data: {str(e)}")
        return None

    # Check if there are any registered faces
    if not face_data_list:
        print("No registered faces found.")
        return None

    print(f"Total registered faces: {len(face_data_list)}")
    print(f"Unique labels: {set(labels)}")

    # Convert lists to numpy arrays for better performance
    X = np.array(face_data_list)
    y = np.array(labels)

    # Initialize the KNN model with more neighbors for better accuracy
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    # Train the KNN model on all data
    knn.fit(X, y)

    print("Starting face recognition. Please look at the camera.")

    # Initialize variables for continuous recognition
    recognition_count = 0
    required_recognitions = 3  # Number of consecutive recognitions required
    last_recognized_label = None

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame with improved parameters
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            crop_img = frame[y:y+h, x:x+w]

            # Preprocess the face
            processed_face = preprocess_face(crop_img)

            # Predict the label of the face
            try:
                result = knn.predict(processed_face.flatten().reshape(1, -1))
                confidence = knn.predict_proba(processed_face.flatten().reshape(1, -1))
                max_confidence = np.max(confidence)
                
                print(f"Predicted label: {result[0]}")
                print(f"Confidence: {max_confidence:.2f}")
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.putText(frame, f"{result[0]} ({max_confidence:.2f})", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow('Face Recognition', frame)

                # Check if the face is recognized with high confidence
                if result[0] in labels and max_confidence > 0.5:  # Lowered confidence threshold
                    if last_recognized_label == result[0]:
                        recognition_count += 1
                    else:
                        recognition_count = 1
                        last_recognized_label = result[0]

                    if recognition_count >= required_recognitions:
                        print(f"Face recognized successfully with label: {result[0]}")
                        video.release()
                        cv2.destroyAllWindows()
                        return result[0]
                else:
                    recognition_count = 0
                    last_recognized_label = None
                    print(f"Face not recognized or low confidence. Predicted: {result[0]}, Confidence: {max_confidence:.2f}")
                    engine = pyttsx3.init()
                    engine.say("You are not registered.")
                    engine.runAndWait()
                    print("User not registered. Redirecting to error page.")
                    return None
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                continue

        # Check for the 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return None
