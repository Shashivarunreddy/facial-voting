import cv2
import numpy as np
import os
import time
import pickle
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import random

# Path to the Dlib shape predictor file
PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"

# Check if the file exists
if not os.path.isfile(PREDICTOR_PATH):
    raise FileNotFoundError(f"Shape predictor file not found at {PREDICTOR_PATH}. Please download it and set the correct path.")

# Constants
EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 3  # Number of frames for blink detection
BLINKS_REQUIRED = 2  # Minimum blinks required to confirm liveness
DATA_DIR = 'data/'
FRAMES_TOTAL = 51
CAPTURE_AFTER_FRAME = 2

# Actions
ACTIONS = ["Turn Left", "Turn Right", "Smile", "Blink Twice"]
ACTION_TIMEOUT = 10  # Time limit to perform action (seconds)

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Load shape predictor

# Camera matrix for 3D pose estimation
camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])

# Define helper functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_random_action():
    return random.choice(ACTIONS)

def verify_action(action, gray, shape, total_blinks, timeout_start):
    """
    Verify if the user performed the requested action.
    """
    if action == "Turn Left" or action == "Turn Right":
        # Use head pose estimation for precise left/right turn detection
        rotation_y = get_head_rotation(shape)
        if action == "Turn Left" and rotation_y < -10:  # Threshold for turning left
            return True
        if action == "Turn Right" and rotation_y > 10:  # Threshold for turning right
            return True

    elif action == "Smile":
        # Check mouth curvature
        mouth = shape[48:68]  # Mouth landmarks
        mouth_width = dist.euclidean(mouth[0], mouth[6])
        mouth_height = dist.euclidean(mouth[3], mouth[9])
        smile_ratio = mouth_height / mouth_width
        if smile_ratio > 0.3:  # Smile detected
            return True

    elif action == "Blink Twice":
        # Check for two blinks
        if total_blinks >= 2:
            return True

    # Check timeout
    if time.time() - timeout_start > ACTION_TIMEOUT:
        return False

    return None

def get_head_rotation(shape):
    """
    Calculates the rotation of the head along the Y-axis using facial landmarks.
    """
    # 3D model points (some key points: nose, eyes, chin, etc.)
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype="double")

    # Image points (corresponding to the landmarks)
    image_points = np.array([
        shape[30],  # Nose tip
        shape[8],  # Chin
        shape[36],  # Left eye left corner
        shape[45],  # Right eye right corner
        shape[48],  # Left mouth corner
        shape[54]  # Right mouth corner
    ], dtype="double")

    # Solve PnP (Perspective-n-Point) to find rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, None)

    # Get the rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Create a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    # Decompose the projection matrix to get Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    # Extract the yaw (Y-axis rotation) from Euler angles
    rotation_y = euler_angles[1, 0]  # Yaw

    return rotation_y


# Initialize variables
left_eye_idxs = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
right_eye_idxs = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
blink_counter = 0
total_blinks = 0
face_data = []
i = 0

# Get user roll number
name = input("Enter your Roll number: ")
random_action = get_random_action()
print(f"Perform this action: {random_action}")

timeout_start = time.time()
action_completed = False

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye regions
        left_eye = shape[left_eye_idxs[0]:left_eye_idxs[1]]
        right_eye = shape[right_eye_idxs[0]:right_eye_idxs[1]]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check for blink
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

        # Verify the action
        result = verify_action(random_action, gray, shape, total_blinks, timeout_start)
        if result is True:
            action_completed = True
            break
        elif result is False:
            print("Action timeout. Please try again.")
            video.release()
            cv2.destroyAllWindows()
            exit()

        # Draw rectangles and landmarks
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    if action_completed:
        print("Action verified! Proceeding to capture face data.")
        break

    cv2.putText(frame, f"Action: {random_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Proceed to capture face data (similar to previous implementation)
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  # Use dlib detector

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye regions
        left_eye = shape[left_eye_idxs[0]:left_eye_idxs[1]]
        right_eye = shape[right_eye_idxs[0]:right_eye_idxs[1]]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Check for blink
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

        # Draw rectangles and landmarks
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # If sufficient blinks are detected, capture the face data
    if total_blinks >= BLINKS_REQUIRED:
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(face_data) <= FRAMES_TOTAL and i % CAPTURE_AFTER_FRAME == 0:
                face_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display frame
    cv2.imshow('frame', frame)

    # Break conditions
    if cv2.waitKey(1) == ord('q') or len(face_data) >= FRAMES_TOTAL:
        break

# Save face data
video.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
face_data = face_data.reshape((FRAMES_TOTAL, -1))

# Save or update names.pkl
names_file = os.path.join(DATA_DIR, 'names.pkl')
if not os.path.isfile(names_file):
    names = [name] * FRAMES_TOTAL
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    try:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
    except EOFError:
        names = []
    names += [name] * FRAMES_TOTAL
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save or update faces_data.pkl
faces_file = os.path.join(DATA_DIR, 'faces_data.pkl')
if not os.path.isfile(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(face_data, f)
else:
    try:
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
    except EOFError:
        faces = np.empty((0, face_data.shape[1]))
    faces = np.append(faces, face_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print("Face data successfully captured with liveness detection.")

