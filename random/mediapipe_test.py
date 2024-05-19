import mediapipe as mp
import cv2
import math
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class BodyParts(Enum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

def calc_xy_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def calc_xy_euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calc_point_to_line_distance(x1, y1, x2, y2, x3, y3):
    return abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def calc_angle_between_lines(x1, y1, x2, y2, x3, y3, x4, y4):
    angle1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle2 = math.degrees(math.atan2(y4 - y3, x4 - x3))
    return abs(angle1 - angle2)

def calc_face_rotation(landmarks):
    left_ear_x = landmarks[BodyParts.LEFT_EAR.value].x
    left_ear_z = landmarks[BodyParts.LEFT_EAR.value].z
    right_ear_x = landmarks[BodyParts.RIGHT_EAR.value].x
    right_ear_z = landmarks[BodyParts.RIGHT_EAR.value].z

    left_mouth_x = landmarks[BodyParts.MOUTH_LEFT.value].x
    left_mouth_z = landmarks[BodyParts.MOUTH_LEFT.value].z

    right_mouth_x = landmarks[BodyParts.MOUTH_RIGHT.value].x
    right_mouth_z = landmarks[BodyParts.MOUTH_RIGHT.value].z

    return (calc_xy_angle(right_ear_x, right_ear_z, left_ear_x, left_ear_z) + calc_xy_angle(right_mouth_x, right_mouth_z, left_mouth_x, left_mouth_z)) / 2

def calc_face_tilt(landmarks):
    left_eye_x = landmarks[BodyParts.LEFT_EAR.value].x
    left_eye_y = landmarks[BodyParts.LEFT_EAR.value].y
    right_eye_x = landmarks[BodyParts.RIGHT_EAR.value].x
    right_eye_y = landmarks[BodyParts.RIGHT_EAR.value].y
    left_mouth_x = landmarks[BodyParts.MOUTH_LEFT.value].x
    left_mouth_y = landmarks[BodyParts.MOUTH_LEFT.value].y
    right_mouth_x = landmarks[BodyParts.MOUTH_RIGHT.value].x
    right_mouth_y = landmarks[BodyParts.MOUTH_RIGHT.value].y

    return (calc_xy_angle(left_eye_y, left_eye_x, right_eye_y, right_eye_x) + calc_xy_angle(left_mouth_y, left_mouth_x, right_mouth_y, right_mouth_x)) / 2 + 90

def calc_face_to_shoulder_distance(landmarks):
    left_shoulder_x = landmarks[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_y = landmarks[BodyParts.LEFT_SHOULDER.value].y
    right_shoulder_x = landmarks[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_y = landmarks[BodyParts.RIGHT_SHOULDER.value].y
    nose_x = landmarks[BodyParts.NOSE.value].x
    nose_y = landmarks[BodyParts.NOSE.value].y

    return calc_point_to_line_distance(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, nose_x, nose_y)

def calc_face_shoulders_angle(landmarks):
    left_shoulder_x = landmarks[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_y = landmarks[BodyParts.LEFT_SHOULDER.value].y
    right_shoulder_x = landmarks[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_y = landmarks[BodyParts.RIGHT_SHOULDER.value].y
    
    return calc_face_tilt(landmarks) - calc_xy_angle(left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x) - 90

def calc_face_up_down_tilt(landmarks):
    left_eye_y = landmarks[BodyParts.LEFT_EAR.value].y
    left_eye_z = landmarks[BodyParts.LEFT_EAR.value].z
    right_eye_y = landmarks[BodyParts.RIGHT_EAR.value].y
    right_eye_z = landmarks[BodyParts.RIGHT_EAR.value].z
    left_mouth_y = landmarks[BodyParts.MOUTH_LEFT.value].y
    left_mouth_z = landmarks[BodyParts.MOUTH_LEFT.value].z
    right_mouth_y = landmarks[BodyParts.MOUTH_RIGHT.value].y
    right_mouth_z = landmarks[BodyParts.MOUTH_RIGHT.value].z

    return (calc_xy_angle(left_eye_z, left_eye_y, right_mouth_z, right_mouth_y) + calc_xy_angle(right_eye_z, right_eye_y, left_mouth_z, left_mouth_y)) / 2 - 180

def calc_body_tilt(landmarks):
    # use the nose and the shoulders
    left_shoulder_x = landmarks[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_y = landmarks[BodyParts.LEFT_SHOULDER.value].y
    right_shoulder_x = landmarks[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_y = landmarks[BodyParts.RIGHT_SHOULDER.value].y
    nose_x = landmarks[BodyParts.NOSE.value].x
    nose_y = landmarks[BodyParts.NOSE.value].y

    return calc_xy_angle(left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x) - calc_xy_angle(nose_y, nose_x, right_shoulder_y, right_shoulder_x)

def calc_body_rotation(landmarks):
    # use the shoulders
    left_shoulder_x = landmarks[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_z = landmarks[BodyParts.LEFT_SHOULDER.value].z
    right_shoulder_x = landmarks[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_z = landmarks[BodyParts.RIGHT_SHOULDER.value].z

    return calc_xy_angle(left_shoulder_z, left_shoulder_x, right_shoulder_z, right_shoulder_x) + 90

def scale_vector(arr, mag):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = np.interp(arr, (min_val, max_val), (0, mag))
    return normalized_arr

def find_metrics(landmark):
    # this will be an array of the metrics that i compare / pass into tensorflow
    left_shoulder_x = landmarks[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_y = landmarks[BodyParts.LEFT_SHOULDER.value].y
    right_shoulder_x = landmarks[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_y = landmarks[BodyParts.RIGHT_SHOULDER.value].y

    metrics = np.zeros(11)
    metrics[0] = calc_face_rotation(landmark) # face rotation
    metrics[1] = calc_face_tilt(landmark) # face tilt
    metrics[2] = calc_face_to_shoulder_distance(landmark) # face to shoulder distance
    metrics[3] = calc_face_shoulders_angle(landmark) # face to shoulders angle
    metrics[4] = calc_face_up_down_tilt(landmark) # face up down tilt
    metrics[5] = calc_body_tilt(landmark) # body tilt
    metrics[6] = calc_body_rotation(landmark) # body rotation
    metrics[7] = calc_xy_angle(left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x) # shoulder tilt
    metrics[9] = metrics[6] - metrics[0] # difference between face and body rotation
    metrics[8] = calc_xy_euclidean_distance(left_shoulder_y, left_shoulder_x, right_shoulder_y, right_shoulder_x) # shoulder width
    metrics[9] = calc_xy_euclidean_distance(landmark[BodyParts.LEFT_EAR.value].y, landmark[BodyParts.LEFT_EAR.value].x, landmark[BodyParts.RIGHT_EAR.value].y, landmark[BodyParts.RIGHT_EAR.value].x) # face width
    metrics[10] = metrics[9] - metrics[8] # difference in face and shoulder width
    
    return metrics

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(y_train);

    model = SVC(kernel='rbf', C=0.01, gamma=0.01)  # You can try different kernels like 'rbf', 'poly', etc.
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print('Cross-validation scores: ', scores)
    print('Average cross-validation score: ', scores.mean())

    y_pred = model.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return model

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Print the accuracy
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return clf

def predict(model, metrics):
    # Use the model to predict the pose
    metrics = np.array(metrics).reshape(1, -1)
    return model.predict(metrics)[0]

# Initialize MediaPipe Face Detection.
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing Utilities.
mp_drawing = mp.solutions.drawing_utils

# Open webcam.
cap = cv2.VideoCapture(0)

captured_pose = None
training_data = np.zeros((0, 11))
labels = np.zeros((0, 1))
start_time = -1
model = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find faces.
    body_results = pose.process(rgb_frame)

    # Draw face detections on the frame.
    if body_results.pose_landmarks:
        # Draw the current pose with semi-transparency.
        landmarks = body_results.pose_landmarks.landmark

        if not all(landmark.visibility > 0.95 for landmark in landmarks[0:13]):
            print("Warning: Not all face and shoulder landmarks are visible")
        else:
            mp_drawing.draw_landmarks(frame, body_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))
            
        if captured_pose is not None:
            # Draw the captured pose with full opacity.
            mp_drawing.draw_landmarks(frame, captured_pose, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Display the frame.
    cv2.imshow('MediaPipe Face Detection', frame)

    if body_results.pose_landmarks is None:
        continue

    if cv2.waitKey(5) & 0xFF == ord('t'):
        # start training mode. Capture the pose and the metrics
        if (start_time == -1):
            start_time = time.time()
            print("Starting training mode")
            time.sleep(3)

    if start_time != -1:
        if time.time() - start_time < 10:
            print("Capturing good posture")
            training_data = np.append(training_data, [find_metrics(body_results.pose_landmarks.landmark)], axis=0)
            labels = np.append(labels, 0)
        elif time.time() - start_time < 20:
            print("Capturing bad posture")
            training_data = np.append(training_data, [find_metrics(body_results.pose_landmarks.landmark)], axis=0)
            labels = np.append(labels, 1) # bad posture
        else:
            print("Training model")
            print(training_data.shape, labels.shape)
            model = train_random_forest(training_data, labels)
            start_time = -1
            training_data = np.zeros((0, 11))
            labels = np.zeros((0, 1))

    elif cv2.waitKey(5) & 0xFF == ord('c'):
        captured_pose = body_results.pose_landmarks

    if model is not None:
        # predict the pose
        print(predict(model, find_metrics(body_results.pose_landmarks.landmark)))

    # Compare the current pose with the captured pose.
    # if captured_pose is not None and body_results.pose_landmarks is not None:
    #     # TODO: Add your comparison logic here.
    #     # print(captured_pose)
    #     print(body_results.pose_landmarks.landmark[0].x, body_results.pose_landmarks.landmark[0].y, body_results.pose_landmarks.landmark[0].z)
    #     print(find_metrics(body_results.pose_landmarks.landmark))
    #     print(scale_vector(find_metrics(body_results.pose_landmarks.landmark), 100))
    #     pass

    # Break the loop on 'q' key press.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





