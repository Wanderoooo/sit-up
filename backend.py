# app.py
from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2
import math
import numpy as np
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import cv2
from model import predict, train_random_forest
from label_process import find_metrics
from esp_comm import send_buzz
from database import get_records, add_record, add_record_record

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE"], "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"], "expose_headers": ["X-Custom-Header"], "supports_credentials": True}})

def generate_frames():
    global training_data, labels, start_time, model, current_labels, valid_landmark_state

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
                valid_landmark_state = False
            else:
                valid_landmark_state = True
                mp_drawing.draw_landmarks(frame, body_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Display the frame.
        # cv2.imshow('MediaPipe Face Detection', frame)

        # if body_results.pose_landmarks is None:
        #     continue

        if body_results.pose_landmarks is not None:
            current_labels = find_metrics(body_results.pose_landmarks.landmark)
            pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/record_good_posture')
def record_good_posture():
    global training_data, labels, current_labels, valid_landmark_state
    if (not valid_landmark_state):
        return "Good posture not recorded"
    
    training_data = np.append(training_data, [current_labels], axis=0)
    labels = np.append(labels, [0])
    return "Good posture recorded"

@app.route('/record_bad_posture')
def record_bad_posture():
    global training_data, labels, current_labels, valid_landmark_state
    if (not valid_landmark_state):
        return "Bad posture not recorded"

    training_data = np.append(training_data, [current_labels], axis=0)
    labels = np.append(labels, [1])
    return "Bad posture recorded"

@app.route('/get_training_data_size')
def get_training_data_size():
    global training_data
    return str(training_data.shape[0])

@app.route('/train_model')
def train_model():
    global model, training_data, labels
    model = train_random_forest(training_data, labels)
    return "Model trained"

@app.route('/clear_training_data')
def clear_training_data():
    global training_data, labels
    training_data = np.zeros((0, 11))
    labels = np.zeros((0, 1))
    return "Training data cleared"

@app.route('/clear_model')
def clear_model():
    global model
    model = None

@app.route('/send_buzz')
def send_buzz():
    send_buzz()
    return "Buzz sent"

@app.route('/predict')
def predict():
    global model
    if model is None:
        return jsonify({"posture":-1})
    else:
        return jsonify({"posture": predict(model, current_labels)})
    
@app.route('/get_current_state')
def get_current_state():
    global valid_landmark_state
    return jsonify({"valid_landmark_state": valid_landmark_state})

@app.route('/get_records_data')
def get_time_chart_data():
    records = get_records()
    print(records)
    return jsonify(records)

@app.route('/add_record')
def add_record():
    date = request.args.get('date')
    model = request.args.get('model')
    start = request.args.get('start')
    end = request.args.get('end')
    percentage = request.args.get('percentage')
    add_record(date, model, start, end, percentage)
    return "Record added"


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    # Initialize MediaPipe Pose.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize MediaPipe Drawing Utilities.
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    training_data = np.zeros((0, 11))
    labels = np.zeros((0, 1))
    model = None
    current_labels = None
    valid_landmark_state = False

    app.run(debug=True)