from enum import Enum
import math
import numpy as np

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
    left_shoulder_x = landmark[BodyParts.LEFT_SHOULDER.value].x
    left_shoulder_y = landmark[BodyParts.LEFT_SHOULDER.value].y
    right_shoulder_x = landmark[BodyParts.RIGHT_SHOULDER.value].x
    right_shoulder_y = landmark[BodyParts.RIGHT_SHOULDER.value].y

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