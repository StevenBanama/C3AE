import os
import cv2
import numpy as np
import math
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector

def quaterniondToRulerAngle(quaterniond):
    q = quaterniond
    y_sqrt = q.y ** 2
    # pitch
    t0 = 2 * (q.w * q.x + q.y * q.z)
    t1 = 1.0 - 2.0 * (q.x ** 2 + y_sqrt) 
    pitch = math.atan(t0 / t1)#math.atan2(t0, t1)

    # yaw
    t2 = 2 * (q.w * q.y - q.z * q.x)
    t2 = max(min(t2, 1), -1)
    yaw = math.asin(t2)

    # roll
    t3 = 2 * (q.w * q.z + q.x * q.y)
    t4 = 1 - 2 * (y_sqrt + q.z * q.z)
    roll = math.atan(t3 / t4)  #math.atan2(t3, t4)
    return pitch, yaw, roll

def tran_euler(rotation_vect):
    theta = cv2.norm(rotation_vect, cv2.NORM_L2)
    class Quation(object):
        def __init__(self, w, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
    quat = Quation(
        math.cos(theta / 2),
        math.sin(theta / 2) * rotation_vect[0][0] / theta,
        math.sin(theta / 2) * rotation_vect[1][0] / theta,
        math.sin(theta / 2) * rotation_vect[2][0] / theta
        )
    return map(lambda x: x / math.pi * 180, quaterniondToRulerAngle(quat))

def trans_landmarks(img, landmark_groups):
    result = []
    for lm in landmark_groups:
        landmarks = np.array([(lm[x], lm[5 + x],) for x in range(5)], dtype="double")
        for p in landmarks:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        result.append(get_rotation_angle(img, landmarks))
    return result

def get_rotation_angle(img, landmarks, draw=False):
    # get four stander point for pnp
    landmarks = np.array([(landmarks[x], landmarks[5 + x],) for x in range(5)], dtype="double") 
    landmarks = landmarks.astype(np.float32)
    landmarks[1] = (landmarks[0] + landmarks[1]) / 2.0
    landmarks = landmarks[1:]
    size = img.shape
    model_points = np.array([
        #(-225.0, 170, -135),  #left eye
        #(225.0, 170, -135.0),
        (0.0, 170, -135.0),
        (0.0, -0.0, 0.0),   # nose
        #(0, -150, -125),
        (-150.0, -150, -125), #left mouth corner
        (150.0, -150, -125),
    ], dtype=np.float32)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2,)
    camera_matrix = np.array([
             [focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]
         ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, trans_vector) = cv2.solvePnP(model_points, landmarks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    f_pitch, f_yaw, f_roll = tran_euler(rotation_vector)

    n_pitch = prod_trans_point((0, 0, 500.0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_yaw = prod_trans_point((200.0, 0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    n_roll = prod_trans_point((0, 500.0, 0), rotation_vector, trans_vector, camera_matrix, dist_coeffs)

    if draw:
        p1 = (int(landmarks[1][0]), int(landmarks[1][1]))
        cv2.line(img, p1, n_roll, (255, 0, 0), 2)
        cv2.line(img, p1, n_yaw, (0, 255, 0), 2)
        cv2.line(img, p1, n_pitch, (0, 0, 255), 2)
        cv2.putText(img, ("r:" + str(f_roll))[:6], n_roll, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("y:" + str(f_yaw))[:6], n_yaw, 1,1, (136, 97, 45), 2) 
        cv2.putText(img, ("p:" + str(f_pitch))[:6], n_pitch, 1,1, (136, 97, 45), 2) 
    return f_pitch, f_yaw, f_roll

def prod_trans_point(p3d, rotation_vector, trans_vector, camera_matrix, dist_coeffs):
    plane_point, _ = cv2.projectPoints(np.array([p3d]), rotation_vector, trans_vector, camera_matrix, dist_coeffs)
    return (int(plane_point[0][0][0]), int(plane_point[0][0][1]))

