import cv2
import dlib
import playsound
import argparse
from scipy.spatial import distance as dist
import numpy as np

def sound_alarm(path):
    playsound.playsound(path)
    
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")

ap.add_argument("-a", "--alarm", type=str, default="", help="path to alarm .WAV file")

ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")

args = vars(ap.parse_args())

args["shape_predictor"] = "C:\\Users\\ubots\\Desktop\\leitor-de-sono\\shape_predictor_68_face_landmarks.dat"

args["alarm"] = "C:\\Users\\ubots\\Desktop\\leitor-de-sono\\alarme.wav"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
COUNTER_EYES_CLOSED = 0

if args["alarm"] != "":
    alarm_path = args["alarm"]

cap = cv2.VideoCapture(args["webcam"])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (450, 300))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)

        shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        leftEAR = eye_aspect_ratio(shape[42:48])
        rightEAR = eye_aspect_ratio(shape[36:42])

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(shape[36:42])
        rightEyeHull = cv2.convexHull(shape[42:48])
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER_EYES_CLOSED += 1

            if COUNTER_EYES_CLOSED >= EYE_AR_CONSEC_FRAMES:
                if alarm_path != "":
                    sound_alarm(alarm_path)

                cv2.putText(frame, "ALERTA DE SONO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER_EYES_CLOSED = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
      break

cap.release()
cv2.destroyAllWindows()