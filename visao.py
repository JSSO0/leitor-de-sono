import cv2
import dlib
import playsound
import argparse
from scipy.spatial import distance as dist

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")

ap.add_argument("-a", "--alarm", type=str, default="", help="path to alarm .WAV file")

ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")

args = vars(ap.parse_args())

# Inicializar o detector de faces e o detector de pontos de referência faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Definir constantes para o EAR e o número de frames consecutivos em que o EAR deve ser menor que o limite
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Inicializar o contador de frames de boca fechada
COUNTER = 0

# Carregar o arquivo de alarme (se fornecido)
if args["alarm"] != "":
    alarm_path = args["alarm"]

# Iniciar o streaming de vídeo
cap = cv2.VideoCapture(args["webcam"])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionar o frame para que ele tenha uma largura de 450 pixels
    frame = cv2.resize(frame, (450, 300))

    # Converter o frame para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    rects = detector(gray, 0)

    for rect in rects:
        # Determinar os pontos de referência faciais para a região do rosto atual
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # Calcular a razão de aspecto dos olhos (EAR) para ambos os olhos
        leftEAR = eye_aspect_ratio(shape[42:48])
        rightEAR = eye_aspect_ratio(shape[36:42])

        # Calcular o EAR médio
        ear = (leftEAR + rightEAR) / 2.0

        # Desenhar a região dos olhos no frame
        leftEyeHull = cv2.convexHull(shape[36:42])
        rightEyeHull = cv2.convexHull(shape[42:48])
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_path != "":
                    sound_alarm(alarm_path)

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0

    # Mostrar o frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
