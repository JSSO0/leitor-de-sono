import cv2
import dlib
import playsound
import argparse
from scipy.spatial import distance as dist
import numpy as np

def sound_alarm(path):
    playsound.playsound(path)
    
# Função para calcular a razão de aspecto dos olhos (EAR)
def eye_aspect_ratio(eye):
    #A,B,C são variaveis que calculam a distancia entre pontos especificos das coordenadas dos olhos
    A = np.linalg.norm(eye[1] - eye[5])#palpebras
    B = np.linalg.norm(eye[2] - eye[4])#olhos direitos e esquerdos
    C = np.linalg.norm(eye[0] - eye[3])#distancia do canto dos olhos

    ear = (A + B) / (2.0 * C) #media da distancia dos olhos e das palpebras, dividido pela distancia do canto dos olhos
    return ear


ap = argparse.ArgumentParser() #criando um argumento para passar na execução do script

ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor") #definir um argumento, nesse caso o .dat para ler as expressões faciais.

ap.add_argument("-a", "--alarm", type=str, default="", help="path to alarm .WAV file")#definir argumento do alarme

ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")#definir a webcam

args = vars(ap.parse_args())

# Adjust the default values for the arguments based on your file paths
args["shape_predictor"] = "C:\\Users\\ubots\\Desktop\\leitor-de-sono\\shape_predictor_68_face_landmarks.dat" #passar o caminho do shape

args["alarm"] = "C:\\Users\\ubots\\Desktop\\leitor-de-sono\\alarme.wav" #passar o caminho do alarme

# Inicializar o detector de faces e o detector de pontos de referência faciais
detector = dlib.get_frontal_face_detector()#criando o detectador de faces
predictor = dlib.shape_predictor(args["shape_predictor"])#passando o detectador que usará o arquivo importado para ler o rosto.

# Definir constantes para o EAR e o número de frames consecutivos em que o EAR deve ser menor que o limite
EYE_AR_THRESH = 0.3#indice de fechamento dos olhos para poder acionar o alarme
EYE_AR_CONSEC_FRAMES = 48#contador em frames que o código espera para acionar o alarme se o olho estiver fechado

# Inicializar o contador de frames de boca fechada e de olhos fechados
COUNTER = 0
COUNTER_EYES_CLOSED = 0

# Carregar o arquivo de alarme (se fornecido)
if args["alarm"] != "":
    alarm_path = args["alarm"]

# Iniciar o streaming de vídeo
cap = cv2.VideoCapture(args["webcam"])

while True:
    ret, frame = cap.read()#ele le o quadro que retorna o que foi lido pela webcam

    if not ret:#se tiver vazio ele para
        break

    # Redimensionar o frame para que ele tenha uma largura de 450 pixels
    frame = cv2.resize(frame, (450, 300))#tamanho da janela

    # Converter o frame para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#deixa o quadro meio cinza para ajudar na leitura ser mais leve

    # Detectar faces no frame
    rects = detector(gray, 0)#usando o detector para retornar na imagem a marcação nos olhos

    for rect in rects:
        # Determinar os pontos de referência faciais para a região do rosto atual
        shape = predictor(gray, rect)

        # Converter os pontos de referência faciais em um formato NumPy
        shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

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

        if ear < EYE_AR_THRESH: #se os olhos estão fechados menor que o limite estabelecido ent está fechado
            COUNTER_EYES_CLOSED += 1 # se o olhos estão fechados adicionamos mais um frame

            if COUNTER_EYES_CLOSED >= EYE_AR_CONSEC_FRAMES: #verificamos se o numero de frames que os olhos estão fechados é igual ou maior que o limite estabelecido
                if alarm_path != "": #acionamos o alarme
                    sound_alarm(alarm_path)

                cv2.putText(frame, "ALERTA DE SONO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#e adicionamos o alerta na tela.

        else:
            COUNTER_EYES_CLOSED = 0 #senaõ for satisfeita, reiniciamos o contador.

    # Mostrar o frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
      break

cap.release()
cv2.destroyAllWindows()