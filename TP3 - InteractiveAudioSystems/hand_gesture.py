import cv2
import mediapipe as mp
import socket
import math

#UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

def identificar_gesto(landmarks, hand_label):
    dedos = []
    if landmarks[8].y < landmarks[6].y:
        dedos.append("indicador")
    if landmarks[12].y < landmarks[10].y:
        dedos.append("medio")
    if landmarks[16].y < landmarks[14].y:
        dedos.append("anelar")
    if landmarks[20].y < landmarks[18].y:
        dedos.append("mindinho")

    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            dedos.append("polegar")
    else:
        if landmarks[4].x > landmarks[3].x:
            dedos.append("polegar")

    return "-".join(sorted(dedos)) if dedos else "nenhum"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label = hand_info.classification[0].label  #"Left" ou "Right"
            gesto = identificar_gesto(hand_landmarks.landmark, label)

            mensagem = f"{label.lower()}:{gesto}"

            if "indicador" in gesto:
                dedo = hand_landmarks.landmark[8]
                x = round(dedo.x, 4)
                y = round(dedo.y, 4)

                mensagem += f";{x};{y}"

                if "polegar" in gesto:
                    polegar = hand_landmarks.landmark[4]
                    dist = math.sqrt((polegar.x - dedo.x) ** 2 + (polegar.y - dedo.y) ** 2)
                    mensagem += f";pinch={round(dist, 4)}"
                    
            print("Enviado:", mensagem)
            sock.sendto(mensagem.encode(), (UDP_IP, UDP_PORT))
    cv2.imshow("Maos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()