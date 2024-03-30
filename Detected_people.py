from flask import Flask, Response
import cv2
from ultralytics import YOLO
import requests

app = Flask(__name__)
camera = cv2.VideoCapture(
    0
)
modelo = YOLO("yolov8n.pt")
server_URL = "http://localhost:8080/pessoa_detectada"


def enviar_pessoa_detectada():
    payload = {"pessoaDetectada": True}
    try:
        response = requests.post(server_URL, json=payload)
        if response.status_code == 200:
            print("Informações enviadas com sucesso para o servidor.")
        else:
            print(
                f"Falha ao enviar informações. Código de resposta: {response.status_code}")
    except Exception as e:
        print(f"Erro ao enviar informações para o servidor: {str(e)}")


def generate_frames():
    while True:
        success, frame = camera.read()
        detec = modelo(frame)
        pessoa_detectada = False
        for objs in detec:
            obj = objs.boxes
            for dados in obj:
                x, y, w, h = dados.xyxy[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                cls = int(dados.cls[0])
                if cls == 0:
                    pessoa_detectada = True
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 5)
        if pessoa_detectada == True:
            enviar_pessoa_detectada()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="192.168.1.5", port=5000, debug=True)
