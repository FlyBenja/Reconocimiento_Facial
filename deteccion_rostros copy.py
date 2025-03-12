import cv2
import requests
import json
import threading

# Configuración de la API de Azure
FACE_API_ENDPOINT = "https://visioface.cognitiveservices.azure.com/face/v1.0/detect?returnFaceRectangle=true"
FACE_API_KEY = "6mwoycYwFGNTe7n04RKJaI5ca3hRJcUJWif7Z3uj9IckkxMUShSgJQQJ99BCACYeBjFXJ3w3AAAKACOGR26G"

HEADERS = {
    'Ocp-Apim-Subscription-Key': FACE_API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Función para detectar rostros en un hilo separado
def detect_faces(frame, results):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)  
        response = requests.post(FACE_API_ENDPOINT, headers=HEADERS, data=img_encoded.tobytes())
        
        if response.status_code == 200:
            faces = response.json()
            results.clear()
            results.extend(faces)  # Almacena los resultados en la lista compartida
        else:
            print("Error en la detección:", response.text)
    except Exception as e:
        print("Error en la detección de rostros:", e)

def main():
    cap = cv2.VideoCapture(0)
    results = []  # Lista compartida para almacenar las caras detectadas
    thread = None  # Hilo de detección

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir la resolución para mejorar la velocidad
        frame_resized = cv2.resize(frame, (640, 480))

        # Ejecutar la detección en un hilo para evitar bloqueos
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=detect_faces, args=(frame_resized, results))
            thread.start()

        # Dibujar los rectángulos de detección
        for face in results:
            rect = face['faceRectangle']
            cv2.rectangle(frame, (rect['left'], rect['top']), 
                          (rect['left'] + rect['width'], rect['top'] + rect['height']), 
                          (0, 255, 0), 2)

        # Mostrar el número de personas detectadas
        cv2.putText(frame, f"Personas detectadas: {len(results)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detección de Rostros", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
