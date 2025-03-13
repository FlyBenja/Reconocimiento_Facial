import cv2
import requests
import json
import threading
import face_recognition
import os
import time

# Configuración de la API de Azure
FACE_API_ENDPOINT = "https://visioface.cognitiveservices.azure.com/face/v1.0/detect?returnFaceRectangle=true"
FACE_API_KEY = "6mwoycYwFGNTe7n04RKJaI5ca3hRJcUJWif7Z3uj9IckkxMUShSgJQQJ99BCACYeBjFXJ3w3AAAKACOGR26G"
HEADERS = {
    'Ocp-Apim-Subscription-Key': FACE_API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Cargar imágenes conocidas y codificarlas
def load_known_faces(known_faces_folder="Data/Imagenes"):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])  

    return known_face_encodings, known_face_names

# Función para detectar rostros
def detect_faces(frame, results, known_face_encodings, known_face_names, last_detection):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)  
        response = requests.post(FACE_API_ENDPOINT, headers=HEADERS, data=img_encoded.tobytes())
        
        if response.status_code == 200:
            faces = response.json()
            results.clear()
            for face in faces:
                face_rectangle = face['faceRectangle']
                face_image = frame[face_rectangle['top']:face_rectangle['top'] + face_rectangle['height'],
                                   face_rectangle['left']:face_rectangle['left'] + face_rectangle['width']]
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(rgb_face)

                if face_encoding:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
                    name = "Desconocido"
                    match_percentage = 0.0

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        match_percentage = face_recognition.face_distance(known_face_encodings, face_encoding[0])[first_match_index] * 100
                        match_percentage = 100 - match_percentage  

                    if match_percentage >= 60:
                        results.append({"status": "Autorizado", "color": (0, 255, 0)})
                    else:
                        if last_detection[0] is None:
                            last_detection[0] = time.time()
                        elif time.time() - last_detection[0] >= 3:
                            results.append({"status": "Denegado", "color": (0, 0, 255)})
                            last_detection[0] = None
                        else:
                            results.append({"status": "Escaneando...", "color": (255, 255, 255)})
        else:
            print("Error en la detección:", response.text)
    except Exception as e:
        print("Error en la detección de rostros:", e)

def main():
    cap = cv2.VideoCapture(0)
    results = []  
    thread = None  
    last_detection = [None]  
    
    known_face_encodings, known_face_names = load_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))

        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=detect_faces, args=(frame_resized, results, known_face_encodings, known_face_names, last_detection))
            thread.start()

        for result in results:
            cv2.putText(frame, result["status"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, result["color"], 3)

        cv2.imshow("Detección de Rostros", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
