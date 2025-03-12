import cv2
import requests
import json
import threading
import face_recognition
import os
import numpy as np

# Configuración de la API de Azure
FACE_API_ENDPOINT = "https://visioface.cognitiveservices.azure.com/face/v1.0/detect?returnFaceRectangle=true"
FACE_API_KEY = "6mwoycYwFGNTe7n04RKJaI5ca3hRJcUJWif7Z3uj9IckkxMUShSgJQQJ99BCACYeBjFXJ3w3AAAKACOGR26G"

HEADERS = {
    'Ocp-Apim-Subscription-Key': FACE_API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Cargar imágenes de la carpeta 'Data/Imagenes' y codificarlas
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
                known_face_names.append(filename)  # Usamos el nombre del archivo como el nombre de la persona

    return known_face_encodings, known_face_names

# Función para detectar rostros en un hilo separado
def detect_faces(frame, results, known_face_encodings, known_face_names):
    try:
        # Obtener la codificación de las caras en el frame actual
        rgb_frame = frame[:, :, ::-1]  # Convertir de BGR a RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results.clear()  # Limpiar resultados previos

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Comparar la cara detectada con las caras conocidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            results.append({
                "name": name,
                "location": face_location
            })

    except Exception as e:
        print("Error en la detección de rostros:", e)

def main():
    cap = cv2.VideoCapture(0)
    results = []  # Lista compartida para almacenar los resultados
    thread = None  # Hilo de detección

    # Cargar las imágenes conocidas
    known_face_encodings, known_face_names = load_known_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducir la resolución para mejorar la velocidad
        frame_resized = cv2.resize(frame, (640, 480))

        # Ejecutar la detección en un hilo para evitar bloqueos
        if thread is None or not thread.is_alive():
            thread = threading.Thread(target=detect_faces, args=(frame_resized, results, known_face_encodings, known_face_names))
            thread.start()

        # Dibujar los rectángulos de detección y el nombre de la persona
        for result in results:
            top, right, bottom, left = result["location"]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, result["name"], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
