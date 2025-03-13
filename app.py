from flask import Flask, request, jsonify, render_template
import os
import cv2
import threading
import time
import face_recognition
from conexion import obtener_conexion

app = Flask(__name__)

UPLOAD_FOLDER = 'Data/Imagenes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ============================
# 📌 CARGAR ROSTROS CONOCIDOS
# ============================
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

# ============================
# 📌 RECONOCIMIENTO FACIAL
# ============================
@app.route('/reconocer', methods=['GET'])
def reconocer_facial():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara"}), 500

    known_face_encodings, known_face_names = load_known_faces()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "No se pudo capturar la imagen"}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if face_encodings:
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"
        match_percentage = 0.0

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            match_percentage = 100 - (face_recognition.face_distance(known_face_encodings, face_encoding)[first_match_index] * 100)

        status = "Autorizado" if match_percentage >= 65 else "Denegado"

        return jsonify({
            "status": status,
            "nombre": name,
            "match_percentage": match_percentage
        })

    return jsonify({"status": "Denegado", "nombre": "Desconocido", "match_percentage": 0.0})

# ============================
# 📌 CRUD PARA USUARIOS
# ============================
@app.route('/usuarios', methods=['POST'])
def agregar_usuario():
    if 'foto' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    foto = request.files['foto']
    nombre = request.form['nombre']

    if foto.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400

    ruta_foto = os.path.join(app.config['UPLOAD_FOLDER'], foto.filename)
    foto.save(ruta_foto)

    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO usuarios (nombre, foto_url) VALUES (%s, %s)", (nombre, ruta_foto))
    conexion.commit()
    cursor.close()
    conexion.close()

    return jsonify({'mensaje': 'Usuario agregado correctamente'})

@app.route('/usuarios', methods=['GET'])
def obtener_usuarios():
    conexion = obtener_conexion()
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarios")
    usuarios = cursor.fetchall()
    cursor.close()
    conexion.close()
    return jsonify(usuarios)

@app.route('/usuarios/<int:id>', methods=['DELETE'])
def eliminar_usuario(id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    cursor.execute("SELECT foto_url FROM usuarios WHERE id = %s", (id,))
    usuario = cursor.fetchone()

    if usuario:
        os.remove(usuario[0])
        cursor.execute("DELETE FROM usuarios WHERE id = %s", (id,))
        conexion.commit()
        mensaje = "Usuario eliminado correctamente"
    else:
        mensaje = "Usuario no encontrado"

    cursor.close()
    conexion.close()
    return jsonify({'mensaje': mensaje})

# ============================
# 📌 SERVIR LA INTERFAZ HTML
# ============================
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
