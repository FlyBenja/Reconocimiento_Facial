from flask import Flask, request, jsonify, render_template
import os
from conexion import obtener_conexion

app = Flask(__name__)

UPLOAD_FOLDER = 'Data/Imagenes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear usuario (subir nombre y foto)
@app.route('/usuarios', methods=['POST'])
def agregar_usuario():
    if 'foto' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    foto = request.files['foto']
    nombre = request.form['nombre']
    
    if foto.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400

    # Guardar la imagen en la carpeta local
    ruta_foto = os.path.join(app.config['UPLOAD_FOLDER'], foto.filename)
    foto.save(ruta_foto)

    # Guardar en la base de datos solo la URL
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO usuarios (nombre, foto_url) VALUES (%s, %s)", (nombre, ruta_foto))
    conexion.commit()
    cursor.close()
    conexion.close()

    return jsonify({'mensaje': 'Usuario agregado correctamente'})

# Obtener todos los usuarios
@app.route('/usuarios', methods=['GET'])
def obtener_usuarios():
    conexion = obtener_conexion()
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarios")
    usuarios = cursor.fetchall()
    cursor.close()
    conexion.close()
    return jsonify(usuarios)

# Eliminar usuario
@app.route('/usuarios/<int:id>', methods=['DELETE'])
def eliminar_usuario(id):
    conexion = obtener_conexion()
    cursor = conexion.cursor()

    # Obtener la foto antes de eliminar el usuario
    cursor.execute("SELECT foto_url FROM usuarios WHERE id = %s", (id,))
    usuario = cursor.fetchone()

    if usuario:
        os.remove(usuario[0])  # Eliminar la imagen del almacenamiento local
        cursor.execute("DELETE FROM usuarios WHERE id = %s", (id,))
        conexion.commit()
        mensaje = "Usuario eliminado correctamente"
    else:
        mensaje = "Usuario no encontrado"

    cursor.close()
    conexion.close()
    return jsonify({'mensaje': mensaje})

# Página principal (interfaz HTML)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
