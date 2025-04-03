from flask import Flask, request, jsonify, send_from_directory, session
import os
import mysql.connector
from database import Database
from face_recognition import FaceRecognition 
import base64
import cv2
import numpy as np
from face_signup import FaceSignUp
from face_login import FaceLogIn
import faiss
import multiprocessing as mp
from numba import jit
from skimage.feature import local_binary_pattern

app = Flask(__name__, static_folder="../frontend")
app.secret_key = 'your_secret_key'


db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'acceso_facial'
}

db = Database(db_config)
db.create_table()
face_recognition = FaceRecognition()
face_signup = FaceSignUp(db)
face_login = FaceLogIn(db)

embeddings = db.get_all_embeddings()
for emb in embeddings:
    embedding = np.frombuffer(emb, dtype=np.float64)  
    face_recognition.add_embedding_to_index(embedding)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/models/<path:filename>')
def serve_models(filename):
    return send_from_directory('models', filename)

@app.route('/api/identify', methods=['POST'])
def identify():
    data = request.json
    image_data = data['image_data']
    image_data = base64.b64decode(image_data)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    embedding = face_recognition.get_embedding_from_image(image)
    if embedding is None:
        return jsonify({"status": "failed", "message": "No se pudo obtener el embedding del rostro"})
    distances, indices = face_recognition.search_similar(embedding, k=1)
    if distances[0] < 0.35:
        user_id = db.find_user_by_embedding(embedding)
        db.insert_log(user_id, "success")
        return jsonify({"status": "success", "user_id": user_id})
    else:
        return jsonify({"status": "failed", "message": "Usuario no reconocido"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    user_id = data['user_id']
    name = data['name']
    surname = data['surname']
    grade = data['grade']
    image_data = data['image_data']
    image_data = base64.b64decode(image_data)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    embedding = face_recognition.get_embedding_from_image(image)
    if embedding is None:
        return jsonify({"status": "failed", "message": "No se pudo obtener el embedding del rostro"})
    user_exists = db.user_exists(user_id, embedding)
    if user_exists == "user_id":
        return jsonify({"status": "failed", "message": "El ID de usuario ya existe"})
    elif user_exists:
        matched_user = db.get_user_by_id(user_exists)
        return jsonify({"status": "failed", "message": "El rostro ya está registrado", "matched_user": matched_user})

    try:
        db.insert_embedding(user_id, embedding, name, surname, grade)
        face_recognition.add_embedding_to_index(embedding)
        return jsonify({"status": "success"})
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)})

@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        users = db.get_all_users()
        return jsonify(users)
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = db.get_user_by_id(user_id)
        if user:
            return jsonify(user)
        else:
            return jsonify({"status": "failed", "message": "Usuario no encontrado"}), 404
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.json
    name = data['name']
    surname = data['surname']
    grade = data['grade']
    try:
        db.update_user(user_id, name, surname, grade)
        return jsonify({"status": "success"})
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        db.delete_user(user_id)
        return jsonify({"status": "success"})
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    user_id = request.args.get('user_id')
    try:
        logs = db.get_all_logs(user_id)
        return jsonify(logs)
    except mysql.connector.Error as e:
        return jsonify({"status": "failed", "message": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)