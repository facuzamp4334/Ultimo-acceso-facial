import mysql.connector
import numpy as np
import time
from contextlib import contextmanager

class Database:
    def __init__(self, db_config):
        self.db_config = db_config

    @contextmanager
    def get_connection(self):
        conn = mysql.connector.connect(**self.db_config)
        try:
            yield conn
        finally:
            conn.close()

    def create_table(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    user_id INT PRIMARY KEY,
                    embedding BLOB,
                    name VARCHAR(255),
                    surname VARCHAR(255),
                    grade VARCHAR(255)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    log_id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(255),
                    FOREIGN KEY(user_id) REFERENCES embeddings(user_id)
                )
            ''')
            conn.commit()

    def check_users(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM embeddings LIMIT 1')
            return cursor.fetchone() is not None

    def get_user_by_id(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT user_id, name, surname, grade FROM embeddings WHERE user_id = %s', (user_id,))
            user = cursor.fetchone()
        if user:
            return user
        return None
    
    def user_exists(self, user_id, embedding):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Verificar si el user_id ya existe
            cursor.execute('SELECT 1 FROM embeddings WHERE user_id = %s', (user_id,))
            if cursor.fetchone():
                return "user_id"

            # Obtener todos los embeddings almacenados junto con sus IDs
            cursor.execute('SELECT user_id, embedding FROM embeddings')
            embeddings = cursor.fetchall()
            min_distance = float('inf')
            matched_user_id = None
            for user_id_db, emb in embeddings:
                # Convertir el embedding almacenado de BLOB a numpy array
                known_embedding = np.frombuffer(emb, dtype=np.float64)

                # Verificar que las dimensiones coincidan
                if known_embedding.shape != (128,):
                    continue

                # Calcular la distancia euclidiana entre los embeddings
                distance = np.linalg.norm(known_embedding - embedding)
                print(f"Distancia entre embeddings: {distance}")
                # Ajustar el umbral de similitud a 0.40
                if distance < 0.35 and distance < min_distance:
                    min_distance = distance
                    matched_user_id = user_id_db

            if matched_user_id is not None:
                return matched_user_id

        return None

    def find_user_by_embedding(self, embedding):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, embedding FROM embeddings')
            embeddings = cursor.fetchall()
            min_distance = float('inf')
            matched_user_id = None
            for user_id_db, emb in embeddings:
                known_embedding = np.frombuffer(emb, dtype=np.float64)
                if known_embedding.shape != (128,):
                    continue
                distance = np.linalg.norm(known_embedding - embedding)
                if distance < 0.35 and distance < min_distance:
                    min_distance = distance
                    matched_user_id = user_id_db
            return matched_user_id
    
    def insert_embedding(self, user_id, embedding, name, surname, grade):
        retries = 5
        while retries > 0:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO embeddings (user_id, embedding, name, surname, grade) VALUES (%s, %s, %s, %s, %s)
                    ''', (user_id, embedding.tobytes(), name, surname, grade))
                    conn.commit()
                break
            except mysql.connector.Error as e:
                if e.errno == mysql.connector.errorcode.ER_LOCK_WAIT_TIMEOUT:
                    retries -= 1
                    time.sleep(1)
                else:
                    raise

    def get_all_embeddings(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT embedding FROM embeddings')
            embeddings = cursor.fetchall()
        return [np.frombuffer(emb[0], dtype=np.float64) for emb in embeddings]

    def get_all_users(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, name, surname, grade FROM embeddings')
            users = cursor.fetchall()
        return [{"user_id": user[0], "name": user[1], "surname": user[2], "grade": user[3]} for user in users]

    def insert_log(self, user_id, status):
        retries = 5
        while retries > 0:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO logs (user_id, status) VALUES (%s, %s)
                    ''', (user_id, status))
                    conn.commit()
                break
            except mysql.connector.Error as e:
                if e.errno == mysql.connector.errorcode.ER_LOCK_WAIT_TIMEOUT:
                    retries -= 1
                    time.sleep(1)
                else:
                    raise

    def update_user(self, user_id, name, surname, grade):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE embeddings
                SET name = %s, surname = %s, grade = %s
                WHERE user_id = %s
            ''', (name, surname, grade, user_id))
            conn.commit()

    def delete_user(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Eliminar los logs asociados al usuario
            cursor.execute('DELETE FROM logs WHERE user_id = %s', (user_id,))
            # Eliminar el usuario de la tabla embeddings
            cursor.execute('DELETE FROM embeddings WHERE user_id = %s', (user_id,))
            conn.commit()

    def get_all_logs(self, user_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM logs WHERE user_id = %s', (user_id,))
            logs = cursor.fetchall()
        return [{"log_id": log[0], "user_id": log[1], "timestamp": log[2], "status": log[3]} for log in logs]