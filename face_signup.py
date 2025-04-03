import numpy as np
import os
import pickle

class FaceSignUp:
    def __init__(self, db):
        self.db = db

    def save_user_data(self, user_id, name, surname, grade, face_encoding):
        user_data = {
            'user_id': user_id,
            'name': name,
            'surname': surname,
            'grade': grade,
            'face_encoding': face_encoding
        }
        with open(f"{self.db.check_users}/{user_id}.pkl", 'wb') as f:
            pickle.dump(user_data, f)

    def is_user_registered(self, face_encoding):
        for file in os.listdir(self.db.check_users):
            if file.endswith('.pkl'):
                with open(f"{self.db.check_users}/{file}", 'rb') as f:
                    user_data = pickle.load(f)
                    known_encoding = user_data['face_encoding']
                    distance = np.linalg.norm(known_encoding - face_encoding)
                    if distance < 0.35:  # Ajustar el umbral de similitud a 0.45
                        return True, user_data['name']
        return False, None