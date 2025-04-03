import os
import numpy as np
import cv2
from database import Database

class FaceLogIn:
    def __init__(self, db):  # Cambiar _init_ a __init__
        self.db = db
        self.user_data = []
        self.load_user_data()

    def load_user_data(self):
        users = self.db.get_all_users()  # Assuming this fetches all users from the database
        if not users:
            print("No users found in the database. Continuing without user data.")
            self.user_data = []  # Initialize with an empty list
        else:
            self.user_data = users

    def recognize_user(self, image):
        # Implementar la lógica de reconocimiento de usuario
        pass