import faiss
import numpy as np
import cv2
from mtcnn import MTCNN
import dlib
from multiprocessing import Pool, cpu_count

class FaceRecognition:
    def __init__(self):
        # Inicializar MTCNN para la detección de rostros
        self.detector = MTCNN()
        # Inicializar el modelo de Dlib para la extracción de embeddings
        self.face_rec_model = dlib.face_recognition_model_v1("backend/models/dlib_face_recognition_resnet_model_v1.dat")
        # Inicializar el predictor de puntos faciales de Dlib
        self.predictor = dlib.shape_predictor("backend/models/shape_predictor_68_face_landmarks.dat")
        # Inicializar FAISS para la búsqueda de similitudes
        self.index = faiss.IndexFlatL2(128)

    def preprocess_image(self, image):
        # Convertir la imagen a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_embedding_from_image(self, image):
        image = self.preprocess_image(image)
        # Detectar rostros en la imagen
        detections = self.detector.detect_faces(image)
        if len(detections) == 0:
            return None
        # Tomar el primer rostro detectado
        detection = detections[0]
        x, y, width, height = detection['box']
        # Extraer el rostro
        face = image[y:y+height, x:x+width]
        # Detectar puntos faciales
        rect = dlib.rectangle(x, y, x + width, y + height)
        shape = self.predictor(image, rect)
        # Extraer el embedding facial
        embedding = self.face_rec_model.compute_face_descriptor(image, shape)
        # Convertir a numpy array y normalizar
        embedding = np.array(embedding, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        embedding = embedding / norm
        return embedding

    def add_embedding_to_index(self, embedding):
        if embedding.shape != (128,):
            return
        self.index.add(np.array([embedding], dtype=np.float64))

    def search_similar(self, embedding, k=1):
        distances, indices = self.index.search(np.array([embedding], dtype=np.float64), k)
        return distances[0], indices[0]

def process_image(image_data):
    face_recognition = FaceRecognition()
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    embedding = face_recognition.get_embedding_from_image(image)
    return embedding

if __name__ == '__main__':
    # Ejemplo de uso de multiprocessing para procesar imágenes en paralelo
    image_data_list = [...]  # Lista de datos de imágenes en formato base64
    with Pool(cpu_count()) as pool:
        embeddings = pool.map(process_image, image_data_list)
    # Agregar embeddings al índice FAISS
    face_recognition = FaceRecognition()
    for embedding in embeddings:
        if embedding is not None:
            face_recognition.add_embedding_to_index(embedding)