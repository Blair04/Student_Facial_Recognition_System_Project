import os
import numpy as np
import cv2
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

embedder = FaceNet()
detector = MTCNN()

X, y = [], []

print("Processing dataset... (This may take a minute)")

for student_dir in os.listdir('dataset'):
    student_path = f'dataset/{student_dir}'
    if not os.path.isdir(student_path): continue
    
    print(f"Training on: {student_dir}")
    for img_name in os.listdir(student_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(f"{student_path}/{img_name}")
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img)

        if results:
            x, y_pos, w, h = results[0]['box']
            x, y_pos = max(0, x), max(0, y_pos)
            face = img[y_pos:y_pos+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160, 160))

            embedding = embedder.embeddings([face])[0]
            X.append(embedding)
            y.append(student_dir)

if len(set(y)) > 1:
    out_encoder = LabelEncoder()
    trainy = out_encoder.fit_transform(y)

    X_arr = np.asarray(X)
    X_norm = X_arr / np.linalg.norm(X_arr, axis=1, keepdims=True)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_norm, trainy)

    centroids = np.zeros((len(out_encoder.classes_), X_norm.shape[1]), dtype='float32')
    for idx in range(len(out_encoder.classes_)):
        mask = trainy == idx
        c = X_norm[mask].mean(axis=0)
        centroids[idx] = c / np.linalg.norm(c)

    with open('student_model.pkl', 'wb') as f:
        pickle.dump((clf, out_encoder, centroids), f)
    print("Success: student_model.pkl created!")
else:
    print("Error: You need at least 2 different people in your dataset folder.")