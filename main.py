import cv2
import time
import csv
import pickle
import os
from datetime import datetime
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np

os.makedirs('unidentified_logs', exist_ok=True)
os.makedirs('logs', exist_ok=True)
detector = MTCNN()
embedder = FaceNet()

with open('student_model.pkl', 'rb') as f:
    loaded = pickle.load(f)
    if len(loaded) == 3:
        clf, out_encoder, centroids = loaded
    else:
        clf, out_encoder = loaded
        centroids = None
        print("WARNING: model has no centroids — retrain to enable unknown detection.")

PROB_THRESHOLD = 60
SIM_THRESHOLD = 0.6
DEBUG = True

already_logged = {} 
last_alert_time = 0

def log_attendance(name):
    filename = f"logs/attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['Name', 'Time'])
        writer.writerow([name, datetime.now().strftime('%H:%M:%S')])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        if w <= 0 or h <= 0:
            continue
        face = rgb_frame[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face = cv2.resize(face, (160, 160))

        embedding = embedder.embeddings([face])[0]
        emb_norm = embedding / np.linalg.norm(embedding)

        probs = clf.predict_proba([emb_norm])
        best_class = np.argmax(probs)
        conf = probs[0][best_class] * 100

        sim = float(np.dot(centroids[best_class], emb_norm)) if centroids is not None else 1.0

        if DEBUG:
            cls_name = out_encoder.inverse_transform([best_class])[0]
            print(f"[debug] best={cls_name} conf={conf:.1f}% sim={sim:.3f}")

        if conf > PROB_THRESHOLD and sim > SIM_THRESHOLD:
            name = out_encoder.inverse_transform([best_class])[0].replace("_", " ")
            color, label = (0, 255, 0), f"{name} ({conf:.1f}%)"
            if name not in already_logged or (time.time() - already_logged[name] > 300):
                log_attendance(name)
                already_logged[name] = time.time()
        else:
            color, label = (0, 0, 255), "UNKNOWN ALERT"
            if time.time() - last_alert_time > 10:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"unidentified_logs/alert_{ts}.jpg", frame)
                last_alert_time = time.time()

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Security Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()