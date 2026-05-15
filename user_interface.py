import streamlit as st
import cv2
import os
import time
import pickle
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image

os.makedirs('unidentified_logs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

st.set_page_config(page_title="Student Security System", layout="wide")

@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

detector, embedder = load_models()

PROB_THRESHOLD = 60
SIM_THRESHOLD = 0.6
DEBUG = True

def load_trained_model():
    if os.path.exists('student_model.pkl'):
        with open('student_model.pkl', 'rb') as f:
            loaded = pickle.load(f)
        if len(loaded) == 3:
            return loaded
        clf, out_encoder = loaded
        return clf, out_encoder, None
    return None, None, None

st.title("Student Facial Recognition System")

tab_home, tab_register, tab_logs = st.tabs([
    "🏠 Home (Detection)", 
    "📝 Register Student", 
    "📂 Tagged Unknown"
])
#home tab
with tab_home:
    st.header("Real-time Monitor")
    clf, out_encoder, centroids = load_trained_model()

    if clf is None:
        st.warning("No model found. Please go to the 'Register Student' tab first.")
    else:
        notification_placeholder = st.empty()
        
        run_cam = st.toggle("Activate Security Camera")
        FRAME_WINDOW = st.image([])
        
        already_logged = {}
        last_alert_time = 0

        cap = cv2.VideoCapture(0)
        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            
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
                        already_logged[name] = time.time()
                        st.toast(f"Attendance recorded for {name}")
                        notification_placeholder.empty() 
                else:
                    color, label = (255, 0, 0), "UNKNOWN ALERT"
                    
                    #notif
                    st.toast("⚠️ ALERT: Unknown Person Detected!", icon="🚨")
                    notification_placeholder.error(f"🚨 UNKNOWN PERSON DETECTED AT {datetime.now().strftime('%I:%M:%S %p')}")
                    
                    with st.sidebar:
                        st.error("🚨 CRITICAL ALERT: Unidentified face active in feed.")
                    
                    if time.time() - last_alert_time > 10:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"unidentified_logs/alert_{ts}.jpg", frame)
                        last_alert_time = time.time()

                cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), color, 4)
                cv2.putText(rgb_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            FRAME_WINDOW.image(rgb_frame)
        cap.release()

# register tab
with tab_register:
    st.header("Student Enrollment")
    
    with st.form("registration_form"):
        new_name = st.text_input("Enter Full Name")
        submit_btn = st.form_submit_button("Start Enrollment Process")

    if submit_btn and new_name:
        folder_name = new_name.replace(" ", "_")
        path = f'dataset/{folder_name}'
        os.makedirs(path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        progress = st.progress(0)
        preview = st.image([])
        
        for count in range(100): # Capturing 100 images as per register.py
            ret, frame = cap.read()
            if not ret: break
            
            cv2.imwrite(f"{path}/{count}.jpg", frame)
            progress.progress((count + 1) / 100)
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=400)
            time.sleep(0.05)
            
        cap.release()
        st.success(f"Images saved for {new_name}. Now click below to train.")

    if st.button("🚀 Re-Train Model"):
        with st.spinner("Analyzing dataset... (This matches train.py logic)"):
            import subprocess
            subprocess.run(["python", "train.py"])
            st.success("System updated with new student data!")

# tagged unknown log
with tab_logs:
    st.header("Intruder / Unknown Logs")
    
    files = [f for f in os.listdir('unidentified_logs') if f.endswith('.jpg')]
    if files:
        files.sort(reverse=True)
        cols = st.columns(4)
        for idx, file in enumerate(files):
            with cols[idx % 4]:
                img_path = os.path.join('unidentified_logs', file)
                timestamp = file.replace("alert_", "").replace(".jpg", "")
                st.image(img_path, caption=f"Detected: {timestamp}")
    else:
        st.info("No unknown persons have been logged yet.")