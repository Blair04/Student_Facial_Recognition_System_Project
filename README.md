# Student Facial Recognition System

Real-time face recognition for student attendance and unknown-person alerting. Uses MTCNN for face detection, FaceNet for 512-d embeddings, and a linear SVM classifier with cosine-similarity gating to reject strangers.

## Pipeline

```
[Webcam frame]
      |
      v
   MTCNN ----> face box (clamped)
      |
      v
  resize 160x160 + per-image standardization
      |
      v
    FaceNet ----> 512-d embedding (L2-normalized)
      |
      v
  Linear SVC.predict_proba  +  cosine sim vs class centroid
      |
      v
  conf > 70% AND sim > 0.5  ?
      |                |
     yes              no
      |                |
  log attendance   save frame to unidentified_logs/
```

## Features

- Webcam enrollment (100 frames per student)
- SVM classifier over FaceNet embeddings
- **Unknown rejection** via cosine similarity to per-class centroids (closed-set SVM alone cannot reject)
- Daily attendance CSV with 5-minute per-name debounce
- Unknown-face snapshots with 10-second global debounce
- Streamlit UI: Home (live detect), Register, Tagged Unknown gallery
- CLI alternatives for register / train / detect

## Requirements

Python 3.9+ and a webcam. No `requirements.txt` in repo — install manually:

```
pip install opencv-python mtcnn keras-facenet scikit-learn numpy streamlit pillow tensorflow
```

`keras-facenet` downloads its own FaceNet weights on first run; the bundled `facenet_keras.h5` is **not** wired in.

## Usage

### Streamlit UI (recommended)

```
streamlit run user_interface.py
```

Tabs:
- **Home** — toggle camera, live detection
- **Register Student** — enter name, capture 100 frames, click Re-Train
- **Tagged Unknown** — gallery of saved unknown snapshots

### CLI

```
python register.py     # capture 100 frames; press SPACE to start, q to quit
python train.py        # rebuild student_model.pkl from dataset/
python main.py         # live detection window; press q to quit
```

## Folder layout

```
.
├── dataset/<Student_Name>/*.jpg     # raw enrollment frames (spaces -> underscores)
├── logs/attendance_YYYY-MM-DD.csv   # daily attendance
├── unidentified_logs/alert_*.jpg    # unknown-face snapshots
├── student_model.pkl                # (clf, label_encoder, centroids)
├── register.py
├── train.py
├── main.py
└── user_interface.py
```

`student_model.pkl` is checked in but **must be regenerated whenever `dataset/` changes**. Train requires ≥ 2 distinct students or it aborts without writing.

## Tuning

Both detection paths (`main.py`, `user_interface.py` Home tab) duplicate two thresholds:

| Constant          | Default | Meaning                                              |
|-------------------|---------|------------------------------------------------------|
| `PROB_THRESHOLD`  | 70      | SVM confidence floor (%)                             |
| `SIM_THRESHOLD`   | 0.5     | Cosine sim to nearest class centroid (unknown gate)  |

Lower `SIM_THRESHOLD` → more permissive (more known matches, more false accepts). Raise it → stricter unknown rejection. Tune both together.

Other constants in detection loops:
- attendance debounce: 300 s per name
- unknown-snapshot debounce: 10 s global

## Known limits

- `already_logged` is in-memory; restarting the script resets the 5-minute debounce.
- Detection logic is duplicated between `main.py` and `user_interface.py`'s `tab_home`. Keep thresholds and standardization in sync manually.
- Backwards-compat: an old `student_model.pkl` without centroids loads with a warning and skips the unknown gate. Retrain to enable it.
- Single webcam at index 0; no multi-camera support.

## Troubleshooting

- **`student_model.pkl` not found** → run `python train.py` after registering ≥ 2 students.
- **Strangers shown as a known name** → centroids missing from old pkl, or `SIM_THRESHOLD` too low. Retrain, then raise threshold.
- **Webcam window black / fails to open** → another app holds the camera, or index 0 is wrong; change `cv2.VideoCapture(0)`.
- **Streamlit "Re-Train" hangs** → `subprocess.run(["python", "train.py"])` runs synchronously; check the terminal for progress.
