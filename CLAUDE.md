# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Webcam-driven scripts. No package manifest in repo — deps must be installed manually:

```
pip install opencv-python mtcnn keras-facenet scikit-learn numpy streamlit pillow tensorflow
```

- Register a student (CLI, captures 100 frames from webcam): `python register.py`
- Train classifier from `dataset/` → writes `student_model.pkl`: `python train.py`
- Run live detection CLI (press `q` to quit): `python main.py`
- Run Streamlit UI (register + detect + view unknown logs): `streamlit run user_interface.py`

No tests, no lint config.

## Architecture

Three-stage pipeline shared by `main.py` and `user_interface.py`. Read them together — `user_interface.py` reimplements `main.py`'s detection loop inline inside a Streamlit tab and shells out to `train.py` via `subprocess`, so the two detection paths must stay in sync manually.

1. **Enroll** (`register.py` / Register tab) → writes 100 raw JPEGs to `dataset/<Student_Name>/`. Name spaces become underscores; this underscore↔space convention is reversed at display time in detection.
2. **Train** (`train.py`) → for each image: MTCNN crop → resize 160×160 → FaceNet 512-d embedding (raw uint8 input — `keras_facenet` does its own preprocessing internally; do NOT pre-standardize) → L2-normalize → linear `SVC(probability=True)` over `LabelEncoder` targets, plus per-class L2-normalized centroids for cosine-sim unknown gating. Pickled tuple `(clf, out_encoder, centroids)` to `student_model.pkl`. Requires ≥2 distinct students or it aborts without writing.
3. **Detect** (`main.py` / Home tab) → same MTCNN→FaceNet pipeline per frame; `predict_proba` → confidence %, plus cosine sim of L2-normed embedding vs `centroids[best_class]`. Gate: `conf > PROB_THRESHOLD AND sim > SIM_THRESHOLD` else UNKNOWN. Two side effects:
   - conf > 50 → append row to `logs/attendance_<YYYY-MM-DD>.csv`, debounced 300s per name via in-memory `already_logged` dict (lost on restart).
   - conf ≤ 50 → save full BGR frame to `unidentified_logs/alert_<ts>.jpg`, debounced 10s globally via `last_alert_time`.

`facenet_keras.h5` sits at the root but `keras_facenet.FaceNet()` downloads/loads its own weights — the bundled `.h5` is not wired in. `student_model.pkl` is checked in and must be regenerated after any dataset change.

When modifying detection logic, change it in **both** `main.py` and the `tab_home` block of `user_interface.py`; thresholds (`PROB_THRESHOLD`, `SIM_THRESHOLD`) and debounce constants are duplicated.
