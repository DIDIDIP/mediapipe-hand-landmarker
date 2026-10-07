"""Shared helpers for gesture collection / training / inference."""
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from hand_webcam import MODEL_PATH, draw_hands  # noqa: F401 (re-exported)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "gestures.csv"
CLF_PATH = ROOT / "models" / "gesture_clf.joblib"
NUM_FEATURES = 21 * 3


def landmarks_to_features(landmarks, handedness_name):
    """21 landmarks -> 63-dim vector, invariant to position, scale and hand side.

    - translate so the wrist (0) is the origin
    - scale so the farthest point is at distance 1
    - mirror x for left hands so one model works for both hands
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 0:
        pts /= scale
    if handedness_name == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def create_landmarker(num_hands=1):
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}")
    return cap


def frames(cap, landmarker):
    """Yield (mirrored BGR frame, HandLandmarkerResult) from the webcam."""
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - start) * 1000)
        yield frame, landmarker.detect_for_video(mp_image, timestamp_ms)


def put_text(frame, text, org, color=(255, 255, 255), scale=0.7):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
