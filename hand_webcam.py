"""MediaPipe Hand Landmarker - webcam demo.

Usage:
    python hand_webcam.py [--camera 0] [--hands 2]
Press 'q' or ESC to quit.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # Label (Left/Right + score) above the wrist-to-fingers bounding box
        xs, ys = zip(*points)
        label = f"{handedness[0].category_name} {handedness[0].score:.2f}"
        cv2.putText(frame, label, (min(xs), max(min(ys) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    args = parser.parse_args()

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("MediaPipe Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
