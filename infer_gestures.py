"""Real-time custom gesture recognition with the trained classifier.

Usage:
    python infer_gestures.py [--threshold 0.7] [--hands 2]
Press 'q' or ESC to quit.
"""
import argparse
from collections import Counter, deque

import cv2
import joblib

from gesture_common import (CLF_PATH, create_landmarker, draw_hands, frames,
                            landmarks_to_features, open_camera, put_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="below this probability the gesture is shown as '?'")
    parser.add_argument("--smooth", type=int, default=5, help="majority vote over N frames")
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not CLF_PATH.exists():
        raise SystemExit(f"{CLF_PATH} not found. Run train_gestures.py first.")
    clf = joblib.load(CLF_PATH)
    history = {}  # handedness -> recent predictions

    cap = open_camera(args.camera)
    with create_landmarker(num_hands=args.hands) as landmarker:
        for frame, result in frames(cap, landmarker):
            draw_hands(frame, result)
            h, w = frame.shape[:2]

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                side = handedness[0].category_name
                feats = landmarks_to_features(landmarks, side)
                proba = clf.predict_proba([feats])[0]
                best = proba.argmax()
                name = clf.classes_[best] if proba[best] >= args.threshold else "?"

                hist = history.setdefault(side, deque(maxlen=args.smooth))
                hist.append(name)
                voted = Counter(hist).most_common(1)[0][0]

                x = int(min(lm.x for lm in landmarks) * w)
                y = int(max(lm.y for lm in landmarks) * h) + 30
                put_text(frame, f"{voted} ({proba[best]:.2f})", (x, min(y, h - 10)),
                         (0, 255, 0), 1.0)

            cv2.imshow("Custom gesture recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
