"""Collect hand-landmark samples for custom gestures from the webcam.

Usage:
    python collect_gestures.py --labels rock,paper,scissors [--samples 200]

Keys:
    1..9   start recording the 1st..9th label (stops after --samples frames)
    SPACE  stop recording
    q/ESC  quit
Samples are appended to data/gestures.csv, so you can run it several times.
"""
import argparse
import csv
from collections import Counter

import cv2

from gesture_common import (DATA_PATH, NUM_FEATURES, create_landmarker, draw_hands,
                            frames, landmarks_to_features, open_camera, put_text)


def load_counts():
    if not DATA_PATH.exists():
        return Counter()
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return Counter(row["label"] for row in csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="comma-separated, max 9")
    parser.add_argument("--samples", type=int, default=200, help="frames per recording")
    parser.add_argument("--every", type=int, default=2, help="keep 1 of every N frames")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not 1 <= len(labels) <= 9:
        raise SystemExit("--labels must contain 1 to 9 names")

    counts = load_counts()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DATA_PATH.exists()

    recording = None   # label currently being recorded
    remaining = 0
    frame_idx = 0

    cap = open_camera(args.camera)
    with create_landmarker(num_hands=1) as landmarker, \
            DATA_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

        for frame, result in frames(cap, landmarker):
            frame_idx += 1
            draw_hands(frame, result)

            if recording and result.hand_landmarks and frame_idx % args.every == 0:
                feats = landmarks_to_features(result.hand_landmarks[0],
                                              result.handedness[0][0].category_name)
                writer.writerow([recording] + [f"{v:.5f}" for v in feats])
                counts[recording] += 1
                remaining -= 1
                if remaining <= 0:
                    recording = None
                    f.flush()

            # UI
            if recording:
                put_text(frame, f"REC [{recording}] {remaining} left", (10, 30), (0, 0, 255))
                if not result.hand_landmarks:
                    put_text(frame, "no hand!", (10, 60), (0, 0, 255))
            else:
                put_text(frame, "Press 1-9 to record, q to quit", (10, 30))
            for i, name in enumerate(labels):
                put_text(frame, f"{i + 1}: {name} ({counts[name]})",
                         (10, 100 + i * 28), (0, 255, 255), 0.6)

            cv2.imshow("Collect gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = None
                f.flush()
            elif ord("1") <= key <= ord(str(len(labels))):
                recording = labels[key - ord("1")]
                remaining = args.samples

    cap.release()
    cv2.destroyAllWindows()
    print("Saved to", DATA_PATH)
    for name, n in sorted(counts.items()):
        print(f"  {name:15s} {n}")


if __name__ == "__main__":
    main()
