"""Export the trained scikit-learn gesture classifier to JSON for the web demo.

Usage:
    python export_web_model.py
Writes web/gesture_model.json (StandardScaler + MLP weights).
"""
import json

import joblib

from gesture_common import CLF_PATH, ROOT

OUT_PATH = ROOT / "web" / "gesture_model.json"


def main():
    if not CLF_PATH.exists():
        raise SystemExit(f"{CLF_PATH} not found. Train a model first.")
    clf = joblib.load(CLF_PATH)
    scaler, mlp = clf[0], clf[-1]
    model = {
        "classes": [str(c) for c in clf.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,
        "out_activation": mlp.out_activation_,
        "weights": [w.tolist() for w in mlp.coefs_],    # each [n_in][n_out]
        "biases": [b.tolist() for b in mlp.intercepts_],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(model), encoding="utf-8")
    print(f"Saved {OUT_PATH} (classes: {model['classes']})")


if __name__ == "__main__":
    main()
