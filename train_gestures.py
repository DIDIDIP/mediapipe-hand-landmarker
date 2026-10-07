"""Train a gesture classifier from data/gestures.csv.

Usage:
    python train_gestures.py [--test-size 0.2]
Saves the model to models/gesture_clf.joblib.
"""
import argparse

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_common import CLF_PATH, DATA_PATH


def train(test_size=0.2, data_path=DATA_PATH, clf_path=CLF_PATH):
    """Train, evaluate and save the classifier. Returns a printable report."""
    if not data_path.exists():
        raise ValueError(f"{data_path} not found. Collect some samples first.")
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, encoding="utf-8")
    data = np.atleast_2d(data)
    y = data[:, 0]
    X = data[:, 1:].astype(np.float32)

    labels, counts = np.unique(y, return_counts=True)
    lines = ["Samples per label:"]
    lines += [f"  {name:15s} {n}" for name, n in zip(labels, counts)]
    if len(labels) < 2:
        raise ValueError("Need at least 2 different gestures to train.")
    if counts.min() < 10:
        raise ValueError("Each gesture needs at least 10 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                      early_stopping=True, random_state=42),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    lines.append("\nTest accuracy: {:.3f}".format((y_pred == y_test).mean()))
    lines.append(classification_report(y_test, y_pred, zero_division=0))
    lines.append("Confusion matrix (rows=true, cols=pred): "
                 + str([str(c) for c in clf.classes_]))
    lines.append(str(confusion_matrix(y_test, y_pred, labels=clf.classes_)))

    # Refit on all data for the final model
    clf.fit(X, y)
    clf_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, clf_path)
    lines.append(f"\nSaved model to {clf_path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    try:
        print(train(args.test_size))
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
