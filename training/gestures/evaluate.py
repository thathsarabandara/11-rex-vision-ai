"""Evaluate a trained gesture classifier."""
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import torch  # type: ignore
        import torch.nn as nn
        import pandas as pd  # type: ignore
        import numpy as np
        from sklearn.preprocessing import LabelEncoder  # type: ignore
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    checkpoint = torch.load(args.model, map_location="cpu")
    classes = checkpoint["classes"]
    n_classes = len(classes)

    model = nn.Sequential(
        nn.Linear(63, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(args.data)
    X = torch.tensor(df.drop("gesture_name", axis=1).values.astype("float32"))
    le = LabelEncoder().fit(classes)
    y_true = le.transform(df["gesture_name"].values)

    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, preds).tolist()

    results = {"accuracy": round(acc, 4), "per_class": report, "confusion_matrix": cm}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
