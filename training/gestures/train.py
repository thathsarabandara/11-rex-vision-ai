"""Gesture classifier training using landmark vectors stored in a dataset CSV.

Format of dataset CSV:
    gesture_name,v0,v1,...,v62

Run:
    python training/gestures/train.py \\
        --data path/to/gestures.csv \\
        --output models/gesture.pt
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train gesture classifier")
    p.add_argument("--data", required=True, help="Path to gesture dataset CSV")
    p.add_argument("--output", default="models/gesture.pt", help="Output model path")
    p.add_argument("--epochs", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import pandas as pd  # type: ignore
        import torch  # type: ignore
        import torch.nn as nn
        from sklearn.preprocessing import LabelEncoder  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    X = df.drop("gesture_name", axis=1).values.astype("float32")
    le = LabelEncoder()
    y = le.fit_transform(df["gesture_name"].values)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    n_classes = len(le.classes_)
    model = nn.Sequential(
        nn.Linear(63, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    torch.save({"model_state": model.state_dict(), "classes": list(le.classes_)}, args.output)
    logger.info(f"Gesture model saved to {args.output}")


if __name__ == "__main__":
    main()
