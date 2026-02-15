import argparse
from pathlib import Path

from vision_model import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to training directory")
    parser.add_argument("--val_dir", required=True, help="Path to validation directory")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
