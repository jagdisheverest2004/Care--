import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def organize_dataset(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    col_filename: str,
    col_label: str,
    use_copy: bool = True,
) -> None:
    df = pd.read_csv(csv_path)

    if col_filename not in df.columns or col_label not in df.columns:
        raise ValueError("CSV missing required columns")

    image_root = Path(image_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    missing = 0
    processed = 0

    for _, row in df.iterrows():
        filename = str(row[col_filename]).strip()
        label = str(row[col_label]).strip()
        if not filename or not label:
            continue

        src = image_root / filename
        if not src.exists():
            missing += 1
            print(f"Missing file: {src}")
            continue

        label_dir = output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        dst = label_dir / src.name

        if use_copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)

        processed += 1

    print(f"Processed: {processed}")
    print(f"Missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to label CSV")
    parser.add_argument("--image_dir", required=True, help="Source image directory")
    parser.add_argument("--output_dir", required=True, help="Destination directory")
    parser.add_argument("--col_filename", required=True, help="CSV column for image filename")
    parser.add_argument("--col_label", required=True, help="CSV column for class label")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy")
    args = parser.parse_args()

    organize_dataset(
        csv_path=args.csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        col_filename=args.col_filename,
        col_label=args.col_label,
        use_copy=not args.move,
    )


if __name__ == "__main__":
    main()
