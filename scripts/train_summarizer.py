import argparse
import pandas as pd

from summarizer import train_summarizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns full_report,summary")
    parser.add_argument("--output_dir", default="t5_medical")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    train_summarizer(df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
