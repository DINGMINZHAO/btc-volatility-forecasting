from __future__ import annotations

from pathlib import Path

import pandas as pd


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEAR_MONTH = "2022-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sample"
    / f"{SYMBOL}-{INTERVAL}-{YEAR_MONTH}.csv"
)

EXPECTED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def main() -> None:
    print("=" * 60)
    print("Inspect Binance Kline Sample Schema")
    print("=" * 60)
    print(f"CSV path: {CSV_PATH}")
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_PATH}"
        )

    try:
        # The downloaded CSV already contains a header row.
        data = pd.read_csv(CSV_PATH)
    except Exception as error:
        raise RuntimeError(
            f"Failed to read CSV file:\n{CSV_PATH}\n"
            f"Reason: {error}"
        ) from error

    print(f"Number of rows:    {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")

    print()
    print("Detected columns:")

    for index, column in enumerate(data.columns, start=1):
        print(f"{index:2d}. {column}")

    actual_columns = data.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        print()
        print("[Warning] Column names or order differ from expectation.")

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        unexpected_columns = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        print(f"Missing columns:    {missing_columns}")
        print(f"Unexpected columns: {unexpected_columns}")

        raise ValueError(
            "The CSV schema does not match the expected "
            "Binance Kline structure."
        )

    print()
    print("[Schema] PASS: all 12 columns match the expected order.")

    print()
    print("Pandas data types:")
    print(data.dtypes.to_string())

    print()
    print("First two rows:")
    print(data.head(2).to_string(index=False))

    print()
    print("Last two rows:")
    print(data.tail(2).to_string(index=False))

    print()
    print("[Success] Basic schema inspection completed.")


if __name__ == "__main__":
    main()