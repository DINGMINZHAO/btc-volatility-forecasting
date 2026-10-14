from __future__ import annotations

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

START_MONTH = "2022-01"
END_MONTH = "2025-12"

EXPECTED_START = pd.Timestamp(
    "2022-01-01 00:00:00",
    tz="UTC",
)

EXPECTED_END_EXCLUSIVE = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
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

COLUMN_DTYPES = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_volume": "float64",
    "count": "int64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "ignore": "int64",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "binance"
    / "futures"
    / "um"
    / "monthly"
    / "klines"
    / SYMBOL
    / INTERVAL
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_2022_2025.parquet"
)


def generate_months(
    start_month: str,
    end_month: str,
) -> list[str]:
    """
    Generate all YYYY-MM strings between two months,
    including both endpoints.
    """
    start_year, start_month_number = map(
        int,
        start_month.split("-"),
    )

    end_year, end_month_number = map(
        int,
        end_month.split("-"),
    )

    months: list[str] = []

    year = start_year
    month = start_month_number

    while (year, month) <= (end_year, end_month_number):
        months.append(f"{year:04d}-{month:02d}")

        month += 1

        if month == 13:
            month = 1
            year += 1

    return months


def read_one_month(
    csv_path: Path,
    year_month: str,
) -> pd.DataFrame:
    """
    Read one monthly CSV and verify its schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Monthly CSV not found:\n{csv_path}"
        )

    data = pd.read_csv(
        csv_path,
        dtype=COLUMN_DTYPES,
    )

    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column schema mismatch for {year_month}."
        )

    data["source_month"] = year_month

    return data


def validate_complete_timeline(
    data: pd.DataFrame,
) -> None:
    """
    Validate the complete five-minute timeline across all months.
    """
    expected_index = pd.date_range(
        start=EXPECTED_START,
        end=EXPECTED_END_EXCLUSIVE,
        freq="5min",
        inclusive="left",
    )

    actual_index = pd.DatetimeIndex(
        data["open_datetime_utc"]
    )

    duplicate_count = int(
        data["open_time"].duplicated().sum()
    )

    missing_times = expected_index.difference(
        actual_index
    )

    unexpected_times = actual_index.difference(
        expected_index
    )

    adjacent_differences = (
        data["open_time"]
        .diff()
        .dropna()
    )

    irregular_gap_count = int(
        (adjacent_differences != 300_000).sum()
    )

    expected_last_open = (
        EXPECTED_END_EXCLUSIVE
        - pd.Timedelta(minutes=5)
    )

    print()
    print("Global timeline validation:")
    print(f"Expected rows:          {len(expected_index)}")
    print(f"Actual rows:            {len(data)}")
    print(f"Duplicate open times:   {duplicate_count}")
    print(f"Missing timestamps:     {len(missing_times)}")
    print(f"Unexpected timestamps:  {len(unexpected_times)}")
    print(f"Irregular gaps:         {irregular_gap_count}")
    print(
        f"First open time:        "
        f"{data['open_datetime_utc'].iloc[0]}"
    )
    print(
        f"Last open time:         "
        f"{data['open_datetime_utc'].iloc[-1]}"
    )

    issues: list[str] = []

    if len(data) != len(expected_index):
        issues.append(
            "The total number of rows is incorrect."
        )

    if duplicate_count > 0:
        issues.append(
            "Duplicate open timestamps were detected."
        )

    if len(missing_times) > 0:
        issues.append(
            "Missing five-minute timestamps were detected."
        )

    if len(unexpected_times) > 0:
        issues.append(
            "Unexpected timestamps were detected."
        )

    if irregular_gap_count > 0:
        issues.append(
            "Irregular adjacent time gaps were detected."
        )

    if data["open_datetime_utc"].iloc[0] != EXPECTED_START:
        issues.append(
            "The first timestamp is incorrect."
        )

    if (
        data["open_datetime_utc"].iloc[-1]
        != expected_last_open
    ):
        issues.append(
            "The last timestamp is incorrect."
        )

    if issues:
        print()
        print("[Global Timeline Check] FAIL")

        for issue in issues:
            print(f"  - {issue}")

        raise RuntimeError(
            "The merged dataset failed timeline validation."
        )

    print()
    print("[Global Timeline Check] PASS")


def main() -> None:
    months = generate_months(
        start_month=START_MONTH,
        end_month=END_MONTH,
    )

    print("=" * 75)
    print("Build Merged BTCUSDT Five-Minute Kline Dataset")
    print("=" * 75)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file:     {OUTPUT_PATH}")
    print(f"Months expected: {len(months)}")
    print()

    monthly_frames: list[pd.DataFrame] = []

    for index, year_month in enumerate(months, start=1):
        csv_path = (
            INPUT_DIR
            / f"{SYMBOL}-{INTERVAL}-{year_month}.csv"
        )

        monthly_data = read_one_month(
            csv_path=csv_path,
            year_month=year_month,
        )

        monthly_frames.append(monthly_data)

        print(
            f"[{index:02d}/{len(months):02d}] "
            f"{year_month}: {len(monthly_data)} rows"
        )

    print()
    print("Combining monthly data...")

    data = pd.concat(
        monthly_frames,
        ignore_index=True,
    )

    del monthly_frames

    data["open_datetime_utc"] = pd.to_datetime(
        data["open_time"],
        unit="ms",
        utc=True,
        errors="raise",
    )

    data["close_datetime_utc"] = pd.to_datetime(
        data["close_time"],
        unit="ms",
        utc=True,
        errors="raise",
    )

    data = (
        data
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    validate_complete_timeline(data)

    output_columns = [
        "open_datetime_utc",
        "close_datetime_utc",
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
        "source_month",
    ]

    data = data[output_columns]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    print()
    print("Writing Parquet file...")

    data.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_path.replace(OUTPUT_PATH)

    print("Reading saved file for verification...")

    verification = pd.read_parquet(
        OUTPUT_PATH,
        columns=[
            "open_datetime_utc",
            "open_time",
        ],
    )

    if len(verification) != len(data):
        raise RuntimeError(
            "Saved Parquet row count does not match "
            "the in-memory dataset."
        )

    if (
        verification["open_time"].iloc[0]
        != data["open_time"].iloc[0]
    ):
        raise RuntimeError(
            "Saved Parquet first timestamp is incorrect."
        )

    if (
        verification["open_time"].iloc[-1]
        != data["open_time"].iloc[-1]
    ):
        raise RuntimeError(
            "Saved Parquet last timestamp is incorrect."
        )

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024**2)
    )

    print()
    print("=" * 75)
    print("Merged Dataset Summary")
    print("=" * 75)
    print(f"Rows:              {len(data)}")
    print(f"Columns:           {data.shape[1]}")
    print(
        f"Start time:        "
        f"{data['open_datetime_utc'].iloc[0]}"
    )
    print(
        f"End time:          "
        f"{data['open_datetime_utc'].iloc[-1]}"
    )
    print(f"Parquet size:      {file_size_mb:.2f} MB")
    print(f"Saved to:          {OUTPUT_PATH}")

    print()
    print("[Merged Dataset] PASS")
    print(
        "[Success] The complete five-minute Kline dataset "
        "was merged, validated, and saved successfully."
    )


if __name__ == "__main__":
    main()