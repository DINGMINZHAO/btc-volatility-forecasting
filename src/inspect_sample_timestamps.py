from __future__ import annotations

from pathlib import Path

import pandas as pd


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEAR_MONTH = "2022-01"

INTERVAL_MILLISECONDS = 5 * 60 * 1000
EXPECTED_KLINE_DURATION = INTERVAL_MILLISECONDS - 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sample"
    / f"{SYMBOL}-{INTERVAL}-{YEAR_MONTH}.csv"
)


def main() -> None:
    print("=" * 65)
    print("Inspect Binance Kline Sample Timestamps")
    print("=" * 65)
    print(f"CSV path: {CSV_PATH}")
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    required_columns = ["open_time", "close_time"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Ensure timestamps are numeric before conversion.
    data["open_time"] = pd.to_numeric(
        data["open_time"],
        errors="raise",
    )

    data["close_time"] = pd.to_numeric(
        data["close_time"],
        errors="raise",
    )

    print("Raw timestamp range:")
    print(f"Minimum open_time:  {data['open_time'].min()}")
    print(f"Maximum open_time:  {data['open_time'].max()}")
    print(f"Minimum close_time: {data['close_time'].min()}")
    print(f"Maximum close_time: {data['close_time'].max()}")

    # Binance timestamps are expected to be Unix milliseconds.
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

    data = data.sort_values("open_time").reset_index(drop=True)

    print()
    print("Converted UTC range:")
    print(
        f"First open time:  "
        f"{data['open_datetime_utc'].iloc[0]}"
    )
    print(
        f"Last open time:   "
        f"{data['open_datetime_utc'].iloc[-1]}"
    )
    print(
        f"First close time: "
        f"{data['close_datetime_utc'].iloc[0]}"
    )
    print(
        f"Last close time:  "
        f"{data['close_datetime_utc'].iloc[-1]}"
    )

    expected_start = pd.Timestamp(
        "2022-01-01 00:00:00",
        tz="UTC",
    )

    expected_last_open = pd.Timestamp(
        "2022-01-31 23:55:00",
        tz="UTC",
    )

    expected_last_close = pd.Timestamp(
        "2022-01-31 23:59:59.999",
        tz="UTC",
    )

    expected_index = pd.date_range(
        start=expected_start,
        end=expected_last_open,
        freq="5min",
    )

    actual_index = pd.DatetimeIndex(
        data["open_datetime_utc"]
    )

    expected_rows = len(expected_index)
    actual_rows = len(data)

    duplicate_count = data["open_time"].duplicated().sum()

    missing_intervals = expected_index.difference(actual_index)
    unexpected_intervals = actual_index.difference(expected_index)

    open_time_differences = data["open_time"].diff().dropna()

    irregular_gap_count = (
        open_time_differences != INTERVAL_MILLISECONDS
    ).sum()

    misaligned_open_times = (
        data["open_time"] % INTERVAL_MILLISECONDS != 0
    ).sum()

    observed_durations = (
        data["close_time"] - data["open_time"]
    )

    invalid_duration_count = (
        observed_durations != EXPECTED_KLINE_DURATION
    ).sum()

    print()
    print("Completeness checks:")
    print(f"Expected rows:             {expected_rows}")
    print(f"Actual rows:               {actual_rows}")
    print(f"Duplicate open times:      {duplicate_count}")
    print(f"Missing 5-minute periods:  {len(missing_intervals)}")
    print(f"Unexpected open times:     {len(unexpected_intervals)}")
    print(f"Irregular adjacent gaps:   {irregular_gap_count}")
    print(f"Misaligned open times:     {misaligned_open_times}")
    print(f"Invalid Kline durations:   {invalid_duration_count}")

    print()
    print("Expected boundaries:")
    print(f"Expected first open: {expected_start}")
    print(f"Expected last open:  {expected_last_open}")
    print(f"Expected last close: {expected_last_close}")

    issues = []

    if actual_rows != expected_rows:
        issues.append(
            f"Expected {expected_rows} rows, found {actual_rows}."
        )

    if data["open_datetime_utc"].iloc[0] != expected_start:
        issues.append("The first open timestamp is incorrect.")

    if data["open_datetime_utc"].iloc[-1] != expected_last_open:
        issues.append("The last open timestamp is incorrect.")

    if data["close_datetime_utc"].iloc[-1] != expected_last_close:
        issues.append("The last close timestamp is incorrect.")

    if duplicate_count > 0:
        issues.append("Duplicate open timestamps were detected.")

    if len(missing_intervals) > 0:
        issues.append("Missing five-minute periods were detected.")

    if len(unexpected_intervals) > 0:
        issues.append("Unexpected timestamps were detected.")

    if irregular_gap_count > 0:
        issues.append("Irregular adjacent time gaps were detected.")

    if misaligned_open_times > 0:
        issues.append(
            "Some open timestamps are not aligned "
            "to five-minute boundaries."
        )

    if invalid_duration_count > 0:
        issues.append(
            "Some Klines do not have the expected duration."
        )

    print()

    if issues:
        print("[Timestamp Check] FAIL")

        for issue in issues:
            print(f"  - {issue}")

        if len(missing_intervals) > 0:
            print()
            print("First missing timestamps:")
            print(missing_intervals[:10])

        raise RuntimeError(
            "Timestamp validation found one or more problems."
        )

    print("[Timestamp Check] PASS")
    print(
        "[Success] The sample contains a complete and correctly "
        "aligned five-minute UTC time series."
    )


if __name__ == "__main__":
    main()