from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "btcusdt_5m_returns_2022_2025.parquet"
)

EXPECTED_INTERVAL_MS = 5 * 60 * 1000

REQUIRED_COLUMNS = [
    "open_datetime_utc",
    "close_datetime_utc",
    "open_time",
    "close",
]


def validate_input_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the ordering, timestamps, and closing prices before
    calculating log returns.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = (
        data
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    duplicate_count = int(
        data["open_time"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate open times detected: {duplicate_count}"
        )

    time_differences = (
        data["open_time"]
        .diff()
        .dropna()
    )

    irregular_gap_count = int(
        (time_differences != EXPECTED_INTERVAL_MS).sum()
    )

    if irregular_gap_count > 0:
        raise ValueError(
            "Irregular five-minute gaps were detected: "
            f"{irregular_gap_count}"
        )

    data["close"] = pd.to_numeric(
        data["close"],
        errors="raise",
    )

    nonpositive_close_count = int(
        (data["close"] <= 0).sum()
    )

    if nonpositive_close_count > 0:
        raise ValueError(
            "Log returns cannot be calculated because "
            f"{nonpositive_close_count} closing prices are "
            "non-positive."
        )

    return data


def calculate_log_returns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate five-minute close-to-close log returns.

    The return is labeled by the exact five-minute boundary at
    which the information becomes available.
    """
    data = data.copy()

    data["return_end_utc"] = (
        data["open_datetime_utc"]
        + pd.Timedelta(minutes=5)
    )

    data["log_close"] = np.log(
        data["close"].astype("float64")
    )

    data["log_return_5m"] = (
        data["log_close"].diff()
    )

    return data


def validate_returns(data: pd.DataFrame) -> None:
    """
    Validate the calculated five-minute return series.
    """
    missing_return_mask = data["log_return_5m"].isna()

    missing_return_count = int(
        missing_return_mask.sum()
    )

    missing_positions = np.flatnonzero(
        missing_return_mask.to_numpy()
    )

    if missing_return_count != 1:
        raise ValueError(
            "Expected exactly one missing return at the beginning, "
            f"but found {missing_return_count}."
        )

    if missing_positions[0] != 0:
        raise ValueError(
            "The only missing return is not located in the "
            "first row."
        )

    valid_returns = (
        data["log_return_5m"].dropna()
    )

    finite_return_count = int(
        np.isfinite(
            valid_returns.to_numpy(dtype=float)
        ).sum()
    )

    if finite_return_count != len(valid_returns):
        raise ValueError(
            "Non-finite values were detected in the "
            "five-minute return series."
        )

    expected_return_end = (
        data["close_datetime_utc"]
        + pd.Timedelta(milliseconds=1)
    )

    incorrect_return_end_count = int(
        (
            data["return_end_utc"]
            != expected_return_end
        ).sum()
    )

    if incorrect_return_end_count > 0:
        raise ValueError(
            "The return-end timestamps are not aligned with "
            f"the Kline closing times in "
            f"{incorrect_return_end_count} rows."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the derived five-minute return dataset safely.
    """
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    data.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_path.replace(OUTPUT_PATH)

    verification = pd.read_parquet(
        OUTPUT_PATH,
        columns=[
            "open_time",
            "return_end_utc",
            "log_return_5m",
        ],
    )

    if len(verification) != len(data):
        raise RuntimeError(
            "Saved Parquet row count does not match "
            "the source dataset."
        )

    if int(verification["log_return_5m"].isna().sum()) != 1:
        raise RuntimeError(
            "The saved return series does not contain the "
            "expected single initial missing value."
        )


def main() -> None:
    print("=" * 75)
    print("Build BTCUSDT Five-Minute Log Return Dataset")
    print("=" * 75)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input Parquet file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(data)}")

    data = validate_input_data(data)

    data = calculate_log_returns(data)

    validate_returns(data)

    valid_returns = (
        data["log_return_5m"].dropna()
    )

    print()
    print("Return construction summary:")
    print(f"Total rows:             {len(data)}")
    print(
        f"Missing log returns:    "
        f"{int(data['log_return_5m'].isna().sum())}"
    )
    print(
        f"Valid log returns:      "
        f"{len(valid_returns)}"
    )
    print(
        f"First return end:       "
        f"{data['return_end_utc'].iloc[0]}"
    )
    print(
        f"First valid return end: "
        f"{data.loc[data['log_return_5m'].notna(), 'return_end_utc'].iloc[0]}"
    )
    print(
        f"Last return end:        "
        f"{data['return_end_utc'].iloc[-1]}"
    )

    print()
    print("Five-minute log-return statistics:")
    print(f"Mean: {valid_returns.mean():.10f}")
    print(f"Std:  {valid_returns.std():.10f}")
    print(f"Min:  {valid_returns.min():.10f}")
    print(f"Max:  {valid_returns.max():.10f}")

    print()
    print("First three observations:")
    print(
        data[
            [
                "open_datetime_utc",
                "return_end_utc",
                "close",
                "log_return_5m",
            ]
        ]
        .head(3)
        .to_string(index=False)
    )

    save_dataset(data)

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024**2)
    )

    print()
    print(f"Parquet size: {file_size_mb:.2f} MB")
    print(f"Saved to:     {OUTPUT_PATH}")

    print()
    print("[Five-Minute Returns] PASS")
    print(
        "[Success] Five-minute close-to-close log returns "
        "were constructed and time-aligned successfully."
    )


if __name__ == "__main__":
    main()