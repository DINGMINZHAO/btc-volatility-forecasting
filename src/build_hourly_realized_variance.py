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
    / "interim"
    / "btcusdt_5m_returns_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_realized_variance_2022_2025.parquet"
)

EXPECTED_RETURNS_PER_HOUR = 12

EXPECTED_FIRST_HOUR_END = pd.Timestamp(
    "2022-01-01 01:00:00",
    tz="UTC",
)

EXPECTED_LAST_HOUR_END = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
)


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the five-minute return data before hourly aggregation.
    """
    required_columns = [
        "return_end_utc",
        "log_return_5m",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = (
        data[required_columns]
        .sort_values("return_end_utc")
        .reset_index(drop=True)
    )

    if data["return_end_utc"].duplicated().any():
        raise ValueError(
            "Duplicate return-end timestamps were detected."
        )

    missing_return_count = int(
        data["log_return_5m"].isna().sum()
    )

    if missing_return_count != 1:
        raise ValueError(
            "Expected exactly one initial missing return, "
            f"but found {missing_return_count}."
        )

    missing_position = np.flatnonzero(
        data["log_return_5m"].isna().to_numpy()
    )

    if len(missing_position) != 1 or missing_position[0] != 0:
        raise ValueError(
            "The missing five-minute return is not located "
            "in the first row."
        )

    valid_returns = data["log_return_5m"].dropna()

    if not np.isfinite(
        valid_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite five-minute returns were detected."
        )

    return data


def build_hourly_realized_variance(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate five-minute squared log returns into hourly
    realized variance.

    The bins use (previous hour, current hour], so an hour
    ending at 01:00 contains return-end timestamps
    00:05, 00:10, ..., 01:00.
    """
    working = data.copy()

    working["squared_return_5m"] = (
        working["log_return_5m"] ** 2
    )

    hourly = (
        working
        .set_index("return_end_utc")
        .resample(
            "1h",
            closed="right",
            label="right",
        )
        .agg(
            n_returns=(
                "log_return_5m",
                "count",
            ),
            realized_variance=(
                "squared_return_5m",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "return_end_utc": "hour_end_utc",
            }
        )
    )

    hourly["hour_start_utc"] = (
        hourly["hour_end_utc"]
        - pd.Timedelta(hours=1)
    )

    return hourly[
        [
            "hour_start_utc",
            "hour_end_utc",
            "n_returns",
            "realized_variance",
        ]
    ]


def validate_hourly_data(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check hourly boundaries, completeness, and realized variance.
    """
    expected_hour_ends = pd.date_range(
        start=EXPECTED_FIRST_HOUR_END,
        end=EXPECTED_LAST_HOUR_END,
        freq="1h",
    )

    actual_hour_ends = pd.DatetimeIndex(
        hourly["hour_end_utc"]
    )

    missing_hour_ends = expected_hour_ends.difference(
        actual_hour_ends
    )

    unexpected_hour_ends = actual_hour_ends.difference(
        expected_hour_ends
    )

    if len(hourly) != len(expected_hour_ends):
        raise ValueError(
            "The number of hourly bins is incorrect. "
            f"Expected {len(expected_hour_ends)}, "
            f"found {len(hourly)}."
        )

    if len(missing_hour_ends) > 0:
        raise ValueError(
            f"Missing hourly bins: {len(missing_hour_ends)}"
        )

    if len(unexpected_hour_ends) > 0:
        raise ValueError(
            f"Unexpected hourly bins: "
            f"{len(unexpected_hour_ends)}"
        )

    incomplete_mask = (
        hourly["n_returns"]
        != EXPECTED_RETURNS_PER_HOUR
    )

    incomplete_hours = hourly.loc[
        incomplete_mask
    ].copy()

    print()
    print("Incomplete hourly bins:")
    print(
        incomplete_hours.to_string(index=False)
    )

    if len(incomplete_hours) != 1:
        raise ValueError(
            "Expected exactly one incomplete hour caused by "
            "the missing pre-sample closing price, but found "
            f"{len(incomplete_hours)}."
        )

    if (
        incomplete_hours["hour_end_utc"].iloc[0]
        != EXPECTED_FIRST_HOUR_END
    ):
        raise ValueError(
            "The incomplete hourly bin is not the first hour."
        )

    complete_hourly = (
        hourly.loc[~incomplete_mask]
        .reset_index(drop=True)
    )

    if (
        complete_hourly["realized_variance"].isna().any()
    ):
        raise ValueError(
            "Missing hourly realized-variance values "
            "were detected."
        )

    if not np.isfinite(
        complete_hourly["realized_variance"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite hourly realized-variance values "
            "were detected."
        )

    if (
        complete_hourly["realized_variance"] < 0
    ).any():
        raise ValueError(
            "Negative realized variance was detected."
        )

    duplicate_hour_count = int(
        complete_hourly["hour_end_utc"]
        .duplicated()
        .sum()
    )

    if duplicate_hour_count > 0:
        raise ValueError(
            f"Duplicate hourly timestamps detected: "
            f"{duplicate_hour_count}"
        )

    return complete_hourly


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the complete hourly realized-variance dataset.
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
        OUTPUT_PATH
    )

    if len(verification) != len(data):
        raise RuntimeError(
            "Saved Parquet row count does not match "
            "the hourly dataset."
        )

    if not verification[
        "hour_end_utc"
    ].equals(data["hour_end_utc"]):
        raise RuntimeError(
            "Saved hourly timestamps do not match "
            "the source dataset."
        )


def main() -> None:
    print("=" * 75)
    print("Build BTCUSDT Hourly Realized Variance")
    print("=" * 75)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "return_end_utc",
            "log_return_5m",
        ],
    )

    print(f"Five-minute rows loaded: {len(data)}")

    data = validate_input(data)

    hourly = build_hourly_realized_variance(data)

    print(f"Hourly bins constructed: {len(hourly)}")

    complete_hourly = validate_hourly_data(hourly)

    print()
    print("Hourly realized-variance summary:")
    print(f"All hourly bins:        {len(hourly)}")
    print(
        f"Incomplete hourly bins: "
        f"{int((hourly['n_returns'] != 12).sum())}"
    )
    print(
        f"Complete hourly bins:   "
        f"{len(complete_hourly)}"
    )
    print(
        f"First valid hour start: "
        f"{complete_hourly['hour_start_utc'].iloc[0]}"
    )
    print(
        f"First valid hour end:   "
        f"{complete_hourly['hour_end_utc'].iloc[0]}"
    )
    print(
        f"Last valid hour end:    "
        f"{complete_hourly['hour_end_utc'].iloc[-1]}"
    )

    print()
    print("Realized-variance statistics:")
    print(
        complete_hourly["realized_variance"]
        .describe(
            percentiles=[
                0.01,
                0.50,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print()
    print("First three complete hours:")
    print(
        complete_hourly
        .head(3)
        .to_string(index=False)
    )

    save_dataset(complete_hourly)

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024**2)
    )

    print()
    print(f"Parquet size: {file_size_mb:.2f} MB")
    print(f"Saved to:     {OUTPUT_PATH}")

    print()
    print("[Hourly Realized Variance] PASS")
    print(
        "[Success] Hourly realized variance was constructed "
        "from exactly 12 five-minute returns per valid hour."
    )


if __name__ == "__main__":
    main()