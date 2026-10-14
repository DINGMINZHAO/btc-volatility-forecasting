from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_target_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_forecast_base_2022_2025.parquet"
)


def validate_hourly_grid(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the continuous hourly time grid before shifting
    the prediction target.
    """
    required_columns = [
        "hour_start_utc",
        "hour_end_utc",
        "realized_variance",
        "log_realized_variance",
        "is_valid_target",
        "has_zero_activity",
        "target_status",
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
        data
        .sort_values("hour_end_utc")
        .reset_index(drop=True)
    )

    duplicate_count = int(
        data["hour_end_utc"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate hourly timestamps detected: "
            f"{duplicate_count}"
        )

    time_differences = (
        data["hour_end_utc"]
        .diff()
        .dropna()
    )

    irregular_gap_count = int(
        (
            time_differences
            != pd.Timedelta(hours=1)
        ).sum()
    )

    if irregular_gap_count > 0:
        raise ValueError(
            f"Irregular hourly gaps detected: "
            f"{irregular_gap_count}"
        )

    incorrect_hour_duration_count = int(
        (
            data["hour_end_utc"]
            - data["hour_start_utc"]
            != pd.Timedelta(hours=1)
        ).sum()
    )

    if incorrect_hour_duration_count > 0:
        raise ValueError(
            "Incorrect hourly interval duration detected in "
            f"{incorrect_hour_duration_count} rows."
        )

    return data


def construct_forecast_alignment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align information available at forecast origin t with the
    next-hour target y_(t+1).

    Shifting is performed before filtering invalid observations,
    preserving the true hourly distance.
    """
    data = data.copy()

    data["is_valid_target"] = (
        data["is_valid_target"].astype(bool)
    )

    # Information available at the forecast origin.
    data["forecast_origin_utc"] = data["hour_end_utc"]
    data["y_current"] = data["log_realized_variance"]

    # The next row corresponds to the following real clock hour.
    data["target_hour_start_utc"] = (
        data["hour_start_utc"].shift(-1)
    )

    data["target_hour_end_utc"] = (
        data["hour_end_utc"].shift(-1)
    )

    data["y_next_hour"] = (
        data["log_realized_variance"].shift(-1)
    )

    data["target_is_valid"] = (
        data["is_valid_target"]
        .shift(-1)
        .fillna(False)
        .astype(bool)
    )

    data["target_status_next_hour"] = (
        data["target_status"].shift(-1)
    )

    data["is_exactly_one_hour_ahead"] = (
        data["target_hour_end_utc"]
        - data["forecast_origin_utc"]
        == pd.Timedelta(hours=1)
    )

    data["is_valid_forecast_pair"] = (
        data["is_valid_target"]
        & data["target_is_valid"]
        & data["is_exactly_one_hour_ahead"]
    )

    return data


def validate_alignment(data: pd.DataFrame) -> None:
    """
    Check that every non-final row points to the immediately
    following clock hour.
    """
    nonfinal = data.iloc[:-1]
    final_row = data.iloc[-1]

    incorrect_target_start = int(
        (
            nonfinal["target_hour_start_utc"]
            != nonfinal["forecast_origin_utc"]
        ).sum()
    )

    incorrect_target_end = int(
        (
            nonfinal["target_hour_end_utc"]
            != (
                nonfinal["forecast_origin_utc"]
                + pd.Timedelta(hours=1)
            )
        ).sum()
    )

    if incorrect_target_start > 0:
        raise RuntimeError(
            "Some target hours do not start at the forecast "
            f"origin: {incorrect_target_start}"
        )

    if incorrect_target_end > 0:
        raise RuntimeError(
            "Some targets are not exactly one hour ahead: "
            f"{incorrect_target_end}"
        )

    if pd.notna(final_row["target_hour_end_utc"]):
        raise RuntimeError(
            "The final observation should not have a future "
            "target timestamp."
        )

    if pd.notna(final_row["y_next_hour"]):
        raise RuntimeError(
            "The final observation should not have a next-hour "
            "target value."
        )

    valid_pairs = data.loc[
        data["is_valid_forecast_pair"]
    ]

    if valid_pairs.empty:
        raise RuntimeError(
            "No valid one-hour-ahead forecast pairs were found."
        )

    if valid_pairs[
        ["y_current", "y_next_hour"]
    ].isna().any().any():
        raise RuntimeError(
            "Missing target values were detected among valid "
            "forecast pairs."
        )

    valid_values = valid_pairs[
        ["y_current", "y_next_hour"]
    ].to_numpy(dtype=float)

    if not np.isfinite(valid_values).all():
        raise RuntimeError(
            "Non-finite values were detected among valid "
            "forecast pairs."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the full aligned hourly table without deleting invalid
    timestamps.
    """
    output_columns = [
        "forecast_origin_utc",
        "hour_start_utc",
        "hour_end_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "realized_variance",
        "y_current",
        "y_next_hour",
        "is_valid_target",
        "target_is_valid",
        "has_zero_activity",
        "target_status",
        "target_status_next_hour",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
    ]

    output_data = data[output_columns].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    output_data.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_path.replace(OUTPUT_PATH)

    verification = pd.read_parquet(OUTPUT_PATH)

    if len(verification) != len(output_data):
        raise RuntimeError(
            "Saved row count does not match the source table."
        )

    if int(
        verification["is_valid_forecast_pair"].sum()
    ) != int(
        output_data["is_valid_forecast_pair"].sum()
    ):
        raise RuntimeError(
            "Saved valid-pair count does not match the "
            "source table."
        )


def main() -> None:
    print("=" * 80)
    print("Build One-Hour-Ahead Forecast Base Table")
    print("=" * 80)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Hourly observations loaded: {len(data)}")

    data = validate_hourly_grid(data)

    data = construct_forecast_alignment(data)

    validate_alignment(data)

    valid_pairs = data.loc[
        data["is_valid_forecast_pair"]
    ]

    invalid_pairs = data.loc[
        ~data["is_valid_forecast_pair"]
    ]

    print()
    print("Forecast alignment summary:")
    print(f"Total hourly rows:          {len(data)}")
    print(
        f"Valid one-hour pairs:       "
        f"{len(valid_pairs)}"
    )
    print(
        f"Invalid or unavailable pairs: "
        f"{len(invalid_pairs)}"
    )
    print(
        f"Current invalid targets:    "
        f"{int((~data['is_valid_target']).sum())}"
    )
    print(
        f"Next-hour invalid targets:  "
        f"{int((~data['target_is_valid']).sum())}"
    )
    print(
        f"Non-one-hour alignments:    "
        f"{int((~data['is_exactly_one_hour_ahead']).sum())}"
    )

    print()
    print("First three valid forecast pairs:")
    print(
        valid_pairs[
            [
                "forecast_origin_utc",
                "target_hour_start_utc",
                "target_hour_end_utc",
                "y_current",
                "y_next_hour",
            ]
        ]
        .head(3)
        .to_string(index=False)
    )

    print()
    print("Last valid forecast pair:")
    print(
        valid_pairs[
            [
                "forecast_origin_utc",
                "target_hour_start_utc",
                "target_hour_end_utc",
                "y_current",
                "y_next_hour",
            ]
        ]
        .tail(1)
        .to_string(index=False)
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Forecast Alignment] PASS")
    print(
        "[Success] Current-hour information and the next-hour "
        "prediction target were aligned without compressing "
        "the time axis."
    )


if __name__ == "__main__":
    main()