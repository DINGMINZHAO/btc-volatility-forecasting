from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_forecast_base_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_har_features_stage1_2022_2025.parquet"
)

HAR_6H_WINDOW = 6


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the continuous hourly forecast table before
    constructing rolling HAR features.
    """
    required_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "y_current",
        "y_next_hour",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
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
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    duplicate_count = int(
        data["forecast_origin_utc"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate forecast origins detected: "
            f"{duplicate_count}"
        )

    time_differences = (
        data["forecast_origin_utc"]
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

    return data


def construct_har_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct current-hour and trailing six-hour HAR features.

    The rolling window is backward-looking and includes the
    current forecast-origin hour. min_periods=6 ensures that
    all six hourly target values must be valid.
    """
    data = data.copy()

    # Latest one-hour log realized variance.
    data["har_1h"] = data["y_current"]

    # Number of valid y observations inside each trailing
    # six-hour clock-time window.
    data["har_6h_n_valid"] = (
        data["y_current"]
        .rolling(
            window=HAR_6H_WINDOW,
            min_periods=1,
        )
        .count()
        .astype("int64")
    )

    # Mean of y_t, y_(t-1), ..., y_(t-5).
    # The result is missing whenever any of these six hours
    # has an invalid target.
    data["har_6h"] = (
        data["y_current"]
        .rolling(
            window=HAR_6H_WINDOW,
            min_periods=HAR_6H_WINDOW,
        )
        .mean()
    )

    data["har_1h_available"] = (
        data["har_1h"].notna()
        & np.isfinite(
            data["har_1h"].fillna(0).to_numpy(dtype=float)
        )
    )

    data["har_6h_available"] = (
        data["har_6h_n_valid"].eq(HAR_6H_WINDOW)
        & data["har_6h"].notna()
    )

    data["is_valid_har1_pair"] = (
        data["is_valid_forecast_pair"]
        & data["har_1h_available"]
    )

    data["is_valid_har6_pair"] = (
        data["is_valid_forecast_pair"]
        & data["har_1h_available"]
        & data["har_6h_available"]
    )

    return data


def validate_features(data: pd.DataFrame) -> None:
    """
    Validate rolling-window construction and missing-value logic.
    """
    # Before six observations exist, HAR6 must be unavailable.
    if data["har_6h"].iloc[:5].notna().any():
        raise RuntimeError(
            "HAR6 should be missing for the first five rows."
        )

    if data["har_6h"].iloc[5:].isna().all():
        raise RuntimeError(
            "No valid HAR6 observations were constructed."
        )

    available_har6 = data.loc[
        data["har_6h_available"],
        "har_6h",
    ]

    if not np.isfinite(
        available_har6.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite HAR6 values were detected."
        )

    # Verify the first available HAR6 value manually.
    first_available_index = data.index[
        data["har_6h_available"]
    ][0]

    manual_window = data.loc[
        first_available_index - 5:first_available_index,
        "y_current",
    ]

    manual_mean = manual_window.mean()
    calculated_mean = data.loc[
        first_available_index,
        "har_6h",
    ]

    if not np.isclose(
        manual_mean,
        calculated_mean,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The first HAR6 value does not equal the manual "
            "six-hour average."
        )

    # HAR1 should be identical to y_current, including NaNs.
    if not data["har_1h"].equals(data["y_current"]):
        raise RuntimeError(
            "HAR1 is not identical to the current-hour target."
        )

    invalid_har6_windows = data.loc[
        data["har_6h_n_valid"] < HAR_6H_WINDOW
    ]

    if invalid_har6_windows["har_6h"].notna().any():
        raise RuntimeError(
            "HAR6 was calculated despite an incomplete "
            "six-hour window."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the stage-one HAR feature table.
    """
    output_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "y_current",
        "y_next_hour",
        "har_1h",
        "har_6h",
        "har_6h_n_valid",
        "har_1h_available",
        "har_6h_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har1_pair",
        "is_valid_har6_pair",
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
        verification["is_valid_har6_pair"].sum()
    ) != int(
        output_data["is_valid_har6_pair"].sum()
    ):
        raise RuntimeError(
            "Saved valid HAR6-pair count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Stage-One HAR Volatility Features")
    print("=" * 80)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Hourly rows loaded: {len(data)}")

    data = validate_input(data)

    data = construct_har_features(data)

    validate_features(data)

    first_available = data.loc[
        data["har_6h_available"]
    ].iloc[0]

    unavailable_due_to_window = int(
        (~data["har_6h_available"]).sum()
    )

    print()
    print("HAR feature summary:")
    print(f"Total hourly rows:             {len(data)}")
    print(
        f"HAR1 available rows:           "
        f"{int(data['har_1h_available'].sum())}"
    )
    print(
        f"HAR6 available rows:           "
        f"{int(data['har_6h_available'].sum())}"
    )
    print(
        f"HAR6 unavailable rows:         "
        f"{unavailable_due_to_window}"
    )
    print(
        f"Valid HAR1 forecast pairs:     "
        f"{int(data['is_valid_har1_pair'].sum())}"
    )
    print(
        f"Valid HAR1+HAR6 forecast pairs:"
        f" {int(data['is_valid_har6_pair'].sum())}"
    )

    print()
    print("First available HAR6 observation:")
    print(
        first_available[
            [
                "forecast_origin_utc",
                "y_current",
                "har_1h",
                "har_6h",
                "har_6h_n_valid",
                "y_next_hour",
            ]
        ].to_string()
    )

    print()
    print("First eight rows:")
    print(
        data[
            [
                "forecast_origin_utc",
                "y_current",
                "har_6h_n_valid",
                "har_6h",
                "har_6h_available",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Stage-One HAR Features] PASS")
    print(
        "[Success] Current-hour and trailing six-hour "
        "volatility features were constructed using only "
        "information available at each forecast origin."
    )


if __name__ == "__main__":
    main()