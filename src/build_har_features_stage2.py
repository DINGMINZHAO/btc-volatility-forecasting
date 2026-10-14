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
    / "btcusdt_har_features_stage1_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_har_features_stage2_2022_2025.parquet"
)

HAR_24H_WINDOW = 24


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the stage-one HAR table before constructing HAR24.
    """
    required_columns = [
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


def construct_har24_feature(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the trailing 24-hour average of log realized
    variance.

    The window contains:
        y_t, y_(t-1), ..., y_(t-23)

    It therefore uses only information available at forecast
    origin t.
    """
    data = data.copy()

    data["har_24h_n_valid"] = (
        data["y_current"]
        .rolling(
            window=HAR_24H_WINDOW,
            min_periods=1,
        )
        .count()
        .astype("int64")
    )

    data["har_24h"] = (
        data["y_current"]
        .rolling(
            window=HAR_24H_WINDOW,
            min_periods=HAR_24H_WINDOW,
        )
        .mean()
    )

    data["har_24h_available"] = (
        data["har_24h_n_valid"].eq(HAR_24H_WINDOW)
        & data["har_24h"].notna()
    )

    data["is_valid_har24_pair"] = (
        data["is_valid_forecast_pair"]
        & data["har_1h_available"]
        & data["har_6h_available"]
        & data["har_24h_available"]
    )

    return data


def validate_har24(data: pd.DataFrame) -> None:
    """
    Validate the 24-hour rolling-window calculation.
    """
    # Fewer than 24 observations exist in the first 23 rows.
    if data["har_24h"].iloc[:23].notna().any():
        raise RuntimeError(
            "HAR24 should be missing for the first 23 rows."
        )

    incomplete_windows = (
        data["har_24h_n_valid"] < HAR_24H_WINDOW
    )

    if data.loc[
        incomplete_windows,
        "har_24h",
    ].notna().any():
        raise RuntimeError(
            "HAR24 was calculated despite an incomplete "
            "24-hour window."
        )

    available_values = data.loc[
        data["har_24h_available"],
        "har_24h",
    ]

    if available_values.empty:
        raise RuntimeError(
            "No valid HAR24 observations were constructed."
        )

    if not np.isfinite(
        available_values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite HAR24 values were detected."
        )

    # Manually verify the first available value.
    first_available_index = data.index[
        data["har_24h_available"]
    ][0]

    if first_available_index < HAR_24H_WINDOW - 1:
        raise RuntimeError(
            "HAR24 became available before 24 observations "
            "were present."
        )

    manual_window = data.loc[
        first_available_index - 23:first_available_index,
        "y_current",
    ]

    if len(manual_window) != HAR_24H_WINDOW:
        raise RuntimeError(
            "The manual HAR24 verification window does not "
            "contain 24 rows."
        )

    manual_mean = manual_window.mean()

    calculated_mean = data.loc[
        first_available_index,
        "har_24h",
    ]

    if not np.isclose(
        manual_mean,
        calculated_mean,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The first HAR24 value does not equal the manual "
            "24-hour average."
        )

    valid_model_rows = data.loc[
        data["is_valid_har24_pair"]
    ]

    required_model_columns = [
        "har_1h",
        "har_6h",
        "har_24h",
        "y_next_hour",
    ]

    if valid_model_rows[
        required_model_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing values were detected among valid "
            "HAR24 forecast rows."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the stage-two HAR feature dataset.
    """
    output_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "y_current",
        "y_next_hour",
        "har_1h",
        "har_6h",
        "har_24h",
        "har_6h_n_valid",
        "har_24h_n_valid",
        "har_1h_available",
        "har_6h_available",
        "har_24h_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har1_pair",
        "is_valid_har6_pair",
        "is_valid_har24_pair",
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
        verification["is_valid_har24_pair"].sum()
    ) != int(
        output_data["is_valid_har24_pair"].sum()
    ):
        raise RuntimeError(
            "Saved valid HAR24-pair count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Stage-Two HAR Volatility Features")
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

    data = construct_har24_feature(data)

    validate_har24(data)

    first_available = data.loc[
        data["har_24h_available"]
    ].iloc[0]

    print()
    print("HAR24 feature summary:")
    print(f"Total hourly rows:              {len(data)}")
    print(
        f"HAR24 available rows:           "
        f"{int(data['har_24h_available'].sum())}"
    )
    print(
        f"HAR24 unavailable rows:         "
        f"{int((~data['har_24h_available']).sum())}"
    )
    print(
        f"Valid HAR1+HAR6 pairs:           "
        f"{int(data['is_valid_har6_pair'].sum())}"
    )
    print(
        f"Valid HAR1+HAR6+HAR24 pairs:     "
        f"{int(data['is_valid_har24_pair'].sum())}"
    )

    print()
    print("First available HAR24 observation:")
    print(
        first_available[
            [
                "forecast_origin_utc",
                "y_current",
                "har_1h",
                "har_6h",
                "har_24h",
                "har_24h_n_valid",
                "y_next_hour",
            ]
        ].to_string()
    )

    print()
    print("Rows around first HAR24 availability:")
    print(
        data[
            [
                "forecast_origin_utc",
                "y_current",
                "har_24h_n_valid",
                "har_24h",
                "har_24h_available",
            ]
        ]
        .head(27)
        .tail(7)
        .to_string(index=False)
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Stage-Two HAR Features] PASS")
    print(
        "[Success] The trailing 24-hour volatility feature "
        "was constructed using only complete historical "
        "hourly observations."
    )


if __name__ == "__main__":
    main()