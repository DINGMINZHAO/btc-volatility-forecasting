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
    / "btcusdt_har_features_stage2_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_har_features_stage3_2022_2025.parquet"
)

HAR_168H_WINDOW = 168


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the stage-two HAR table before constructing HAR168.
    """
    required_columns = [
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


def construct_har168_feature(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the trailing 168-hour average of log realized
    variance.

    The window contains:
        y_t, y_(t-1), ..., y_(t-167)

    It therefore uses only information available at the
    forecast origin.
    """
    data = data.copy()

    data["har_168h_n_valid"] = (
        data["y_current"]
        .rolling(
            window=HAR_168H_WINDOW,
            min_periods=1,
        )
        .count()
        .astype("int64")
    )

    data["har_168h"] = (
        data["y_current"]
        .rolling(
            window=HAR_168H_WINDOW,
            min_periods=HAR_168H_WINDOW,
        )
        .mean()
    )

    data["har_168h_available"] = (
        data["har_168h_n_valid"].eq(HAR_168H_WINDOW)
        & data["har_168h"].notna()
    )

    data["is_valid_har168_pair"] = (
        data["is_valid_forecast_pair"]
        & data["har_1h_available"]
        & data["har_6h_available"]
        & data["har_24h_available"]
        & data["har_168h_available"]
    )

    return data


def validate_har168(data: pd.DataFrame) -> None:
    """
    Validate the 168-hour rolling-window calculation.
    """
    # The first 167 rows cannot contain a complete 168-hour window.
    if data["har_168h"].iloc[:167].notna().any():
        raise RuntimeError(
            "HAR168 should be missing for the first 167 rows."
        )

    incomplete_windows = (
        data["har_168h_n_valid"] < HAR_168H_WINDOW
    )

    if data.loc[
        incomplete_windows,
        "har_168h",
    ].notna().any():
        raise RuntimeError(
            "HAR168 was calculated despite an incomplete "
            "168-hour window."
        )

    available_values = data.loc[
        data["har_168h_available"],
        "har_168h",
    ]

    if available_values.empty:
        raise RuntimeError(
            "No valid HAR168 observations were constructed."
        )

    if not np.isfinite(
        available_values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite HAR168 values were detected."
        )

    first_available_index = data.index[
        data["har_168h_available"]
    ][0]

    if first_available_index < HAR_168H_WINDOW - 1:
        raise RuntimeError(
            "HAR168 became available before 168 hourly "
            "observations were present."
        )

    manual_window = data.loc[
        first_available_index - 167:first_available_index,
        "y_current",
    ]

    if len(manual_window) != HAR_168H_WINDOW:
        raise RuntimeError(
            "The manual HAR168 verification window does not "
            "contain 168 rows."
        )

    manual_mean = manual_window.mean()

    calculated_mean = data.loc[
        first_available_index,
        "har_168h",
    ]

    if not np.isclose(
        manual_mean,
        calculated_mean,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The first HAR168 value does not equal the manual "
            "168-hour average."
        )

    valid_model_rows = data.loc[
        data["is_valid_har168_pair"]
    ]

    required_model_columns = [
        "har_1h",
        "har_6h",
        "har_24h",
        "har_168h",
        "y_next_hour",
    ]

    if valid_model_rows[
        required_model_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing values were detected among valid "
            "HAR168 forecast rows."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the stage-three HAR feature dataset.
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
        "har_168h",
        "har_6h_n_valid",
        "har_24h_n_valid",
        "har_168h_n_valid",
        "har_1h_available",
        "har_6h_available",
        "har_24h_available",
        "har_168h_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har1_pair",
        "is_valid_har6_pair",
        "is_valid_har24_pair",
        "is_valid_har168_pair",
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
        verification["is_valid_har168_pair"].sum()
    ) != int(
        output_data["is_valid_har168_pair"].sum()
    ):
        raise RuntimeError(
            "Saved valid HAR168-pair count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Stage-Three HAR Volatility Features")
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

    data = construct_har168_feature(data)

    validate_har168(data)

    first_available = data.loc[
        data["har_168h_available"]
    ].iloc[0]

    print()
    print("HAR168 feature summary:")
    print(f"Total hourly rows:                  {len(data)}")
    print(
        f"HAR168 available rows:              "
        f"{int(data['har_168h_available'].sum())}"
    )
    print(
        f"HAR168 unavailable rows:            "
        f"{int((~data['har_168h_available']).sum())}"
    )
    print(
        f"Valid HAR1+HAR6+HAR24 pairs:         "
        f"{int(data['is_valid_har24_pair'].sum())}"
    )
    print(
        f"Valid full HAR forecast pairs:       "
        f"{int(data['is_valid_har168_pair'].sum())}"
    )

    print()
    print("First available HAR168 observation:")
    print(
        first_available[
            [
                "forecast_origin_utc",
                "y_current",
                "har_1h",
                "har_6h",
                "har_24h",
                "har_168h",
                "har_168h_n_valid",
                "y_next_hour",
            ]
        ].to_string()
    )

    print()
    print("Rows around first HAR168 availability:")
    print(
        data[
            [
                "forecast_origin_utc",
                "y_current",
                "har_168h_n_valid",
                "har_168h",
                "har_168h_available",
            ]
        ]
        .head(171)
        .tail(7)
        .to_string(index=False)
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Stage-Three HAR Features] PASS")
    print(
        "[Success] The trailing 168-hour volatility feature "
        "was constructed using only complete historical "
        "hourly observations."
    )


if __name__ == "__main__":
    main()