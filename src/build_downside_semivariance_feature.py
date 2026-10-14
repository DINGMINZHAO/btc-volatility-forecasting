from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HAR_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_har_features_stage3_2022_2025.parquet"
)

RETURNS_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "btcusdt_5m_returns_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_harx_features_stage1_2022_2025.parquet"
)

EXPECTED_RETURNS_PER_HOUR = 12


def validate_har_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the full HAR feature table before adding the
    downside-semivariance feature.
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
        "har_168h",
        "is_valid_target",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing HAR columns: {missing_columns}"
        )

    data = (
        data
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    if data["forecast_origin_utc"].duplicated().any():
        raise ValueError(
            "Duplicate forecast origins were detected."
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


def validate_return_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the five-minute returns used to construct hourly
    downside realized semivariance.
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
            f"Missing return columns: {missing_columns}"
        )

    data = (
        data[required_columns]
        .sort_values("return_end_utc")
        .reset_index(drop=True)
    )

    if data["return_end_utc"].duplicated().any():
        raise ValueError(
            "Duplicate five-minute return timestamps "
            "were detected."
        )

    valid_returns = data["log_return_5m"].dropna()

    if not np.isfinite(
        valid_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite five-minute returns were detected."
        )

    return data


def aggregate_downside_semivariance(
    returns_5m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct hourly downside realized semivariance:

        sum(r^2 * I(r < 0))

    Hourly bins are labeled by their ending timestamp.
    """
    working = returns_5m.copy()

    working["squared_return_5m"] = (
        working["log_return_5m"] ** 2
    )

    working["is_negative_return"] = (
        working["log_return_5m"] < 0
    )

    working["downside_component"] = np.where(
        working["log_return_5m"].isna(),
        np.nan,
        np.where(
            working["is_negative_return"],
            working["squared_return_5m"],
            0.0,
        ),
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
            n_5m_returns=(
                "log_return_5m",
                "count",
            ),
            n_negative_5m_returns=(
                "is_negative_return",
                "sum",
            ),
            realized_variance_check=(
                "squared_return_5m",
                "sum",
            ),
            downside_semivariance=(
                "downside_component",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "return_end_utc": "forecast_origin_utc",
            }
        )
    )

    hourly["n_negative_5m_returns"] = (
        hourly["n_negative_5m_returns"]
        .astype("int64")
    )

    return hourly


def merge_feature(
    har_data: pd.DataFrame,
    hourly_downside: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the hourly downside feature onto the continuous HAR
    table using the forecast-origin timestamp.
    """
    merged = har_data.merge(
        hourly_downside,
        on="forecast_origin_utc",
        how="left",
        validate="one_to_one",
    )

    feature_columns = [
        "n_5m_returns",
        "n_negative_5m_returns",
        "realized_variance_check",
        "downside_semivariance",
    ]

    if merged[feature_columns].isna().any().any():
        problem_rows = merged.loc[
            merged[feature_columns].isna().any(axis=1),
            ["forecast_origin_utc"] + feature_columns,
        ]

        print()
        print("Rows with unmatched downside features:")
        print(problem_rows.to_string(index=False))

        raise RuntimeError(
            "Some HAR rows could not be matched to the "
            "hourly downside-semivariance data."
        )

    incomplete_hours = merged.loc[
        merged["n_5m_returns"]
        != EXPECTED_RETURNS_PER_HOUR
    ]

    if not incomplete_hours.empty:
        raise RuntimeError(
            "At least one forecast-origin hour does not contain "
            "exactly 12 valid five-minute returns."
        )

    # Zero-activity hours are retained on the time grid but are
    # not treated as valid market observations.
    invalid_current_hour = (
        ~merged["is_valid_target"].astype(bool)
    )

    merged.loc[
        invalid_current_hour,
        "downside_semivariance",
    ] = np.nan

    merged.loc[
        invalid_current_hour,
        "n_negative_5m_returns",
    ] = np.nan

    merged["downside_semivariance_available"] = (
        merged["is_valid_target"].astype(bool)
        & merged["downside_semivariance"].notna()
    )

    merged["is_valid_harx_downside_pair"] = (
        merged["is_valid_har168_pair"].astype(bool)
        & merged["downside_semivariance_available"]
    )

    return merged


def validate_feature(data: pd.DataFrame) -> None:
    """
    Validate sign, magnitude, and missing-value handling.
    """
    available_mask = (
        data["downside_semivariance_available"]
    )

    available_values = data.loc[
        available_mask,
        "downside_semivariance",
    ]

    if available_values.empty:
        raise RuntimeError(
            "No valid downside-semivariance observations "
            "were constructed."
        )

    if not np.isfinite(
        available_values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite downside-semivariance values "
            "were detected."
        )

    if (available_values < 0).any():
        raise RuntimeError(
            "Negative downside semivariance was detected."
        )

    comparison_tolerance = 1e-15

    downside_above_total = (
        data.loc[
            available_mask,
            "downside_semivariance",
        ]
        > (
            data.loc[
                available_mask,
                "realized_variance_check",
            ]
            + comparison_tolerance
        )
    )

    if downside_above_total.any():
        raise RuntimeError(
            "Downside semivariance exceeded total realized "
            "variance in at least one hour."
        )

    negative_counts = data.loc[
        available_mask,
        "n_negative_5m_returns",
    ]

    if (
        (negative_counts < 0)
        | (negative_counts > EXPECTED_RETURNS_PER_HOUR)
    ).any():
        raise RuntimeError(
            "Invalid counts of negative five-minute returns "
            "were detected."
        )

    invalid_hours = ~data["is_valid_target"].astype(bool)

    if data.loc[
        invalid_hours,
        "downside_semivariance",
    ].notna().any():
        raise RuntimeError(
            "Invalid zero-activity hours should not retain "
            "downside-semivariance values."
        )

    # Manually verify the first valid observation.
    first_index = data.index[
        data["downside_semivariance_available"]
    ][0]

    downside_value = data.loc[
        first_index,
        "downside_semivariance",
    ]

    total_rv = data.loc[
        first_index,
        "realized_variance_check",
    ]

    if downside_value > total_rv + comparison_tolerance:
        raise RuntimeError(
            "The first downside-semivariance observation "
            "failed the total-variance comparison."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the first-stage HAR-X feature dataset.
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
        "downside_semivariance",
        "n_negative_5m_returns",
        "downside_semivariance_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
        "is_valid_harx_downside_pair",
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
        verification[
            "is_valid_harx_downside_pair"
        ].sum()
    ) != int(
        output_data[
            "is_valid_harx_downside_pair"
        ].sum()
    ):
        raise RuntimeError(
            "Saved valid HAR-X pair count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Hourly Downside Realized-Semivariance Feature")
    print("=" * 80)
    print(f"HAR input:     {HAR_INPUT_PATH}")
    print(f"Returns input: {RETURNS_INPUT_PATH}")
    print(f"Output file:   {OUTPUT_PATH}")
    print()

    if not HAR_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"HAR input file not found:\n{HAR_INPUT_PATH}"
        )

    if not RETURNS_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Returns input file not found:\n"
            f"{RETURNS_INPUT_PATH}"
        )

    har_data = pd.read_parquet(HAR_INPUT_PATH)

    returns_5m = pd.read_parquet(
        RETURNS_INPUT_PATH,
        columns=[
            "return_end_utc",
            "log_return_5m",
        ],
    )

    print(f"HAR rows loaded:       {len(har_data)}")
    print(f"Five-minute rows loaded: {len(returns_5m)}")

    har_data = validate_har_table(har_data)
    returns_5m = validate_return_table(returns_5m)

    hourly_downside = aggregate_downside_semivariance(
        returns_5m
    )

    data = merge_feature(
        har_data=har_data,
        hourly_downside=hourly_downside,
    )

    validate_feature(data)

    available = data.loc[
        data["downside_semivariance_available"]
    ]

    feature_values = available[
        "downside_semivariance"
    ]

    zero_feature_count = int(
        (feature_values == 0).sum()
    )

    print()
    print("Downside-semivariance summary:")
    print(f"Total hourly rows:                  {len(data)}")
    print(
        f"Feature available rows:             "
        f"{int(data['downside_semivariance_available'].sum())}"
    )
    print(
        f"Feature unavailable rows:           "
        f"{int((~data['downside_semivariance_available']).sum())}"
    )
    print(
        f"Zero downside-semivariance hours:   "
        f"{zero_feature_count}"
    )
    print(
        f"Valid full HAR pairs:               "
        f"{int(data['is_valid_har168_pair'].sum())}"
    )
    print(
        f"Valid HAR-X downside pairs:         "
        f"{int(data['is_valid_harx_downside_pair'].sum())}"
    )

    print()
    print("Downside-semivariance statistics:")
    print(
        feature_values.describe(
            percentiles=[
                0.01,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Negative-return count distribution:")
    print(
        available[
            "n_negative_5m_returns"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("First three available observations:")
    print(
        available[
            [
                "forecast_origin_utc",
                "n_negative_5m_returns",
                "downside_semivariance",
                "y_current",
                "y_next_hour",
            ]
        ]
        .head(3)
        .to_string(index=False)
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Downside Semivariance Feature] PASS")
    print(
        "[Success] Hourly downside realized semivariance "
        "was constructed using only five-minute returns "
        "available at each forecast origin."
    )


if __name__ == "__main__":
    main()