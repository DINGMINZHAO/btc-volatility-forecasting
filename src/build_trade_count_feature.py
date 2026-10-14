from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_harx_features_stage1_2022_2025.parquet"
)

KLINE_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_harx_features_stage2_2022_2025.parquet"
)

EXPECTED_BARS_PER_HOUR = 12


def validate_feature_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the existing HAR-X feature table.
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
        "downside_semivariance",
        "is_valid_target",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
        "is_valid_harx_downside_pair",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
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


def validate_kline_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate five-minute trade-count observations.
    """
    required_columns = [
        "open_datetime_utc",
        "count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing Kline columns: {missing_columns}"
        )

    data = (
        data[required_columns]
        .sort_values("open_datetime_utc")
        .reset_index(drop=True)
    )

    data["count"] = pd.to_numeric(
        data["count"],
        errors="raise",
    )

    if data["open_datetime_utc"].duplicated().any():
        raise ValueError(
            "Duplicate five-minute timestamps were detected."
        )

    if data["count"].isna().any():
        raise ValueError(
            "Missing trade counts were detected."
        )

    if (data["count"] < 0).any():
        raise ValueError(
            "Negative trade counts were detected."
        )

    if (data["count"] % 1 != 0).any():
        raise ValueError(
            "Non-integer trade counts were detected."
        )

    return data


def aggregate_hourly_trade_count(
    klines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sum five-minute trade counts within each hourly interval.

    An hour ending at t contains bars ending at:
        t-55 minutes, ..., t-5 minutes, t.
    """
    data = klines.copy()

    data["bar_end_utc"] = (
        data["open_datetime_utc"]
        + pd.Timedelta(minutes=5)
    )

    hourly = (
        data
        .set_index("bar_end_utc")
        .resample(
            "1h",
            closed="right",
            label="right",
        )
        .agg(
            total_5m_bars_trade=(
                "count",
                "size",
            ),
            number_of_trades=(
                "count",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "bar_end_utc": "forecast_origin_utc",
            }
        )
    )

    hourly["total_5m_bars_trade"] = (
        hourly["total_5m_bars_trade"]
        .astype("int64")
    )

    hourly["number_of_trades"] = (
        hourly["number_of_trades"]
        .astype("int64")
    )

    return hourly


def merge_trade_feature(
    feature_data: pd.DataFrame,
    hourly_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge hourly trade counts onto the forecast-origin table.
    """
    data = feature_data.merge(
        hourly_trades,
        on="forecast_origin_utc",
        how="left",
        validate="one_to_one",
    )

    merge_columns = [
        "total_5m_bars_trade",
        "number_of_trades",
    ]

    if data[merge_columns].isna().any().any():
        problem_rows = data.loc[
            data[merge_columns].isna().any(axis=1),
            ["forecast_origin_utc"] + merge_columns,
        ]

        print()
        print("Rows with unmatched trade-count data:")
        print(problem_rows.to_string(index=False))

        raise RuntimeError(
            "Some forecast origins could not be matched "
            "to hourly trade counts."
        )

    invalid_bar_count = (
        data["total_5m_bars_trade"]
        != EXPECTED_BARS_PER_HOUR
    )

    if invalid_bar_count.any():
        raise RuntimeError(
            "At least one forecast-origin hour does not "
            "contain exactly 12 five-minute Klines."
        )

    data["log1p_number_of_trades"] = np.log1p(
        data["number_of_trades"].astype("float64")
    )

    # Zero-activity contaminated hours remain on the timeline,
    # but they are not treated as valid market observations.
    invalid_current_hour = (
        ~data["is_valid_target"].astype(bool)
    )

    data.loc[
        invalid_current_hour,
        "log1p_number_of_trades",
    ] = np.nan

    data["trade_count_available"] = (
        data["is_valid_target"].astype(bool)
        & data["log1p_number_of_trades"].notna()
    )

    data["is_valid_harx_trade_pair"] = (
        data["is_valid_harx_downside_pair"].astype(bool)
        & data["trade_count_available"]
    )

    return data


def validate_trade_feature(
    data: pd.DataFrame,
) -> None:
    """
    Validate raw and transformed trade-count features.
    """
    available_mask = data["trade_count_available"]

    available = data.loc[
        available_mask,
        [
            "number_of_trades",
            "log1p_number_of_trades",
        ],
    ]

    if available.empty:
        raise RuntimeError(
            "No valid trade-count observations were constructed."
        )

    if (available["number_of_trades"] <= 0).any():
        raise RuntimeError(
            "A valid market hour has a non-positive "
            "trade count."
        )

    if not np.isfinite(
        available["log1p_number_of_trades"]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite transformed trade-count values "
            "were detected."
        )

    expected_log_values = np.log1p(
        available["number_of_trades"]
        .to_numpy(dtype=float)
    )

    if not np.allclose(
        available["log1p_number_of_trades"]
        .to_numpy(dtype=float),
        expected_log_values,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The log-transformed trade-count feature "
            "was calculated incorrectly."
        )

    invalid_hours = ~data["is_valid_target"].astype(bool)

    if data.loc[
        invalid_hours,
        "log1p_number_of_trades",
    ].notna().any():
        raise RuntimeError(
            "Invalid zero-activity hours should not contain "
            "a transformed trade-count feature."
        )


def save_dataset(
    data: pd.DataFrame,
) -> None:
    """
    Save the second-stage HAR-X feature dataset.
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
        "number_of_trades",
        "log1p_number_of_trades",
        "downside_semivariance_available",
        "trade_count_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
        "is_valid_harx_downside_pair",
        "is_valid_harx_trade_pair",
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
        verification["is_valid_harx_trade_pair"].sum()
    ) != int(
        output_data["is_valid_harx_trade_pair"].sum()
    ):
        raise RuntimeError(
            "Saved valid HAR-X trade-pair count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Hourly Trade-Count Activity Feature")
    print("=" * 80)
    print(f"Feature input: {FEATURE_INPUT_PATH}")
    print(f"Kline input:   {KLINE_INPUT_PATH}")
    print(f"Output file:   {OUTPUT_PATH}")
    print()

    if not FEATURE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature input not found:\n{FEATURE_INPUT_PATH}"
        )

    if not KLINE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Kline input not found:\n{KLINE_INPUT_PATH}"
        )

    feature_data = pd.read_parquet(
        FEATURE_INPUT_PATH
    )

    klines = pd.read_parquet(
        KLINE_INPUT_PATH,
        columns=[
            "open_datetime_utc",
            "count",
        ],
    )

    print(f"Hourly feature rows loaded: {len(feature_data)}")
    print(f"Five-minute Klines loaded:  {len(klines)}")

    feature_data = validate_feature_table(
        feature_data
    )

    klines = validate_kline_table(klines)

    hourly_trades = aggregate_hourly_trade_count(
        klines
    )

    data = merge_trade_feature(
        feature_data=feature_data,
        hourly_trades=hourly_trades,
    )

    validate_trade_feature(data)

    available = data.loc[
        data["trade_count_available"]
    ]

    raw_values = available["number_of_trades"]
    log_values = available["log1p_number_of_trades"]

    print()
    print("Trade-count feature summary:")
    print(f"Total hourly rows:              {len(data)}")
    print(
        f"Feature available rows:         "
        f"{int(data['trade_count_available'].sum())}"
    )
    print(
        f"Feature unavailable rows:       "
        f"{int((~data['trade_count_available']).sum())}"
    )
    print(
        f"Zero raw trade-count hours:      "
        f"{int((data['number_of_trades'] == 0).sum())}"
    )
    print(
        f"Valid HAR-X downside pairs:      "
        f"{int(data['is_valid_harx_downside_pair'].sum())}"
    )
    print(
        f"Valid HAR-X trade pairs:         "
        f"{int(data['is_valid_harx_trade_pair'].sum())}"
    )

    print()
    print("Raw hourly trade-count statistics:")
    print(
        raw_values.describe(
            percentiles=[
                0.01,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Log-transformed trade-count statistics:")
    print(
        log_values.describe(
            percentiles=[
                0.01,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("First three available observations:")
    print(
        available[
            [
                "forecast_origin_utc",
                "number_of_trades",
                "log1p_number_of_trades",
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
    print("[Trade-Count Feature] PASS")
    print(
        "[Success] Hourly transaction activity was constructed "
        "from the 12 five-minute Kline trade counts available "
        "at each forecast origin."
    )


if __name__ == "__main__":
    main()