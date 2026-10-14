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
    / "btcusdt_harx_features_stage2_2022_2025.parquet"
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
    / "btcusdt_harx_features_complete_2022_2025.parquet"
)

EXPECTED_BARS_PER_HOUR = 12

BASE_VOLUME_TOLERANCE = 1e-10
QUOTE_VOLUME_TOLERANCE = 1e-6
PROXY_TOLERANCE = 1e-10


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
        "number_of_trades",
        "log1p_number_of_trades",
        "is_valid_target",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
        "is_valid_harx_downside_pair",
        "is_valid_harx_trade_pair",
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
    Validate the five-minute volume fields used for the
    aggressive-buy proxy.
    """
    required_columns = [
        "open_datetime_utc",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
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

    numeric_columns = [
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data["open_datetime_utc"].duplicated().any():
        raise ValueError(
            "Duplicate five-minute timestamps were detected."
        )

    if data[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing volume values were detected."
        )

    if (data[numeric_columns] < 0).any().any():
        raise ValueError(
            "Negative volume values were detected."
        )

    base_exceeds_total = (
        data["taker_buy_volume"]
        > data["volume"] + BASE_VOLUME_TOLERANCE
    )

    quote_exceeds_total = (
        data["taker_buy_quote_volume"]
        > data["quote_volume"] + QUOTE_VOLUME_TOLERANCE
    )

    if base_exceeds_total.any():
        raise ValueError(
            "Taker buy base volume exceeded total base volume."
        )

    if quote_exceeds_total.any():
        raise ValueError(
            "Taker buy quote volume exceeded total quote volume."
        )

    return data


def aggregate_hourly_volumes(
    klines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate five-minute total and taker-buy volumes into
    hourly intervals.
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
            total_5m_bars_volume=(
                "volume",
                "size",
            ),
            hourly_base_volume=(
                "volume",
                "sum",
            ),
            hourly_quote_volume=(
                "quote_volume",
                "sum",
            ),
            hourly_taker_buy_base_volume=(
                "taker_buy_volume",
                "sum",
            ),
            hourly_taker_buy_quote_volume=(
                "taker_buy_quote_volume",
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

    hourly["total_5m_bars_volume"] = (
        hourly["total_5m_bars_volume"]
        .astype("int64")
    )

    return hourly


def construct_aggressive_buy_proxy(
    feature_data: pd.DataFrame,
    hourly_volumes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge hourly volumes and construct the base-volume
    aggressive-buy proxy.

    A quote-volume version is calculated only as a diagnostic.
    It will not be used as a separate model feature.
    """
    data = feature_data.merge(
        hourly_volumes,
        on="forecast_origin_utc",
        how="left",
        validate="one_to_one",
    )

    volume_columns = [
        "total_5m_bars_volume",
        "hourly_base_volume",
        "hourly_quote_volume",
        "hourly_taker_buy_base_volume",
        "hourly_taker_buy_quote_volume",
    ]

    if data[volume_columns].isna().any().any():
        problem_rows = data.loc[
            data[volume_columns].isna().any(axis=1),
            ["forecast_origin_utc"] + volume_columns,
        ]

        print()
        print("Rows with unmatched hourly volumes:")
        print(problem_rows.to_string(index=False))

        raise RuntimeError(
            "Some forecast origins could not be matched "
            "to hourly volume data."
        )

    invalid_bar_count = (
        data["total_5m_bars_volume"]
        != EXPECTED_BARS_PER_HOUR
    )

    if invalid_bar_count.any():
        raise RuntimeError(
            "At least one forecast-origin hour does not "
            "contain exactly 12 five-minute Klines."
        )

    base_above_total = (
        data["hourly_taker_buy_base_volume"]
        > (
            data["hourly_base_volume"]
            + BASE_VOLUME_TOLERANCE
        )
    )

    quote_above_total = (
        data["hourly_taker_buy_quote_volume"]
        > (
            data["hourly_quote_volume"]
            + QUOTE_VOLUME_TOLERANCE
        )
    )

    if base_above_total.any():
        raise RuntimeError(
            "Hourly taker-buy base volume exceeded total volume."
        )

    if quote_above_total.any():
        raise RuntimeError(
            "Hourly taker-buy quote volume exceeded "
            "total quote volume."
        )

    positive_base_volume = (
        data["hourly_base_volume"] > 0
    )

    positive_quote_volume = (
        data["hourly_quote_volume"] > 0
    )

    data["aggressive_buy_proxy"] = np.nan

    data.loc[
        positive_base_volume,
        "aggressive_buy_proxy",
    ] = (
        2.0
        * (
            data.loc[
                positive_base_volume,
                "hourly_taker_buy_base_volume",
            ]
            / data.loc[
                positive_base_volume,
                "hourly_base_volume",
            ]
        )
        - 1.0
    )

    # Diagnostic only: quote-volume version.
    data["aggressive_buy_proxy_quote_check"] = np.nan

    data.loc[
        positive_quote_volume,
        "aggressive_buy_proxy_quote_check",
    ] = (
        2.0
        * (
            data.loc[
                positive_quote_volume,
                "hourly_taker_buy_quote_volume",
            ]
            / data.loc[
                positive_quote_volume,
                "hourly_quote_volume",
            ]
        )
        - 1.0
    )

    # Invalid zero-activity-contaminated hours remain on the
    # time grid but cannot be used as market observations.
    invalid_current_hour = (
        ~data["is_valid_target"].astype(bool)
    )

    data.loc[
        invalid_current_hour,
        [
            "aggressive_buy_proxy",
            "aggressive_buy_proxy_quote_check",
        ],
    ] = np.nan

    data["aggressive_buy_available"] = (
        data["is_valid_target"].astype(bool)
        & data["aggressive_buy_proxy"].notna()
    )

    data["is_valid_complete_harx_pair"] = (
        data["is_valid_harx_trade_pair"].astype(bool)
        & data["aggressive_buy_available"]
    )

    return data


def validate_proxy(
    data: pd.DataFrame,
) -> None:
    """
    Validate proxy range, missing values, and diagnostic
    consistency.
    """
    available_mask = data["aggressive_buy_available"]

    available = data.loc[
        available_mask,
        [
            "aggressive_buy_proxy",
            "aggressive_buy_proxy_quote_check",
            "hourly_base_volume",
            "hourly_taker_buy_base_volume",
        ],
    ]

    if available.empty:
        raise RuntimeError(
            "No valid aggressive-buy observations "
            "were constructed."
        )

    if not np.isfinite(
        available[
            "aggressive_buy_proxy"
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite aggressive-buy proxy values "
            "were detected."
        )

    below_lower_bound = (
        available["aggressive_buy_proxy"]
        < -1.0 - PROXY_TOLERANCE
    )

    above_upper_bound = (
        available["aggressive_buy_proxy"]
        > 1.0 + PROXY_TOLERANCE
    )

    if below_lower_bound.any() or above_upper_bound.any():
        raise RuntimeError(
            "Aggressive-buy proxy values outside [-1, 1] "
            "were detected."
        )

    if (
        available["hourly_base_volume"] <= 0
    ).any():
        raise RuntimeError(
            "A valid aggressive-buy observation has "
            "non-positive total volume."
        )

    invalid_hours = (
        ~data["is_valid_target"].astype(bool)
    )

    if data.loc[
        invalid_hours,
        "aggressive_buy_proxy",
    ].notna().any():
        raise RuntimeError(
            "Invalid zero-activity hours should not contain "
            "an aggressive-buy proxy."
        )


def save_dataset(
    data: pd.DataFrame,
) -> None:
    """
    Save the complete candidate HAR-X feature table.
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
        "number_of_trades",
        "log1p_number_of_trades",
        "hourly_base_volume",
        "hourly_taker_buy_base_volume",
        "aggressive_buy_proxy",
        "n_negative_5m_returns",
        "downside_semivariance_available",
        "trade_count_available",
        "aggressive_buy_available",
        "is_valid_target",
        "target_is_valid",
        "is_exactly_one_hour_ahead",
        "is_valid_forecast_pair",
        "is_valid_har168_pair",
        "is_valid_harx_downside_pair",
        "is_valid_harx_trade_pair",
        "is_valid_complete_harx_pair",
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
            "is_valid_complete_harx_pair"
        ].sum()
    ) != int(
        output_data[
            "is_valid_complete_harx_pair"
        ].sum()
    ):
        raise RuntimeError(
            "Saved valid complete HAR-X pair count "
            "is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Hourly Aggressive-Buy Proxy")
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
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ],
    )

    print(f"Hourly feature rows loaded: {len(feature_data)}")
    print(f"Five-minute Klines loaded:  {len(klines)}")

    feature_data = validate_feature_table(
        feature_data
    )

    klines = validate_kline_table(klines)

    hourly_volumes = aggregate_hourly_volumes(
        klines
    )

    data = construct_aggressive_buy_proxy(
        feature_data=feature_data,
        hourly_volumes=hourly_volumes,
    )

    validate_proxy(data)

    available = data.loc[
        data["aggressive_buy_available"]
    ]

    proxy = available["aggressive_buy_proxy"]

    diagnostic = available[
        [
            "aggressive_buy_proxy",
            "aggressive_buy_proxy_quote_check",
        ]
    ].dropna()

    proxy_correlation = diagnostic.corr().iloc[0, 1]

    maximum_proxy_difference = (
        diagnostic["aggressive_buy_proxy"]
        - diagnostic["aggressive_buy_proxy_quote_check"]
    ).abs().max()

    print()
    print("Aggressive-buy feature summary:")
    print(f"Total hourly rows:                 {len(data)}")
    print(
        f"Feature available rows:            "
        f"{int(data['aggressive_buy_available'].sum())}"
    )
    print(
        f"Feature unavailable rows:          "
        f"{int((~data['aggressive_buy_available']).sum())}"
    )
    print(
        f"Zero-total-volume hours:            "
        f"{int((data['hourly_base_volume'] == 0).sum())}"
    )
    print(
        f"Valid HAR-X trade pairs:            "
        f"{int(data['is_valid_harx_trade_pair'].sum())}"
    )
    print(
        f"Valid complete HAR-X pairs:         "
        f"{int(data['is_valid_complete_harx_pair'].sum())}"
    )

    print()
    print("Aggressive-buy proxy statistics:")
    print(
        proxy.describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Base-versus-quote diagnostic:")
    print(
        f"Correlation:             "
        f"{proxy_correlation:.10f}"
    )
    print(
        f"Maximum absolute difference: "
        f"{maximum_proxy_difference:.10f}"
    )

    print()
    print("First three available observations:")
    print(
        available[
            [
                "forecast_origin_utc",
                "hourly_base_volume",
                "hourly_taker_buy_base_volume",
                "aggressive_buy_proxy",
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
    print("[Aggressive-Buy Feature] PASS")
    print(
        "[Success] The hourly aggressive-buy proxy was "
        "constructed from aggregated taker-buy and total "
        "base-asset volumes available at each forecast origin."
    )


if __name__ == "__main__":
    main()