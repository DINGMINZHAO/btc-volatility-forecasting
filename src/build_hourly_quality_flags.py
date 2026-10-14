from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KLINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_2022_2025.parquet"
)

HOURLY_RV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_realized_variance_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "btcusdt_hourly_quality_flags_2022_2025.parquet"
)

AFFECTED_HOURS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "zero_activity_affected_hours.csv"
)

EXPECTED_BARS_PER_HOUR = 12


def identify_zero_activity_bars(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify five-minute Klines with:

    1. identical OHLC prices,
    2. zero trading volume,
    3. zero trade count.

    Each bar is labeled by its ending timestamp because hourly
    realized variance is also constructed using return-end times.
    """
    data = data.copy()

    identical_ohlc = (
        (data["open"] == data["high"])
        & (data["open"] == data["low"])
        & (data["open"] == data["close"])
    )

    zero_volume = (
        (data["volume"] == 0)
        & (data["quote_volume"] == 0)
        & (data["taker_buy_volume"] == 0)
        & (data["taker_buy_quote_volume"] == 0)
    )

    zero_trade_count = data["count"] == 0

    data["zero_activity"] = (
        identical_ohlc
        & zero_volume
        & zero_trade_count
    )

    data["bar_end_utc"] = (
        data["open_datetime_utc"]
        + pd.Timedelta(minutes=5)
    )

    return data


def aggregate_hourly_flags(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate five-minute quality indicators into hourly bins.

    The bin (20:00, 21:00] contains bars ending at
    20:05, 20:10, ..., 21:00.
    """
    hourly_flags = (
        data
        .set_index("bar_end_utc")
        .resample(
            "1h",
            closed="right",
            label="right",
        )
        .agg(
            total_5m_bars=(
                "zero_activity",
                "size",
            ),
            zero_activity_bars=(
                "zero_activity",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "bar_end_utc": "hour_end_utc",
            }
        )
    )

    hourly_flags["zero_activity_bars"] = (
        hourly_flags["zero_activity_bars"]
        .astype("int64")
    )

    hourly_flags["active_5m_bars"] = (
        hourly_flags["total_5m_bars"]
        - hourly_flags["zero_activity_bars"]
    )

    hourly_flags["zero_activity_share"] = (
        hourly_flags["zero_activity_bars"]
        / hourly_flags["total_5m_bars"]
    )

    hourly_flags["has_zero_activity"] = (
        hourly_flags["zero_activity_bars"] > 0
    )

    return hourly_flags


def save_parquet(data: pd.DataFrame) -> None:
    """
    Save the quality-flag dataset safely.
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


def main() -> None:
    print("=" * 80)
    print("Build Hourly Zero-Activity Quality Flags")
    print("=" * 80)
    print(f"Kline file:     {KLINE_PATH}")
    print(f"Hourly RV file: {HOURLY_RV_PATH}")
    print()

    if not KLINE_PATH.exists():
        raise FileNotFoundError(
            f"Kline dataset not found:\n{KLINE_PATH}"
        )

    if not HOURLY_RV_PATH.exists():
        raise FileNotFoundError(
            f"Hourly RV dataset not found:\n{HOURLY_RV_PATH}"
        )

    kline_columns = [
        "open_datetime_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ]

    klines = pd.read_parquet(
        KLINE_PATH,
        columns=kline_columns,
    )

    hourly_rv = pd.read_parquet(
        HOURLY_RV_PATH,
    )

    klines = (
        klines
        .sort_values("open_datetime_utc")
        .reset_index(drop=True)
    )

    klines = identify_zero_activity_bars(klines)

    total_zero_activity_bars = int(
        klines["zero_activity"].sum()
    )

    hourly_flags = aggregate_hourly_flags(klines)

    merged = hourly_rv.merge(
        hourly_flags,
        on="hour_end_utc",
        how="left",
        validate="one_to_one",
    )

    flag_columns = [
        "total_5m_bars",
        "zero_activity_bars",
        "active_5m_bars",
        "zero_activity_share",
        "has_zero_activity",
    ]

    if merged[flag_columns].isna().any().any():
        raise RuntimeError(
            "Some hourly RV observations could not be matched "
            "to hourly quality flags."
        )

    invalid_bar_count_hours = merged.loc[
        merged["total_5m_bars"]
        != EXPECTED_BARS_PER_HOUR
    ]

    if not invalid_bar_count_hours.empty:
        raise RuntimeError(
            "At least one valid realized-variance hour does not "
            "contain exactly 12 five-minute Klines."
        )

    affected = (
        merged.loc[
            merged["has_zero_activity"]
        ]
        .sort_values("hour_end_utc")
        .reset_index(drop=True)
    )

    flagged_bars_in_rv_sample = int(
        merged["zero_activity_bars"].sum()
    )

    print("Quality-flag summary:")
    print(f"Hourly RV observations:           {len(merged)}")
    print(f"Total zero-activity 5m bars:      {total_zero_activity_bars}")
    print(f"Flagged bars in hourly RV sample: {flagged_bars_in_rv_sample}")
    print(f"Affected hourly observations:     {len(affected)}")
    print(
        f"Unaffected hourly observations:   "
        f"{int((~merged['has_zero_activity']).sum())}"
    )

    print()
    print("Affected hourly observations:")

    display_columns = [
        "hour_start_utc",
        "hour_end_utc",
        "zero_activity_bars",
        "active_5m_bars",
        "zero_activity_share",
        "realized_variance",
    ]

    print(
        affected[display_columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.10e}",
        )
    )

    if flagged_bars_in_rv_sample != total_zero_activity_bars:
        raise RuntimeError(
            "The number of zero-activity bars in the hourly "
            "sample does not match the full Kline dataset."
        )

    save_parquet(merged)

    affected[display_columns].to_csv(
        AFFECTED_HOURS_PATH,
        index=False,
    )

    print()
    print(f"Quality flags saved to: {OUTPUT_PATH}")
    print(f"Affected hours saved to: {AFFECTED_HOURS_PATH}")

    print()
    print("[Hourly Quality Flags] PASS")
    print(
        "[Success] Every hourly observation containing at least "
        "one zero-activity Kline has been identified."
    )


if __name__ == "__main__":
    main()