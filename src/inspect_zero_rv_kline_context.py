from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_5m_2022_2025.parquet"
)

ZERO_HOUR_START = pd.Timestamp(
    "2024-10-28 20:00:00",
    tz="UTC",
)

ZERO_HOUR_END = pd.Timestamp(
    "2024-10-28 21:00:00",
    tz="UTC",
)

CONTEXT_START = ZERO_HOUR_START - pd.Timedelta(minutes=10)
CONTEXT_END = ZERO_HOUR_END + pd.Timedelta(minutes=10)


def main() -> None:
    print("=" * 80)
    print("Inspect Kline Context Around Zero Realized Variance")
    print("=" * 80)
    print(f"Input file: {INPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    columns = [
        "open_datetime_utc",
        "close_datetime_utc",
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

    data = pd.read_parquet(
        INPUT_PATH,
        columns=columns,
    )

    context = data.loc[
        (data["open_datetime_utc"] >= CONTEXT_START)
        & (data["open_datetime_utc"] < CONTEXT_END)
    ].copy()

    if context.empty:
        raise RuntimeError(
            "No Kline observations were found in the selected period."
        )

    context = (
        context
        .sort_values("open_datetime_utc")
        .reset_index(drop=True)
    )

    context["price_range"] = (
        context["high"] - context["low"]
    )

    context["close_change"] = (
        context["close"].diff()
    )

    zero_hour = context.loc[
        (context["open_datetime_utc"] >= ZERO_HOUR_START)
        & (context["open_datetime_utc"] < ZERO_HOUR_END)
    ].copy()

    if len(zero_hour) != 12:
        raise RuntimeError(
            "Expected 12 Klines in the zero-RV hour, "
            f"but found {len(zero_hour)}."
        )

    print("Klines around the zero-RV hour:")
    print(
        context[
            [
                "open_datetime_utc",
                "open",
                "high",
                "low",
                "close",
                "price_range",
                "close_change",
                "volume",
                "count",
            ]
        ].to_string(index=False)
    )

    print()
    print("Zero-hour OHLC uniqueness:")
    print(f"Unique open prices:  {zero_hour['open'].nunique()}")
    print(f"Unique high prices:  {zero_hour['high'].nunique()}")
    print(f"Unique low prices:   {zero_hour['low'].nunique()}")
    print(f"Unique close prices: {zero_hour['close'].nunique()}")

    print()
    print("Zero-hour trading activity:")
    print(f"Total base volume:  {zero_hour['volume'].sum():.8f}")
    print(f"Total quote volume: {zero_hour['quote_volume'].sum():.8f}")
    print(f"Total trade count:  {int(zero_hour['count'].sum())}")
    print(
        f"Total taker buy volume: "
        f"{zero_hour['taker_buy_volume'].sum():.8f}"
    )

    print()
    print("Zero-hour price-range checks:")
    print(
        f"Klines with positive high-low range: "
        f"{int((zero_hour['price_range'] > 0).sum())}"
    )
    print(
        f"Maximum five-minute high-low range: "
        f"{zero_hour['price_range'].max():.8f}"
    )

    all_ohlc_identical = (
        (zero_hour["open"] == zero_hour["high"])
        & (zero_hour["open"] == zero_hour["low"])
        & (zero_hour["open"] == zero_hour["close"])
    ).all()

    all_volume_zero = (
        zero_hour["volume"] == 0
    ).all()

    all_count_zero = (
        zero_hour["count"] == 0
    ).all()

    print()
    print("Final indicators:")
    print(f"All OHLC identical: {all_ohlc_identical}")
    print(f"All volume zero:    {all_volume_zero}")
    print(f"All trade counts 0: {all_count_zero}")

    print()
    print("[Zero-Hour Kline Inspection] PASS")


if __name__ == "__main__":
    main()