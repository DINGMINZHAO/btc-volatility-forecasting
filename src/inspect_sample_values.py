from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEAR_MONTH = "2022-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sample"
    / f"{SYMBOL}-{INTERVAL}-{YEAR_MONTH}.csv"
)

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

VOLUME_COLUMNS = [
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

REQUIRED_COLUMNS = (
    ["open_time", "close_time"]
    + PRICE_COLUMNS
    + VOLUME_COLUMNS
    + ["count", "ignore"]
)


def main() -> None:
    print("=" * 70)
    print("Inspect Binance Kline Sample Values")
    print("=" * 70)
    print(f"CSV path: {CSV_PATH}")
    print()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert every required field to numeric.
    # Any non-numeric damaged value will cause an error.
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    print(f"Number of rows: {len(data)}")

    # --------------------------------------------------
    # 1. Missing and infinite values
    # --------------------------------------------------

    missing_counts = data[REQUIRED_COLUMNS].isna().sum()

    numeric_array = data[REQUIRED_COLUMNS].to_numpy(
        dtype=float
    )
    infinite_count = int(np.isinf(numeric_array).sum())

    print()
    print("Missing-value counts:")
    print(missing_counts.to_string())

    print()
    print(f"Total infinite values: {infinite_count}")

    # --------------------------------------------------
    # 2. OHLC price logic
    # --------------------------------------------------

    nonpositive_price_rows = (
        data[PRICE_COLUMNS] <= 0
    ).any(axis=1)

    high_below_open = data["high"] < data["open"]
    high_below_close = data["high"] < data["close"]
    high_below_low = data["high"] < data["low"]

    low_above_open = data["low"] > data["open"]
    low_above_close = data["low"] > data["close"]

    invalid_ohlc_rows = (
        high_below_open
        | high_below_close
        | high_below_low
        | low_above_open
        | low_above_close
    )

    print()
    print("OHLC checks:")
    print(
        f"Rows with non-positive prices: "
        f"{int(nonpositive_price_rows.sum())}"
    )
    print(
        f"Rows with invalid OHLC ordering: "
        f"{int(invalid_ohlc_rows.sum())}"
    )

    # --------------------------------------------------
    # 3. Volume and trade-count logic
    # --------------------------------------------------

    negative_volume_rows = (
        data[VOLUME_COLUMNS] < 0
    ).any(axis=1)

    negative_trade_count_rows = data["count"] < 0

    noninteger_trade_count_rows = (
        data["count"] % 1 != 0
    )

    taker_volume_above_total = (
        data["taker_buy_volume"] > data["volume"]
    )

    taker_quote_above_total = (
        data["taker_buy_quote_volume"]
        > data["quote_volume"]
    )

    print()
    print("Trading-field checks:")
    print(
        f"Rows with negative volume fields: "
        f"{int(negative_volume_rows.sum())}"
    )
    print(
        f"Rows with negative trade counts: "
        f"{int(negative_trade_count_rows.sum())}"
    )
    print(
        f"Rows with non-integer trade counts: "
        f"{int(noninteger_trade_count_rows.sum())}"
    )
    print(
        f"Rows where taker buy volume exceeds total volume: "
        f"{int(taker_volume_above_total.sum())}"
    )
    print(
        f"Rows where taker buy quote volume exceeds "
        f"total quote volume: "
        f"{int(taker_quote_above_total.sum())}"
    )

    # --------------------------------------------------
    # 4. Quote-volume consistency check
    # --------------------------------------------------

    positive_volume = data["volume"] > 0

    implied_average_price = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )

    implied_average_price.loc[positive_volume] = (
        data.loc[positive_volume, "quote_volume"]
        / data.loc[positive_volume, "volume"]
    )

    implied_price_outside_range = (
        positive_volume
        & (
            (implied_average_price < data["low"])
            | (implied_average_price > data["high"])
        )
    )

    print()
    print("Quote-volume consistency:")
    print(
        f"Rows with positive base volume: "
        f"{int(positive_volume.sum())}"
    )
    print(
        f"Rows where implied average price is outside "
        f"the low-high range: "
        f"{int(implied_price_outside_range.sum())}"
    )

    # --------------------------------------------------
    # 5. Ignore field
    # --------------------------------------------------

    ignore_unique_values = sorted(
        data["ignore"].dropna().unique().tolist()
    )

    print()
    print(f"Unique values in ignore: {ignore_unique_values}")

    # --------------------------------------------------
    # 6. Basic ranges
    # --------------------------------------------------

    summary_columns = [
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

    print()
    print("Basic numeric ranges:")
    print(
        data[summary_columns]
        .agg(["min", "max", "mean"])
        .T
        .to_string()
    )

    # --------------------------------------------------
    # Final decision
    # --------------------------------------------------

    issues = []

    if int(missing_counts.sum()) > 0:
        issues.append("Missing values were detected.")

    if infinite_count > 0:
        issues.append("Infinite values were detected.")

    if nonpositive_price_rows.any():
        issues.append("Non-positive OHLC prices were detected.")

    if invalid_ohlc_rows.any():
        issues.append("Invalid OHLC ordering was detected.")

    if negative_volume_rows.any():
        issues.append("Negative volume values were detected.")

    if negative_trade_count_rows.any():
        issues.append("Negative trade counts were detected.")

    if noninteger_trade_count_rows.any():
        issues.append("Non-integer trade counts were detected.")

    if taker_volume_above_total.any():
        issues.append(
            "Taker buy volume exceeded total volume."
        )

    if taker_quote_above_total.any():
        issues.append(
            "Taker buy quote volume exceeded total quote volume."
        )

    if implied_price_outside_range.any():
        issues.append(
            "Implied average prices outside OHLC ranges "
            "were detected."
        )

    print()

    if issues:
        print("[Value Check] FAIL")

        for issue in issues:
            print(f"  - {issue}")

        raise RuntimeError(
            "The sample failed one or more value checks."
        )

    print("[Value Check] PASS")
    print(
        "[Success] No missing, infinite, logically invalid, "
        "or obviously corrupted Kline records were detected."
    )


if __name__ == "__main__":
    main()