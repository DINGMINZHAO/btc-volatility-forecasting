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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "zero_activity_runs.csv"
)

FIVE_MINUTES = pd.Timedelta(minutes=5)


def main() -> None:
    print("=" * 80)
    print("Inspect Zero-Activity Five-Minute Kline Runs")
    print("=" * 80)
    print(f"Input file: {INPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    columns = [
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

    data = pd.read_parquet(
        INPUT_PATH,
        columns=columns,
    )

    data = (
        data
        .sort_values("open_datetime_utc")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Define a completely inactive five-minute Kline
    # --------------------------------------------------

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

    print("Five-minute observation counts:")
    print(f"Total Klines:                     {len(data)}")
    print(
        f"Identical-OHLC Klines:            "
        f"{int(identical_ohlc.sum())}"
    )
    print(
        f"Zero-volume Klines:               "
        f"{int(zero_volume.sum())}"
    )
    print(
        f"Zero-trade-count Klines:          "
        f"{int(zero_trade_count.sum())}"
    )
    print(
        f"Complete zero-activity Klines:    "
        f"{int(data['zero_activity'].sum())}"
    )

    inactive = data.loc[
        data["zero_activity"],
        [
            "open_datetime_utc",
            "open",
            "close",
        ],
    ].copy()

    if inactive.empty:
        print()
        print("[Zero-Activity Audit] No inactive Klines found.")
        return

    # A new run begins whenever the previous inactive Kline
    # is not exactly five minutes earlier.
    inactive["new_run"] = (
        inactive["open_datetime_utc"]
        .diff()
        .ne(FIVE_MINUTES)
    )

    inactive["run_id"] = (
        inactive["new_run"]
        .cumsum()
    )

    runs = (
        inactive
        .groupby("run_id", as_index=False)
        .agg(
            run_start_utc=(
                "open_datetime_utc",
                "min",
            ),
            last_kline_open_utc=(
                "open_datetime_utc",
                "max",
            ),
            n_5m_klines=(
                "open_datetime_utc",
                "size",
            ),
            unique_prices=(
                "close",
                "nunique",
            ),
            first_price=(
                "close",
                "first",
            ),
            last_price=(
                "close",
                "last",
            ),
        )
    )

    runs["run_end_utc"] = (
        runs["last_kline_open_utc"]
        + FIVE_MINUTES
    )

    runs["duration_minutes"] = (
        runs["n_5m_klines"] * 5
    )

    runs = runs[
        [
            "run_start_utc",
            "run_end_utc",
            "duration_minutes",
            "n_5m_klines",
            "unique_prices",
            "first_price",
            "last_price",
        ]
    ].sort_values(
        "run_start_utc"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Zero-activity runs:")
    print(
        runs.to_string(index=False)
    )

    print()
    print("Run summary:")
    print(f"Number of runs:      {len(runs)}")
    print(
        f"Total inactive bars: "
        f"{int(runs['n_5m_klines'].sum())}"
    )
    print(
        f"Longest run:         "
        f"{int(runs['duration_minutes'].max())} minutes"
    )
    print(f"Saved to:            {OUTPUT_PATH}")

    print()
    print("[Zero-Activity Audit] PASS")


if __name__ == "__main__":
    main()