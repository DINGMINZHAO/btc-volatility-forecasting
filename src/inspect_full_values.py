from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

START_MONTH = "2022-01"
END_MONTH = "2025-12"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "binance"
    / "futures"
    / "um"
    / "monthly"
    / "klines"
    / SYMBOL
    / INTERVAL
)

AUDIT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "full_value_audit.csv"
)

EXPECTED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

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

NUMERIC_COLUMNS = EXPECTED_COLUMNS.copy()


def generate_months(
    start_month: str,
    end_month: str,
) -> list[str]:
    """
    Generate all YYYY-MM values between two months,
    including both endpoints.
    """
    start_year, start_month_number = map(
        int,
        start_month.split("-"),
    )

    end_year, end_month_number = map(
        int,
        end_month.split("-"),
    )

    months: list[str] = []

    year = start_year
    month = start_month_number

    while (year, month) <= (end_year, end_month_number):
        months.append(f"{year:04d}-{month:02d}")

        month += 1

        if month == 13:
            month = 1
            year += 1

    return months


def inspect_one_month(
    csv_path: Path,
    year_month: str,
) -> dict[str, object]:
    """
    Inspect numeric and logical consistency for one monthly CSV.
    """
    data = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    if data.columns.tolist() != EXPECTED_COLUMNS:
        return {
            "month": year_month,
            "status": "FAIL",
            "rows": len(data),
            "missing_values": None,
            "infinite_values": None,
            "nonpositive_price_rows": None,
            "invalid_ohlc_rows": None,
            "negative_volume_rows": None,
            "zero_volume_rows": None,
            "negative_count_rows": None,
            "zero_count_rows": None,
            "noninteger_count_rows": None,
            "taker_base_above_total_rows": None,
            "taker_quote_above_total_rows": None,
            "implied_price_outside_range_rows": None,
            "ignore_values": None,
            "issues": "Column schema mismatch.",
        }

    # A non-numeric damaged value raises an exception immediately.
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    missing_values = int(
        data[NUMERIC_COLUMNS].isna().sum().sum()
    )

    numeric_array = data[NUMERIC_COLUMNS].to_numpy(
        dtype=float
    )

    infinite_values = int(
        np.isinf(numeric_array).sum()
    )

    # --------------------------------------------------
    # Price checks
    # --------------------------------------------------

    nonpositive_price_rows = int(
        (data[PRICE_COLUMNS] <= 0)
        .any(axis=1)
        .sum()
    )

    invalid_ohlc_mask = (
        (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["high"] < data["low"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    invalid_ohlc_rows = int(
        invalid_ohlc_mask.sum()
    )

    # --------------------------------------------------
    # Volume and trade-count checks
    # --------------------------------------------------

    negative_volume_rows = int(
        (data[VOLUME_COLUMNS] < 0)
        .any(axis=1)
        .sum()
    )

    zero_volume_rows = int(
        (data["volume"] == 0).sum()
    )

    negative_count_rows = int(
        (data["count"] < 0).sum()
    )

    zero_count_rows = int(
        (data["count"] == 0).sum()
    )

    noninteger_count_rows = int(
        (data["count"] % 1 != 0).sum()
    )

    # Small tolerances protect against harmless decimal rounding.
    base_volume_tolerance = 1e-10
    quote_volume_tolerance = 1e-6

    taker_base_above_total_rows = int(
        (
            data["taker_buy_volume"]
            > data["volume"] + base_volume_tolerance
        ).sum()
    )

    taker_quote_above_total_rows = int(
        (
            data["taker_buy_quote_volume"]
            > data["quote_volume"] + quote_volume_tolerance
        ).sum()
    )

    # --------------------------------------------------
    # Quote-volume consistency
    # --------------------------------------------------

    positive_volume_mask = data["volume"] > 0

    implied_average_price = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )

    implied_average_price.loc[positive_volume_mask] = (
        data.loc[positive_volume_mask, "quote_volume"]
        / data.loc[positive_volume_mask, "volume"]
    )

    # Relative tolerance for harmless floating-point rounding.
    price_tolerance = np.maximum(
        1e-8,
        data["high"].abs() * 1e-10,
    )

    implied_price_outside_mask = (
        positive_volume_mask
        & (
            (
                implied_average_price
                < data["low"] - price_tolerance
            )
            | (
                implied_average_price
                > data["high"] + price_tolerance
            )
        )
    )

    implied_price_outside_range_rows = int(
        implied_price_outside_mask.sum()
    )

    ignore_values = sorted(
        data["ignore"].dropna().unique().tolist()
    )

    # --------------------------------------------------
    # Final monthly status
    # --------------------------------------------------

    issues: list[str] = []

    if missing_values > 0:
        issues.append(
            f"Missing values: {missing_values}."
        )

    if infinite_values > 0:
        issues.append(
            f"Infinite values: {infinite_values}."
        )

    if nonpositive_price_rows > 0:
        issues.append(
            f"Non-positive price rows: "
            f"{nonpositive_price_rows}."
        )

    if invalid_ohlc_rows > 0:
        issues.append(
            f"Invalid OHLC rows: {invalid_ohlc_rows}."
        )

    if negative_volume_rows > 0:
        issues.append(
            f"Negative volume rows: {negative_volume_rows}."
        )

    if negative_count_rows > 0:
        issues.append(
            f"Negative trade-count rows: "
            f"{negative_count_rows}."
        )

    if noninteger_count_rows > 0:
        issues.append(
            f"Non-integer trade-count rows: "
            f"{noninteger_count_rows}."
        )

    if taker_base_above_total_rows > 0:
        issues.append(
            "Taker buy base volume exceeded total volume "
            f"in {taker_base_above_total_rows} rows."
        )

    if taker_quote_above_total_rows > 0:
        issues.append(
            "Taker buy quote volume exceeded total quote volume "
            f"in {taker_quote_above_total_rows} rows."
        )

    if implied_price_outside_range_rows > 0:
        issues.append(
            "Implied average price was outside the OHLC range "
            f"in {implied_price_outside_range_rows} rows."
        )

    status = "PASS" if not issues else "FAIL"

    return {
        "month": year_month,
        "status": status,
        "rows": len(data),
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "nonpositive_price_rows": nonpositive_price_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "negative_volume_rows": negative_volume_rows,
        "zero_volume_rows": zero_volume_rows,
        "negative_count_rows": negative_count_rows,
        "zero_count_rows": zero_count_rows,
        "noninteger_count_rows": noninteger_count_rows,
        "taker_base_above_total_rows": (
            taker_base_above_total_rows
        ),
        "taker_quote_above_total_rows": (
            taker_quote_above_total_rows
        ),
        "implied_price_outside_range_rows": (
            implied_price_outside_range_rows
        ),
        "ignore_values": ",".join(
            str(value)
            for value in ignore_values
        ),
        "issues": " | ".join(issues),
    }


def main() -> None:
    months = generate_months(
        start_month=START_MONTH,
        end_month=END_MONTH,
    )

    print("=" * 80)
    print("Inspect Full Binance Kline Value Quality")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Months expected: {len(months)}")
    print()

    results: list[dict[str, object]] = []

    for index, year_month in enumerate(months, start=1):
        csv_path = (
            INPUT_DIR
            / f"{SYMBOL}-{INTERVAL}-{year_month}.csv"
        )

        if not csv_path.exists():
            result = {
                "month": year_month,
                "status": "FAIL",
                "rows": None,
                "missing_values": None,
                "infinite_values": None,
                "nonpositive_price_rows": None,
                "invalid_ohlc_rows": None,
                "negative_volume_rows": None,
                "zero_volume_rows": None,
                "negative_count_rows": None,
                "zero_count_rows": None,
                "noninteger_count_rows": None,
                "taker_base_above_total_rows": None,
                "taker_quote_above_total_rows": None,
                "implied_price_outside_range_rows": None,
                "ignore_values": None,
                "issues": "CSV file not found.",
            }
        else:
            try:
                result = inspect_one_month(
                    csv_path=csv_path,
                    year_month=year_month,
                )
            except Exception as error:
                result = {
                    "month": year_month,
                    "status": "FAIL",
                    "rows": None,
                    "missing_values": None,
                    "infinite_values": None,
                    "nonpositive_price_rows": None,
                    "invalid_ohlc_rows": None,
                    "negative_volume_rows": None,
                    "zero_volume_rows": None,
                    "negative_count_rows": None,
                    "zero_count_rows": None,
                    "noninteger_count_rows": None,
                    "taker_base_above_total_rows": None,
                    "taker_quote_above_total_rows": None,
                    "implied_price_outside_range_rows": None,
                    "ignore_values": None,
                    "issues": str(error),
                }

        results.append(result)

        print(
            f"[{index:02d}/{len(months):02d}] "
            f"{year_month}  "
            f"{result['status']}  "
            f"rows={result['rows']}  "
            f"missing={result['missing_values']}  "
            f"invalid_ohlc={result['invalid_ohlc_rows']}"
        )

    audit = pd.DataFrame(results)

    AUDIT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_OUTPUT_PATH,
        index=False,
    )

    passed_count = int(
        (audit["status"] == "PASS").sum()
    )

    failed_count = int(
        (audit["status"] == "FAIL").sum()
    )

    count_columns = [
        "rows",
        "missing_values",
        "infinite_values",
        "nonpositive_price_rows",
        "invalid_ohlc_rows",
        "negative_volume_rows",
        "zero_volume_rows",
        "negative_count_rows",
        "zero_count_rows",
        "noninteger_count_rows",
        "taker_base_above_total_rows",
        "taker_quote_above_total_rows",
        "implied_price_outside_range_rows",
    ]

    totals = (
        audit[count_columns]
        .apply(pd.to_numeric, errors="coerce")
        .sum()
    )

    ignore_values_seen = sorted(
        audit["ignore_values"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print()
    print("=" * 80)
    print("Full Value Audit Summary")
    print("=" * 80)
    print(f"Months inspected:                    {len(audit)}")
    print(f"Months passed:                       {passed_count}")
    print(f"Months failed:                       {failed_count}")
    print(f"Total rows:                          {int(totals['rows'])}")
    print(f"Missing values:                      {int(totals['missing_values'])}")
    print(f"Infinite values:                     {int(totals['infinite_values'])}")
    print(
        "Non-positive price rows:             "
        f"{int(totals['nonpositive_price_rows'])}"
    )
    print(
        "Invalid OHLC rows:                   "
        f"{int(totals['invalid_ohlc_rows'])}"
    )
    print(
        "Negative volume rows:                "
        f"{int(totals['negative_volume_rows'])}"
    )
    print(
        "Zero base-volume rows:               "
        f"{int(totals['zero_volume_rows'])}"
    )
    print(
        "Negative trade-count rows:           "
        f"{int(totals['negative_count_rows'])}"
    )
    print(
        "Zero trade-count rows:               "
        f"{int(totals['zero_count_rows'])}"
    )
    print(
        "Non-integer trade-count rows:        "
        f"{int(totals['noninteger_count_rows'])}"
    )
    print(
        "Taker base above total rows:         "
        f"{int(totals['taker_base_above_total_rows'])}"
    )
    print(
        "Taker quote above total rows:        "
        f"{int(totals['taker_quote_above_total_rows'])}"
    )
    print(
        "Implied price outside range rows:    "
        f"{int(totals['implied_price_outside_range_rows'])}"
    )
    print(f"Ignore values seen:                  {ignore_values_seen}")
    print(f"Audit table saved to:                {AUDIT_OUTPUT_PATH}")

    if failed_count > 0:
        print()
        print("Failed months:")

        print(
            audit.loc[
                audit["status"] == "FAIL",
                ["month", "issues"],
            ].to_string(index=False)
        )

        raise RuntimeError(
            "One or more monthly files failed value inspection."
        )

    print()
    print("[Full Value Audit] PASS")
    print(
        "[Success] All 420,768 Kline records passed "
        "the full numeric and logical quality checks."
    )


if __name__ == "__main__":
    main()