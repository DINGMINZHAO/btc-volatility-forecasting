from __future__ import annotations

from pathlib import Path

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
    / "monthly_file_audit.csv"
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


def detect_timestamp_unit(timestamp_series: pd.Series) -> str:
    """
    Detect whether Unix timestamps are stored in milliseconds
    or microseconds.
    """
    maximum_timestamp = int(timestamp_series.abs().max())

    if maximum_timestamp < 10**14:
        return "ms"

    if maximum_timestamp < 10**17:
        return "us"

    raise ValueError(
        f"Unsupported timestamp magnitude: {maximum_timestamp}"
    )


def inspect_one_month(
    csv_path: Path,
    year_month: str,
) -> dict[str, object]:
    """
    Inspect the schema and timestamp completeness of one monthly CSV.
    """
    data = pd.read_csv(
        csv_path,
        low_memory=False,
    )

    issues: list[str] = []

    actual_columns = data.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        issues.append("Column schema mismatch.")

        return {
            "month": year_month,
            "status": "FAIL",
            "rows_expected": None,
            "rows_actual": len(data),
            "timestamp_unit": None,
            "duplicate_times": None,
            "missing_times": None,
            "unexpected_times": None,
            "first_open_utc": None,
            "last_open_utc": None,
            "last_close_utc": None,
            "issues": " | ".join(issues),
        }

    data["open_time"] = pd.to_numeric(
        data["open_time"],
        errors="raise",
    )

    data["close_time"] = pd.to_numeric(
        data["close_time"],
        errors="raise",
    )

    timestamp_unit = detect_timestamp_unit(
        data["open_time"]
    )

    data["open_datetime_utc"] = pd.to_datetime(
        data["open_time"],
        unit=timestamp_unit,
        utc=True,
        errors="raise",
    )

    data["close_datetime_utc"] = pd.to_datetime(
        data["close_time"],
        unit=timestamp_unit,
        utc=True,
        errors="raise",
    )

    data = data.sort_values(
        "open_datetime_utc"
    ).reset_index(drop=True)

    expected_start = pd.Timestamp(
        f"{year_month}-01",
        tz="UTC",
    )

    expected_next_month = (
        expected_start + pd.offsets.MonthBegin(1)
    )

    expected_index = pd.date_range(
        start=expected_start,
        end=expected_next_month,
        freq="5min",
        inclusive="left",
    )

    expected_last_open = (
        expected_next_month
        - pd.Timedelta(minutes=5)
    )

    if timestamp_unit == "ms":
        expected_last_close = (
            expected_next_month
            - pd.Timedelta(milliseconds=1)
        )
    else:
        expected_last_close = (
            expected_next_month
            - pd.Timedelta(microseconds=1)
        )

    actual_index = pd.DatetimeIndex(
        data["open_datetime_utc"]
    )

    rows_expected = len(expected_index)
    rows_actual = len(data)

    duplicate_times = int(
        data["open_time"].duplicated().sum()
    )

    missing_times = expected_index.difference(
        actual_index
    )

    unexpected_times = actual_index.difference(
        expected_index
    )

    first_open = data["open_datetime_utc"].iloc[0]
    last_open = data["open_datetime_utc"].iloc[-1]
    last_close = data["close_datetime_utc"].iloc[-1]

    if rows_actual != rows_expected:
        issues.append(
            f"Expected {rows_expected} rows, found {rows_actual}."
        )

    if first_open != expected_start:
        issues.append(
            f"Incorrect first open time: {first_open}."
        )

    if last_open != expected_last_open:
        issues.append(
            f"Incorrect last open time: {last_open}."
        )

    if last_close != expected_last_close:
        issues.append(
            f"Incorrect last close time: {last_close}."
        )

    if duplicate_times > 0:
        issues.append(
            f"Duplicate open times: {duplicate_times}."
        )

    if len(missing_times) > 0:
        issues.append(
            f"Missing five-minute periods: "
            f"{len(missing_times)}."
        )

    if len(unexpected_times) > 0:
        issues.append(
            f"Unexpected open times: "
            f"{len(unexpected_times)}."
        )

    status = "PASS" if not issues else "FAIL"

    return {
        "month": year_month,
        "status": status,
        "rows_expected": rows_expected,
        "rows_actual": rows_actual,
        "timestamp_unit": timestamp_unit,
        "duplicate_times": duplicate_times,
        "missing_times": len(missing_times),
        "unexpected_times": len(unexpected_times),
        "first_open_utc": first_open,
        "last_open_utc": last_open,
        "last_close_utc": last_close,
        "issues": " | ".join(issues),
    }


def main() -> None:
    months = generate_months(
        start_month=START_MONTH,
        end_month=END_MONTH,
    )

    print("=" * 80)
    print("Inspect All Monthly Binance Kline Files")
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
                "rows_expected": None,
                "rows_actual": None,
                "timestamp_unit": None,
                "duplicate_times": None,
                "missing_times": None,
                "unexpected_times": None,
                "first_open_utc": None,
                "last_open_utc": None,
                "last_close_utc": None,
                "issues": "CSV file not found.",
            }

            results.append(result)

            print(
                f"[{index:02d}/{len(months):02d}] "
                f"{year_month}  FAIL: file not found"
            )

            continue

        try:
            result = inspect_one_month(
                csv_path=csv_path,
                year_month=year_month,
            )
        except Exception as error:
            result = {
                "month": year_month,
                "status": "FAIL",
                "rows_expected": None,
                "rows_actual": None,
                "timestamp_unit": None,
                "duplicate_times": None,
                "missing_times": None,
                "unexpected_times": None,
                "first_open_utc": None,
                "last_open_utc": None,
                "last_close_utc": None,
                "issues": str(error),
            }

        results.append(result)

        print(
            f"[{index:02d}/{len(months):02d}] "
            f"{year_month}  "
            f"{result['status']}  "
            f"rows={result['rows_actual']}  "
            f"unit={result['timestamp_unit']}  "
            f"missing={result['missing_times']}  "
            f"duplicates={result['duplicate_times']}"
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

    total_rows = pd.to_numeric(
        audit["rows_actual"],
        errors="coerce",
    ).sum()

    timestamp_units = sorted(
        audit["timestamp_unit"]
        .dropna()
        .unique()
        .tolist()
    )

    print()
    print("=" * 80)
    print("Monthly File Audit Summary")
    print("=" * 80)
    print(f"Months inspected:     {len(audit)}")
    print(f"Months passed:        {passed_count}")
    print(f"Months failed:        {failed_count}")
    print(f"Total data rows:      {int(total_rows)}")
    print(f"Timestamp units seen: {timestamp_units}")
    print(f"Audit table saved to: {AUDIT_OUTPUT_PATH}")

    if failed_count > 0:
        print()
        print("Failed months:")

        failed_rows = audit.loc[
            audit["status"] == "FAIL",
            ["month", "issues"],
        ]

        print(
            failed_rows.to_string(index=False)
        )

        raise RuntimeError(
            "One or more monthly files failed inspection."
        )

    print()
    print("[Monthly File Audit] PASS")
    print(
        "[Success] All monthly files have consistent schemas "
        "and complete five-minute UTC timestamps."
    )


if __name__ == "__main__":
    main()