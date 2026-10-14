from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

START_MONTH = "2022-01"
END_MONTH = "2025-12"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "binance"
    / "futures"
    / "um"
    / "monthly"
    / "klines"
    / SYMBOL
    / INTERVAL
)

OUTPUT_DIR = (
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


def generate_months(
    start_month: str,
    end_month: str,
) -> list[str]:
    """
    Generate all YYYY-MM values from start_month to end_month,
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


def extract_one_archive(
    zip_path: Path,
    output_dir: Path,
) -> bool:
    """
    Validate and extract one monthly ZIP archive.

    Returns:
        True if the CSV was newly extracted.
        False if an existing complete CSV was reused.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found:\n{zip_path}"
        )

    if zip_path.stat().st_size == 0:
        raise ValueError(
            f"ZIP file is empty:\n{zip_path}"
        )

    try:
        with ZipFile(zip_path, "r") as archive:
            damaged_member = archive.testzip()

            if damaged_member is not None:
                raise RuntimeError(
                    f"CRC check failed for: {damaged_member}"
                )

            csv_members = [
                member
                for member in archive.infolist()
                if not member.is_dir()
                and member.filename.lower().endswith(".csv")
            ]

            if len(csv_members) != 1:
                raise RuntimeError(
                    f"Expected exactly one CSV inside {zip_path.name}, "
                    f"but found {len(csv_members)}."
                )

            csv_info = csv_members[0]
            output_path = output_dir / Path(csv_info.filename).name

            output_dir.mkdir(parents=True, exist_ok=True)

            # If an existing CSV has exactly the expected uncompressed
            # size, reuse it.
            if (
                output_path.exists()
                and output_path.stat().st_size == csv_info.file_size
                and output_path.stat().st_size > 0
            ):
                return False

            temporary_path = output_path.with_name(
                output_path.name + ".part"
            )

            if temporary_path.exists():
                temporary_path.unlink()

            with archive.open(csv_info, "r") as source:
                with temporary_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

            if temporary_path.stat().st_size != csv_info.file_size:
                temporary_path.unlink(missing_ok=True)

                raise RuntimeError(
                    f"Extracted size does not match ZIP metadata:\n"
                    f"File: {csv_info.filename}\n"
                    f"Expected: {csv_info.file_size} bytes\n"
                    f"Actual: {temporary_path.stat().st_size} bytes"
                )

            temporary_path.replace(output_path)

            return True

    except BadZipFile as error:
        raise RuntimeError(
            f"Invalid ZIP archive:\n{zip_path}"
        ) from error


def main() -> None:
    months = generate_months(
        start_month=START_MONTH,
        end_month=END_MONTH,
    )

    print("=" * 70)
    print("Extract Complete Binance BTCUSDT 5-Minute Kline Data")
    print("=" * 70)
    print(f"Start month:     {START_MONTH}")
    print(f"End month:       {END_MONTH}")
    print(f"Months expected: {len(months)}")
    print(f"Raw directory:   {RAW_DIR}")
    print(f"Output directory:{OUTPUT_DIR}")
    print()

    extracted_count = 0
    existing_count = 0

    for index, year_month in enumerate(months, start=1):
        zip_name = f"{SYMBOL}-{INTERVAL}-{year_month}.zip"
        zip_path = RAW_DIR / zip_name

        print(
            f"[{index:02d}/{len(months):02d}] "
            f"Extracting {year_month}..."
        )

        try:
            extracted = extract_one_archive(
                zip_path=zip_path,
                output_dir=OUTPUT_DIR,
            )
        except Exception as error:
            print()
            print("[Extraction Failed]")
            print(f"Month:  {year_month}")
            print(f"Reason: {error}")
            raise

        if extracted:
            extracted_count += 1
            print("         PASS: newly extracted")
        else:
            existing_count += 1
            print("         PASS: existing complete CSV reused")

    expected_csv_names = {
        f"{SYMBOL}-{INTERVAL}-{month}.csv"
        for month in months
    }

    actual_csv_paths = sorted(OUTPUT_DIR.glob("*.csv"))
    actual_csv_names = {
        path.name
        for path in actual_csv_paths
    }

    missing_csv_names = sorted(
        expected_csv_names - actual_csv_names
    )

    total_size_bytes = sum(
        path.stat().st_size
        for path in actual_csv_paths
    )

    print()
    print("=" * 70)
    print("Extraction Summary")
    print("=" * 70)
    print(f"Months requested:       {len(months)}")
    print(f"CSV files extracted:    {extracted_count}")
    print(f"Existing CSV files:     {existing_count}")
    print(f"CSV files in directory: {len(actual_csv_paths)}")
    print(
        f"Total extracted size:   "
        f"{total_size_bytes / (1024**2):.2f} MB"
    )
    print(f"Missing expected CSVs:  {len(missing_csv_names)}")

    if missing_csv_names:
        print()
        print("Missing CSV files:")

        for name in missing_csv_names:
            print(f"  - {name}")

        raise RuntimeError(
            "One or more expected monthly CSV files are missing."
        )

    if len(actual_csv_paths) != len(months):
        raise RuntimeError(
            "The number of extracted CSV files is not equal "
            "to the number of requested months."
        )

    print()
    print("[Full Extraction] PASS")
    print(
        "[Success] All monthly CSV files were extracted "
        "and validated successfully."
    )


if __name__ == "__main__":
    main()