from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEAR_MONTH = "2022-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = (
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
    / f"{SYMBOL}-{INTERVAL}-{YEAR_MONTH}.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "sample"
)


def extract_csv_from_zip(zip_path: Path, output_dir: Path) -> Path:
    """
    Inspect the ZIP archive and extract its single CSV file.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found:\n{zip_path}"
        )

    if zip_path.stat().st_size == 0:
        raise ValueError(
            f"ZIP file is empty:\n{zip_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(zip_path, "r") as archive:
            archive_members = archive.namelist()

            print("Files inside ZIP:")
            for member in archive_members:
                print(f"  - {member}")

            damaged_member = archive.testzip()

            if damaged_member is not None:
                raise RuntimeError(
                    f"CRC check failed for archive member: "
                    f"{damaged_member}"
                )

            csv_members = [
                member
                for member in archive_members
                if member.lower().endswith(".csv")
            ]

            if len(csv_members) != 1:
                raise RuntimeError(
                    "Expected exactly one CSV file inside the ZIP, "
                    f"but found {len(csv_members)}."
                )

            csv_member = csv_members[0]

            # Use only the filename to avoid extracting an unsafe path.
            output_path = output_dir / Path(csv_member).name

            with archive.open(csv_member) as source:
                with output_path.open("wb") as destination:
                    destination.write(source.read())

    except BadZipFile as error:
        raise RuntimeError(
            f"Invalid or damaged ZIP archive:\n{zip_path}"
        ) from error

    return output_path


def preview_csv(csv_path: Path) -> None:
    """
    Read only the first five rows without assuming column names.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Extracted CSV file not found:\n{csv_path}"
        )

    try:
        preview = pd.read_csv(
            csv_path,
            header=None,
            nrows=5,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to read CSV file:\n{csv_path}\n"
            f"Reason: {error}"
        ) from error

    print()
    print(f"Extracted CSV: {csv_path}")
    print(
        f"CSV size: "
        f"{csv_path.stat().st_size / (1024**2):.2f} MB"
    )
    print(f"Columns detected: {preview.shape[1]}")
    print(f"Preview rows read: {preview.shape[0]}")

    print()
    print("First five raw rows:")
    print(preview.to_string(index=False, header=False))


def main() -> None:
    print("=" * 60)
    print("Extract and Preview Binance Kline Sample")
    print("=" * 60)
    print(f"ZIP path: {ZIP_PATH}")
    print()

    csv_path = extract_csv_from_zip(
        zip_path=ZIP_PATH,
        output_dir=OUTPUT_DIR,
    )

    preview_csv(csv_path)

    print()
    print("[Success] CSV extraction and preview completed.")


if __name__ == "__main__":
    main()