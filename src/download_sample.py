from __future__ import annotations

import hashlib
from pathlib import Path

import requests


# --------------------------------------------------
# Download configuration
# --------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
YEAR_MONTH = "2022-01"

BASE_URL = (
    "https://data.binance.vision/data/futures/um/"
    "monthly/klines"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
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

FILE_NAME = f"{SYMBOL}-{INTERVAL}-{YEAR_MONTH}.zip"
CHECKSUM_FILE_NAME = f"{FILE_NAME}.CHECKSUM"

FILE_URL = (
    f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{FILE_NAME}"
)

CHECKSUM_URL = (
    f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{CHECKSUM_FILE_NAME}"
)

CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_file(url: str, destination: Path) -> None:
    """
    Download one file safely.

    The file is first written as a temporary .part file.
    It is renamed only after the download finishes successfully.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[Skip] File already exists: {destination.name}")
        return

    temporary_path = destination.with_name(
        destination.name + ".part"
    )

    headers = {
        "User-Agent": "btc-volatility-course-project/1.0"
    }

    print(f"[Download] {url}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 120),
            headers=headers,
        ) as response:
            response.raise_for_status()

            downloaded_bytes = 0

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if chunk:
                        file.write(chunk)
                        downloaded_bytes += len(chunk)

        temporary_path.replace(destination)

        size_mb = downloaded_bytes / (1024**2)

        print(
            f"[Saved] {destination.name} "
            f"({size_mb:.2f} MB)"
        )

    except (requests.RequestException, OSError) as error:
        if temporary_path.exists():
            temporary_path.unlink()

        raise RuntimeError(
            f"Failed to download {url}\n"
            f"Reason: {error}"
        ) from error


def read_expected_checksum(checksum_path: Path) -> str:
    """
    Read the expected SHA-256 hash from Binance's CHECKSUM file.
    """
    checksum_text = checksum_path.read_text(
        encoding="utf-8"
    ).strip()

    if not checksum_text:
        raise ValueError("The CHECKSUM file is empty.")

    expected_hash = checksum_text.split()[0].lower()

    if len(expected_hash) != 64:
        raise ValueError(
            "The CHECKSUM file does not contain "
            "a valid SHA-256 hash."
        )

    return expected_hash


def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a local file.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main() -> None:
    print("=" * 60)
    print("Binance BTCUSDT 5-Minute Kline Sample Download")
    print("=" * 60)

    print(f"Symbol:       {SYMBOL}")
    print(f"Interval:     {INTERVAL}")
    print(f"Sample month: {YEAR_MONTH}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print()

    checksum_path = OUTPUT_DIR / CHECKSUM_FILE_NAME
    zip_path = OUTPUT_DIR / FILE_NAME

    # Download the official checksum first.
    download_file(CHECKSUM_URL, checksum_path)

    # Download the ZIP archive.
    download_file(FILE_URL, zip_path)

    # Verify that the ZIP file matches the official checksum.
    expected_hash = read_expected_checksum(checksum_path)
    actual_hash = calculate_sha256(zip_path)

    print()
    print(f"Expected SHA-256: {expected_hash}")
    print(f"Actual SHA-256:   {actual_hash}")

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Checksum verification failed. "
            "The downloaded ZIP may be incomplete or corrupted."
        )

    print("[Checksum] PASS")
    print()
    print(f"ZIP file: {zip_path}")
    print(
        f"ZIP size: "
        f"{zip_path.stat().st_size / (1024**2):.2f} MB"
    )
    print()
    print(
        "[Success] The one-month sample was downloaded "
        "and verified successfully."
    )


if __name__ == "__main__":
    main()