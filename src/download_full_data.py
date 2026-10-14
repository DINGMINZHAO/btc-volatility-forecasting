from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# --------------------------------------------------
# Download configuration
# --------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

START_MONTH = "2022-01"
END_MONTH = "2025-12"

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

CHUNK_SIZE = 1024 * 1024  # 1 MB


def generate_months(
    start_month: str,
    end_month: str,
) -> Iterator[str]:
    """
    Generate every YYYY-MM month from start_month to end_month,
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

    year = start_year
    month = start_month_number

    while (year, month) <= (end_year, end_month_number):
        yield f"{year:04d}-{month:02d}"

        month += 1

        if month == 13:
            month = 1
            year += 1


def create_session() -> requests.Session:
    """
    Create an HTTP session with limited automatic retries.

    Retries are used only for temporary network/server errors.
    """
    retry_policy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_policy)

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "btc-volatility-course-project/1.0"
            )
        }
    )

    return session


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
) -> bool:
    """
    Download one file.

    Returns:
        True  if a new file was downloaded.
        False if an existing non-empty file was kept.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return False

    temporary_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        with session.get(
            url,
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()

            with temporary_path.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if chunk:
                        file.write(chunk)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError(
                f"Downloaded file is empty: {url}"
            )

        temporary_path.replace(destination)
        return True

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def read_expected_checksum(checksum_path: Path) -> str:
    """
    Read the expected SHA-256 hash from a Binance CHECKSUM file.
    """
    checksum_text = checksum_path.read_text(
        encoding="utf-8"
    ).strip()

    if not checksum_text:
        raise ValueError(
            f"Empty CHECKSUM file: {checksum_path}"
        )

    expected_hash = checksum_text.split()[0].lower()

    if len(expected_hash) != 64:
        raise ValueError(
            f"Invalid SHA-256 value in: {checksum_path}"
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


def process_one_month(
    session: requests.Session,
    year_month: str,
) -> tuple[bool, bool]:
    """
    Download and verify one monthly ZIP archive.

    Returns:
        checksum_downloaded, zip_downloaded
    """
    zip_name = f"{SYMBOL}-{INTERVAL}-{year_month}.zip"
    checksum_name = f"{zip_name}.CHECKSUM"

    month_url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}"

    zip_url = f"{month_url}/{zip_name}"
    checksum_url = f"{month_url}/{checksum_name}"

    zip_path = OUTPUT_DIR / zip_name
    checksum_path = OUTPUT_DIR / checksum_name

    checksum_downloaded = download_file(
        session=session,
        url=checksum_url,
        destination=checksum_path,
    )

    zip_downloaded = download_file(
        session=session,
        url=zip_url,
        destination=zip_path,
    )

    expected_hash = read_expected_checksum(checksum_path)
    actual_hash = calculate_sha256(zip_path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "\nChecksum verification failed.\n"
            f"Month:    {year_month}\n"
            f"ZIP file: {zip_path}\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return checksum_downloaded, zip_downloaded


def main() -> None:
    months = list(
        generate_months(
            start_month=START_MONTH,
            end_month=END_MONTH,
        )
    )

    print("=" * 70)
    print("Download Complete Binance BTCUSDT 5-Minute Kline Data")
    print("=" * 70)
    print(f"Symbol:       {SYMBOL}")
    print(f"Interval:     {INTERVAL}")
    print(f"Start month:  {START_MONTH}")
    print(f"End month:    {END_MONTH}")
    print(f"Total months: {len(months)}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print()

    session = create_session()

    downloaded_zip_count = 0
    existing_zip_count = 0
    verified_count = 0

    try:
        for index, year_month in enumerate(months, start=1):
            print(
                f"[{index:02d}/{len(months):02d}] "
                f"Processing {year_month}..."
            )

            try:
                _, zip_downloaded = process_one_month(
                    session=session,
                    year_month=year_month,
                )
            except Exception as error:
                print()
                print("[Download Failed]")
                print(f"Month: {year_month}")
                print(f"Reason: {error}")
                print()
                print(
                    "You may rerun the script after fixing the "
                    "problem. Already downloaded files will be reused."
                )
                raise

            verified_count += 1

            if zip_downloaded:
                downloaded_zip_count += 1
                status = "downloaded and verified"
            else:
                existing_zip_count += 1
                status = "already existed and verified"

            print(f"         PASS: {status}")

    finally:
        session.close()

    zip_files = sorted(OUTPUT_DIR.glob("*.zip"))
    checksum_files = sorted(OUTPUT_DIR.glob("*.zip.CHECKSUM"))

    print()
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Months requested:       {len(months)}")
    print(f"ZIP files downloaded:   {downloaded_zip_count}")
    print(f"Existing ZIP files:     {existing_zip_count}")
    print(f"ZIP files verified:     {verified_count}")
    print(f"ZIP files in directory: {len(zip_files)}")
    print(f"CHECKSUM files:          {len(checksum_files)}")

    if verified_count != len(months):
        raise RuntimeError(
            "Not all requested monthly files were verified."
        )

    if len(zip_files) < len(months):
        raise RuntimeError(
            "The output directory contains fewer ZIP files "
            "than expected."
        )

    print()
    print("[Full Download] PASS")
    print(
        "[Success] All requested monthly archives were "
        "downloaded and verified."
    )


if __name__ == "__main__":
    main()