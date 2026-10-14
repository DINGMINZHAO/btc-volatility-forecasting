from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_harx_features_complete_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_with_splits_2022_2025.parquet"
)

TRAIN_START = pd.Timestamp(
    "2022-01-01 00:00:00",
    tz="UTC",
)

VALIDATION_START = pd.Timestamp(
    "2024-01-01 00:00:00",
    tz="UTC",
)

TEST_START = pd.Timestamp(
    "2025-01-01 00:00:00",
    tz="UTC",
)

TEST_END = pd.Timestamp(
    "2026-01-01 00:00:00",
    tz="UTC",
)

EXPECTED_TOTAL_ROWS = 35063
EXPECTED_TRAIN_ROWS = 17518
EXPECTED_VALIDATION_ROWS = 8784
EXPECTED_TEST_ROWS = 8760
EXPECTED_NOT_USED_ROWS = 1


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the complete HAR-X feature table before assigning
    chronological sample splits.
    """
    required_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "y_current",
        "y_next_hour",
        "har_1h",
        "har_6h",
        "har_24h",
        "har_168h",
        "downside_semivariance",
        "log1p_number_of_trades",
        "aggressive_buy_proxy",
        "is_valid_complete_harx_pair",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = (
        data
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    if len(data) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS} rows, "
            f"but found {len(data)}."
        )

    duplicate_count = int(
        data["forecast_origin_utc"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate forecast origins detected: "
            f"{duplicate_count}"
        )

    time_differences = (
        data["forecast_origin_utc"]
        .diff()
        .dropna()
    )

    irregular_gap_count = int(
        (
            time_differences
            != pd.Timedelta(hours=1)
        ).sum()
    )

    if irregular_gap_count > 0:
        raise ValueError(
            f"Irregular hourly gaps detected: "
            f"{irregular_gap_count}"
        )

    return data


def assign_time_splits(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign splits strictly according to forecast-origin time.

    No random shuffling is used.
    """
    data = data.copy()

    origin = data["forecast_origin_utc"]

    conditions = [
        (
            (origin >= TRAIN_START)
            & (origin < VALIDATION_START)
        ),
        (
            (origin >= VALIDATION_START)
            & (origin < TEST_START)
        ),
        (
            (origin >= TEST_START)
            & (origin < TEST_END)
        ),
    ]

    choices = [
        "train",
        "validation",
        "test",
    ]

    data["split"] = np.select(
        conditions,
        choices,
        default="not_used",
    )

    data["is_model_ready"] = (
        data["is_valid_complete_harx_pair"].astype(bool)
        & data["split"].isin(
            ["train", "validation", "test"]
        )
    )

    return data


def validate_splits(data: pd.DataFrame) -> None:
    """
    Confirm exact calendar boundaries and row counts.
    """
    expected_counts = {
        "train": EXPECTED_TRAIN_ROWS,
        "validation": EXPECTED_VALIDATION_ROWS,
        "test": EXPECTED_TEST_ROWS,
        "not_used": EXPECTED_NOT_USED_ROWS,
    }

    actual_counts = (
        data["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in expected_counts.items():
        actual_count = int(
            actual_counts.get(split_name, 0)
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Incorrect row count for {split_name}: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    split_boundaries = {
        "train": (
            TRAIN_START,
            VALIDATION_START,
        ),
        "validation": (
            VALIDATION_START,
            TEST_START,
        ),
        "test": (
            TEST_START,
            TEST_END,
        ),
    }

    for split_name, (
        expected_start,
        expected_end,
    ) in split_boundaries.items():
        split_rows = data.loc[
            data["split"] == split_name
        ]

        if split_rows.empty:
            raise RuntimeError(
                f"The {split_name} split is empty."
            )

        if (
            split_rows["forecast_origin_utc"].min()
            < expected_start
        ):
            raise RuntimeError(
                f"The {split_name} split begins too early."
            )

        if (
            split_rows["forecast_origin_utc"].max()
            >= expected_end
        ):
            raise RuntimeError(
                f"The {split_name} split ends too late."
            )

    not_used = data.loc[
        data["split"] == "not_used"
    ]

    if len(not_used) != 1:
        raise RuntimeError(
            "Expected exactly one unused final observation."
        )

    if (
        not_used["forecast_origin_utc"].iloc[0]
        != TEST_END
    ):
        raise RuntimeError(
            "The unused observation is not the final "
            "2026-01-01 forecast origin."
        )

    if not_used["y_next_hour"].notna().any():
        raise RuntimeError(
            "The unused final observation unexpectedly "
            "contains a future target."
        )

    model_ready = data.loc[
        data["is_model_ready"]
    ]

    feature_columns = [
        "har_1h",
        "har_6h",
        "har_24h",
        "har_168h",
        "downside_semivariance",
        "log1p_number_of_trades",
        "aggressive_buy_proxy",
        "y_next_hour",
    ]

    if model_ready[feature_columns].isna().any().any():
        raise RuntimeError(
            "Missing values were detected among model-ready rows."
        )

    if not np.isfinite(
        model_ready[feature_columns]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite values were detected among "
            "model-ready rows."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the complete feature table with fixed split labels.
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

    verification = pd.read_parquet(
        OUTPUT_PATH,
        columns=[
            "forecast_origin_utc",
            "split",
            "is_model_ready",
        ],
    )

    if len(verification) != len(data):
        raise RuntimeError(
            "Saved row count does not match the source table."
        )

    if int(
        verification["is_model_ready"].sum()
    ) != int(
        data["is_model_ready"].sum()
    ):
        raise RuntimeError(
            "Saved model-ready row count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Chronological Training, Validation, and Test Splits")
    print("=" * 80)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(data)}")

    data = validate_input(data)

    data = assign_time_splits(data)

    validate_splits(data)

    summary = (
        data
        .groupby("split", sort=False)
        .agg(
            total_rows=(
                "forecast_origin_utc",
                "size",
            ),
            model_ready_rows=(
                "is_model_ready",
                "sum",
            ),
            first_origin=(
                "forecast_origin_utc",
                "min",
            ),
            last_origin=(
                "forecast_origin_utc",
                "max",
            ),
        )
        .reset_index()
    )

    summary["excluded_rows"] = (
        summary["total_rows"]
        - summary["model_ready_rows"]
    )

    summary = summary[
        [
            "split",
            "total_rows",
            "model_ready_rows",
            "excluded_rows",
            "first_origin",
            "last_origin",
        ]
    ]

    print()
    print("Time-split summary:")
    print(
        summary.to_string(index=False)
    )

    print()
    print("Total model-ready observations:")
    print(
        int(data["is_model_ready"].sum())
    )

    save_dataset(data)

    print()
    print(f"Saved to: {OUTPUT_PATH}")

    print()
    print("[Time Splits] PASS")
    print(
        "[Success] Training, validation, and test periods "
        "were fixed chronologically without random shuffling."
    )


if __name__ == "__main__":
    main()