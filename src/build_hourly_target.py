from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "btcusdt_hourly_quality_flags_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_target_2022_2025.parquet"
)

EXPECTED_RETURNS_PER_HOUR = 12
EXPECTED_BARS_PER_HOUR = 12


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate hourly realized variance and quality flags before
    constructing the log-realized-variance target.
    """
    required_columns = [
        "hour_start_utc",
        "hour_end_utc",
        "n_returns",
        "realized_variance",
        "total_5m_bars",
        "zero_activity_bars",
        "active_5m_bars",
        "zero_activity_share",
        "has_zero_activity",
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
        .sort_values("hour_end_utc")
        .reset_index(drop=True)
    )

    duplicate_count = int(
        data["hour_end_utc"].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate hourly timestamps detected: "
            f"{duplicate_count}"
        )

    hour_differences = (
        data["hour_end_utc"]
        .diff()
        .dropna()
    )

    irregular_gap_count = int(
        (
            hour_differences
            != pd.Timedelta(hours=1)
        ).sum()
    )

    if irregular_gap_count > 0:
        raise ValueError(
            f"Irregular hourly gaps detected: "
            f"{irregular_gap_count}"
        )

    return data


def construct_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct log realized variance for valid trading hours.

    Hours containing at least one completely inactive
    five-minute Kline are retained on the time grid but their
    target values are marked as missing.
    """
    data = data.copy()

    data["has_zero_activity"] = (
        data["has_zero_activity"].astype(bool)
    )

    realized_variance = pd.to_numeric(
        data["realized_variance"],
        errors="raise",
    )

    finite_rv = pd.Series(
        np.isfinite(
            realized_variance.to_numpy(dtype=float)
        ),
        index=data.index,
    )

    structurally_complete = (
        data["n_returns"].eq(
            EXPECTED_RETURNS_PER_HOUR
        )
        & data["total_5m_bars"].eq(
            EXPECTED_BARS_PER_HOUR
        )
    )

    positive_finite_rv = (
        realized_variance.gt(0)
        & finite_rv
    )

    data["is_valid_target"] = (
        structurally_complete
        & positive_finite_rv
        & ~data["has_zero_activity"]
    )

    # Detect unexpected invalid observations that were not
    # explained by the zero-activity quality flag.
    unexplained_invalid = (
        ~data["has_zero_activity"]
        & ~data["is_valid_target"]
    )

    if unexplained_invalid.any():
        problem_rows = data.loc[
            unexplained_invalid,
            [
                "hour_end_utc",
                "n_returns",
                "total_5m_bars",
                "realized_variance",
                "has_zero_activity",
            ],
        ]

        print()
        print("Unexpected invalid observations:")
        print(problem_rows.to_string(index=False))

        raise RuntimeError(
            "Some invalid targets are not explained by "
            "the zero-activity quality flag."
        )

    data["log_realized_variance"] = np.nan

    valid_mask = data["is_valid_target"]

    data.loc[
        valid_mask,
        "log_realized_variance",
    ] = np.log(
        data.loc[
            valid_mask,
            "realized_variance",
        ]
    )

    data["target_status"] = np.where(
        data["is_valid_target"],
        "valid",
        "invalid_zero_activity",
    )

    return data


def validate_target(data: pd.DataFrame) -> None:
    """
    Validate the completed target dataset.
    """
    valid_data = data.loc[
        data["is_valid_target"]
    ]

    invalid_data = data.loc[
        ~data["is_valid_target"]
    ]

    if valid_data["log_realized_variance"].isna().any():
        raise RuntimeError(
            "Missing log target values were found among "
            "valid observations."
        )

    if not np.isfinite(
        valid_data["log_realized_variance"]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite log target values were found."
        )

    if invalid_data[
        "log_realized_variance"
    ].notna().any():
        raise RuntimeError(
            "Invalid hours should not contain log target values."
        )

    if not invalid_data[
        "has_zero_activity"
    ].all():
        raise RuntimeError(
            "At least one invalid target is not associated "
            "with zero activity."
        )


def save_dataset(data: pd.DataFrame) -> None:
    """
    Save the hourly target dataset safely.
    """
    output_columns = [
        "hour_start_utc",
        "hour_end_utc",
        "realized_variance",
        "log_realized_variance",
        "n_returns",
        "total_5m_bars",
        "zero_activity_bars",
        "active_5m_bars",
        "zero_activity_share",
        "has_zero_activity",
        "is_valid_target",
        "target_status",
    ]

    output_data = data[output_columns].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    output_data.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_path.replace(OUTPUT_PATH)

    verification = pd.read_parquet(
        OUTPUT_PATH
    )

    if len(verification) != len(output_data):
        raise RuntimeError(
            "Saved dataset row count does not match "
            "the source dataset."
        )

    if int(
        verification[
            "log_realized_variance"
        ].isna().sum()
    ) != int(
        (~output_data["is_valid_target"]).sum()
    ):
        raise RuntimeError(
            "The saved target missing-value count is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Build Clean Hourly Log Realized-Variance Target")
    print("=" * 80)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Hourly observations loaded: {len(data)}")

    data = validate_input(data)

    data = construct_target(data)

    validate_target(data)

    valid_target = data.loc[
        data["is_valid_target"],
        "log_realized_variance",
    ]

    invalid_rows = data.loc[
        ~data["is_valid_target"],
        [
            "hour_start_utc",
            "hour_end_utc",
            "zero_activity_bars",
            "zero_activity_share",
            "realized_variance",
            "target_status",
        ],
    ]

    print()
    print("Target construction summary:")
    print(f"Total hourly observations: {len(data)}")
    print(
        f"Valid target observations: "
        f"{int(data['is_valid_target'].sum())}"
    )
    print(
        f"Invalid target observations: "
        f"{int((~data['is_valid_target']).sum())}"
    )
    print(
        f"Missing log target values: "
        f"{int(data['log_realized_variance'].isna().sum())}"
    )
    print("Epsilon added before logging: 0")

    print()
    print("Log realized-variance statistics:")
    print(
        valid_target.describe(
            percentiles=[
                0.01,
                0.50,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print()
    print("Invalid target observations:")
    print(
        invalid_rows.to_string(
            index=False,
            float_format=lambda value: f"{value:.10e}",
        )
    )

    save_dataset(data)

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024**2)
    )

    print()
    print(f"Parquet size: {file_size_mb:.2f} MB")
    print(f"Saved to:     {OUTPUT_PATH}")

    print()
    print("[Hourly Target] PASS")
    print(
        "[Success] Log realized variance was constructed "
        "without adding epsilon, while zero-activity hours "
        "were retained and marked as invalid."
    )


if __name__ == "__main__":
    main()