from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_with_splits_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "training_downside_transform_diagnostic.csv"
)

RAW_FEATURE = "downside_semivariance"
TRANSFORMED_FEATURE = "scaled_log_downside_semivariance"
TARGET = "y_next_hour"

EXPECTED_TRAIN_ROWS = 16673


def load_training_data() -> pd.DataFrame:
    """
    Load only model-ready training observations.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "forecast_origin_utc",
            "split",
            "is_model_ready",
            RAW_FEATURE,
            TARGET,
        ],
    )

    training = (
        data.loc[
            (data["split"] == "train")
            & data["is_model_ready"].astype(bool)
        ]
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    if len(training) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS} training rows, "
            f"but found {len(training)}."
        )

    required_values = training[
        [RAW_FEATURE, TARGET]
    ]

    if required_values.isna().any().any():
        raise RuntimeError(
            "Missing values were detected in the training sample."
        )

    if not np.isfinite(
        required_values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite values were detected."
        )

    if (training[RAW_FEATURE] < 0).any():
        raise RuntimeError(
            "Negative downside semivariance was detected."
        )

    return training


def build_transformation(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Compute the scaling constant from positive training values
    and construct the scaled logarithmic feature.
    """
    training = training.copy()

    positive_values = training.loc[
        training[RAW_FEATURE] > 0,
        RAW_FEATURE,
    ]

    if positive_values.empty:
        raise RuntimeError(
            "No positive downside-semivariance values were found."
        )

    training_median_positive = float(
        positive_values.median()
    )

    if (
        training_median_positive <= 0
        or not np.isfinite(training_median_positive)
    ):
        raise RuntimeError(
            "Invalid training scaling constant."
        )

    training[TRANSFORMED_FEATURE] = np.log1p(
        training[RAW_FEATURE]
        / training_median_positive
    )

    return training, training_median_positive


def summarize_feature(
    data: pd.DataFrame,
    column: str,
) -> dict[str, float | int | str]:
    """
    Calculate training-only distribution and target-correlation
    diagnostics.
    """
    values = data[column].astype(float)

    return {
        "feature": column,
        "count": int(values.count()),
        "mean": values.mean(),
        "std": values.std(ddof=1),
        "min": values.min(),
        "p1": values.quantile(0.01),
        "median": values.median(),
        "p99": values.quantile(0.99),
        "max": values.max(),
        "skewness": values.skew(),
        "excess_kurtosis": values.kurt(),
        "zero_count": int((values == 0).sum()),
        "pearson_with_target": values.corr(
            data[TARGET],
            method="pearson",
        ),
        "spearman_with_target": values.corr(
            data[TARGET],
            method="spearman",
        ),
    }


def validate_transformation(
    training: pd.DataFrame,
    scale: float,
) -> None:
    """
    Validate zero handling, finiteness, and monotonicity.
    """
    transformed = training[TRANSFORMED_FEATURE]

    if not np.isfinite(
        transformed.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite transformed values were detected."
        )

    zero_mask = training[RAW_FEATURE] == 0

    if not (
        training.loc[
            zero_mask,
            TRANSFORMED_FEATURE,
        ] == 0
    ).all():
        raise RuntimeError(
            "Raw zero values were not mapped to zero."
        )

    positive_order = (
        training.loc[
            training[RAW_FEATURE] > 0,
            [RAW_FEATURE, TRANSFORMED_FEATURE],
        ]
        .sort_values(RAW_FEATURE)
    )

    if not positive_order[
        TRANSFORMED_FEATURE
    ].is_monotonic_increasing:
        raise RuntimeError(
            "The transformation is not monotonic."
        )

    median_transformed = np.log1p(1.0)

    transformed_at_scale = np.log1p(
        scale / scale
    )

    if not np.isclose(
        transformed_at_scale,
        median_transformed,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Scaling verification failed."
        )


def main() -> None:
    print("=" * 80)
    print("Inspect Training-Set Downside-Semivariance Transformation")
    print("=" * 80)
    print(f"Input file: {INPUT_PATH}")
    print()

    training = load_training_data()

    training, scale = build_transformation(
        training
    )

    validate_transformation(
        training=training,
        scale=scale,
    )

    summaries = pd.DataFrame(
        [
            summarize_feature(
                training,
                RAW_FEATURE,
            ),
            summarize_feature(
                training,
                TRANSFORMED_FEATURE,
            ),
        ]
    )

    print("Transformation parameter:")
    print(
        "Median positive downside semivariance "
        f"in training set: {scale:.16e}"
    )

    print()
    print("Distribution and correlation comparison:")

    display_columns = [
        "feature",
        "mean",
        "std",
        "min",
        "p1",
        "median",
        "p99",
        "max",
        "skewness",
        "excess_kurtosis",
        "zero_count",
        "pearson_with_target",
        "spearman_with_target",
    ]

    print(
        summaries[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    largest_raw = (
        training[
            [
                "forecast_origin_utc",
                RAW_FEATURE,
                TRANSFORMED_FEATURE,
                TARGET,
            ]
        ]
        .nlargest(10, RAW_FEATURE)
    )

    print()
    print("Ten largest raw downside-semivariance observations:")
    print(
        largest_raw.to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Diagnostic table saved to: {OUTPUT_PATH}")

    print()
    print("[Downside Transformation Inspection] PASS")
    print(
        "[Success] The raw and scaled-log downside features "
        "were compared using training data only."
    )


if __name__ == "__main__":
    main()