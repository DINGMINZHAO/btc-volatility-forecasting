from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_with_splits_2022_2025.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FEATURE_COLUMNS = [
    "har_1h",
    "har_6h",
    "har_24h",
    "har_168h",
    "downside_semivariance",
    "log1p_number_of_trades",
    "aggressive_buy_proxy",
]

TARGET_COLUMN = "y_next_hour"

EXPECTED_TRAIN_MODEL_READY_ROWS = 16673
EXTREME_ROWS_PER_FEATURE = 5


def load_training_data() -> pd.DataFrame:
    """
    Load only model-ready training observations.
    Validation and test observations are excluded before any
    distributional calculation.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "forecast_origin_utc",
        "split",
        "is_model_ready",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
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

    training = (
        data.loc[
            (data["split"] == "train")
            & data["is_model_ready"].astype(bool),
            required_columns,
        ]
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    if len(training) != EXPECTED_TRAIN_MODEL_READY_ROWS:
        raise RuntimeError(
            "Unexpected number of model-ready training rows. "
            f"Expected {EXPECTED_TRAIN_MODEL_READY_ROWS}, "
            f"found {len(training)}."
        )

    numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    if training[numeric_columns].isna().any().any():
        raise RuntimeError(
            "Missing values were detected in the "
            "model-ready training sample."
        )

    numeric_values = training[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise RuntimeError(
            "Non-finite values were detected in the "
            "training sample."
        )

    return training


def build_distribution_summary(
    training: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize feature distributions using training data only.

    Excess kurtosis is reported, so a normal distribution has
    excess kurtosis close to zero.
    """
    rows: list[dict[str, float | int | str]] = []

    for column in FEATURE_COLUMNS:
        series = training[column].astype(float)

        rows.append(
            {
                "feature": column,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(ddof=1),
                "min": series.min(),
                "p0_1": series.quantile(0.001),
                "p1": series.quantile(0.01),
                "median": series.median(),
                "p99": series.quantile(0.99),
                "p99_9": series.quantile(0.999),
                "max": series.max(),
                "skewness": series.skew(),
                "excess_kurtosis": series.kurt(),
                "zero_count": int((series == 0).sum()),
            }
        )

    return pd.DataFrame(rows)


def build_extreme_observation_table(
    training: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify the most extreme observations using robust distance:

        |x - median| / (1.4826 * MAD)

    This is only a diagnostic. No observation is removed.
    """
    extreme_tables: list[pd.DataFrame] = []

    for column in FEATURE_COLUMNS:
        values = training[column].astype(float)

        median = values.median()
        mad = (values - median).abs().median()
        robust_scale = 1.4826 * mad

        if robust_scale <= 0 or not np.isfinite(robust_scale):
            robust_scale = values.std(ddof=1)

        if robust_scale <= 0 or not np.isfinite(robust_scale):
            raise RuntimeError(
                f"Cannot calculate a robust scale for {column}."
            )

        robust_distance = (
            (values - median).abs()
            / robust_scale
        )

        feature_extremes = pd.DataFrame(
            {
                "feature": column,
                "forecast_origin_utc": training[
                    "forecast_origin_utc"
                ],
                "value": values,
                "median": median,
                "robust_distance": robust_distance,
                "y_next_hour": training[TARGET_COLUMN],
            }
        )

        feature_extremes = (
            feature_extremes
            .nlargest(
                EXTREME_ROWS_PER_FEATURE,
                "robust_distance",
            )
        )

        extreme_tables.append(feature_extremes)

    return (
        pd.concat(
            extreme_tables,
            ignore_index=True,
        )
        .sort_values(
            "robust_distance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_correlations(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate Pearson and Spearman correlations using the
    training sample only.
    """
    analysis_columns = (
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    pearson = training[
        analysis_columns
    ].corr(method="pearson")

    spearman = training[
        analysis_columns
    ].corr(method="spearman")

    return pearson, spearman


def strongest_feature_pairs(
    correlation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract unique predictor-predictor correlation pairs,
    ordered by absolute correlation.
    """
    rows: list[dict[str, float | str]] = []

    for feature_1, feature_2 in combinations(
        FEATURE_COLUMNS,
        2,
    ):
        correlation = correlation_matrix.loc[
            feature_1,
            feature_2,
        ]

        rows.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_target_correlations(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare linear and rank correlations between each feature
    and the next-hour target.
    """
    rows: list[dict[str, float | str]] = []

    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "pearson_with_target": pearson.loc[
                    feature,
                    TARGET_COLUMN,
                ],
                "spearman_with_target": spearman.loc[
                    feature,
                    TARGET_COLUMN,
                ],
            }
        )

    output = pd.DataFrame(rows)

    output["absolute_pearson"] = (
        output["pearson_with_target"].abs()
    )

    return (
        output
        .sort_values(
            "absolute_pearson",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_vif_table(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Calculate VIF after standardizing predictors using training
    means and standard deviations.

    Standardization is used here only for numerical diagnostics.
    It is not yet saved as a modeling transformation.
    """
    predictors = training[
        FEATURE_COLUMNS
    ].astype(float)

    means = predictors.mean()
    standard_deviations = predictors.std(ddof=0)

    zero_variance_features = (
        standard_deviations[
            standard_deviations == 0
        ].index.tolist()
    )

    if zero_variance_features:
        raise RuntimeError(
            "Zero-variance training features detected: "
            f"{zero_variance_features}"
        )

    standardized = (
        predictors - means
    ) / standard_deviations

    condition_number = float(
        np.linalg.cond(
            standardized.to_numpy(dtype=float)
        )
    )

    design_matrix = sm.add_constant(
        standardized,
        has_constant="add",
    )

    vif_rows: list[dict[str, float | str]] = []

    for column_index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        vif_value = variance_inflation_factor(
            design_matrix.to_numpy(dtype=float),
            column_index,
        )

        vif_rows.append(
            {
                "feature": feature,
                "vif": float(vif_value),
            }
        )

    vif_table = (
        pd.DataFrame(vif_rows)
        .sort_values(
            "vif",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return vif_table, condition_number


def save_outputs(
    distribution_summary: pd.DataFrame,
    extremes: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    target_correlations: pd.DataFrame,
    feature_pairs: pd.DataFrame,
    vif_table: pd.DataFrame,
) -> None:
    """
    Save all training-only diagnostic tables.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    distribution_summary.to_csv(
        OUTPUT_DIR / "training_feature_summary.csv",
        index=False,
    )

    extremes.to_csv(
        OUTPUT_DIR / "training_feature_extremes.csv",
        index=False,
    )

    pearson.to_csv(
        OUTPUT_DIR / "training_pearson_correlation.csv",
    )

    spearman.to_csv(
        OUTPUT_DIR / "training_spearman_correlation.csv",
    )

    target_correlations.to_csv(
        OUTPUT_DIR / "training_target_correlations.csv",
        index=False,
    )

    feature_pairs.to_csv(
        OUTPUT_DIR / "training_feature_correlation_pairs.csv",
        index=False,
    )

    vif_table.to_csv(
        OUTPUT_DIR / "training_feature_vif.csv",
        index=False,
    )


def main() -> None:
    print("=" * 85)
    print("Inspect Training-Set Feature Distributions and Multicollinearity")
    print("=" * 85)
    print(f"Input file: {INPUT_PATH}")
    print()

    training = load_training_data()

    print("Training sample:")
    print(f"Model-ready rows: {len(training)}")
    print(
        f"First origin:     "
        f"{training['forecast_origin_utc'].iloc[0]}"
    )
    print(
        f"Last origin:      "
        f"{training['forecast_origin_utc'].iloc[-1]}"
    )

    distribution_summary = (
        build_distribution_summary(training)
    )

    extremes = build_extreme_observation_table(
        training
    )

    pearson, spearman = build_correlations(
        training
    )

    target_correlations = build_target_correlations(
        pearson=pearson,
        spearman=spearman,
    )

    feature_pairs = strongest_feature_pairs(
        pearson
    )

    vif_table, condition_number = build_vif_table(
        training
    )

    print()
    print("Training feature distribution summary:")
    display_summary_columns = [
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
    ]

    print(
        distribution_summary[
            display_summary_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    print()
    print("Feature correlations with next-hour target:")
    print(
        target_correlations[
            [
                "feature",
                "pearson_with_target",
                "spearman_with_target",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("Strongest predictor-predictor Pearson correlations:")
    print(
        feature_pairs
        .head(10)
        [
            [
                "feature_1",
                "feature_2",
                "correlation",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("Variance Inflation Factors:")
    print(
        vif_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(
        "Condition number of standardized predictor matrix: "
        f"{condition_number:.6f}"
    )

    print()
    print("Most extreme training observations by robust distance:")
    print(
        extremes
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )

    save_outputs(
        distribution_summary=distribution_summary,
        extremes=extremes,
        pearson=pearson,
        spearman=spearman,
        target_correlations=target_correlations,
        feature_pairs=feature_pairs,
        vif_table=vif_table,
    )

    print()
    print(f"Diagnostic tables saved to: {OUTPUT_DIR}")

    print()
    print("[Training Feature Inspection] PASS")
    print(
        "[Success] Feature distributions, extreme observations, "
        "correlations, and multicollinearity were evaluated "
        "using the training set only."
    )


if __name__ == "__main__":
    main()