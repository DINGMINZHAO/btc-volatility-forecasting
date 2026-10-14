from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_transformed_2022_2025.parquet"
)

STANDARDIZED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_standardized_2022_2025.parquet"
)

NAIVE_METRIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "naive_baseline_metrics_train_validation.csv"
)

AR_SELECTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "ar_order_selection_train_validation.csv"
)

PARAMETER_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "harx_standardization_parameters.csv"
)

METRIC_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "fixed_harx_metrics_train_validation.csv"
)

COEFFICIENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "fixed_harx_coefficients_hac24.csv"
)

RESIDUAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "fixed_harx_residual_summary.csv"
)

LJUNG_BOX_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "fixed_harx_ljung_box.csv"
)

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "fixed_harx_predictions_train_validation.parquet"
)

RAW_FEATURES = [
    "har_1h",
    "har_6h",
    "har_24h",
    "har_168h",
    "scaled_log_downside_semivariance",
    "log1p_number_of_trades",
    "aggressive_buy_proxy",
]

STANDARDIZED_FEATURES = [
    f"z_{feature}"
    for feature in RAW_FEATURES
]

EXPECTED_COUNTS = {
    "train": 16673,
    "validation": 8610,
}

HAC_MAX_LAGS = 24
LJUNG_BOX_LAGS = [24, 48, 168]


# --------------------------------------------------
# Metric functions
# --------------------------------------------------

def calculate_qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate QLIKE from log realized variance.
    """
    delta = y_true - y_pred

    qlike = np.expm1(delta) - delta

    return np.maximum(
        qlike,
        0.0,
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE and QLIKE.
    """
    errors = y_true - y_pred

    return {
        "mae_log_rv": float(
            np.mean(np.abs(errors))
        ),
        "rmse_log_rv": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "qlike": float(
            np.mean(
                calculate_qlike(
                    y_true=y_true,
                    y_pred=y_pred,
                )
            )
        ),
    }


# --------------------------------------------------
# Load and validate
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the complete model table.

    Test targets are not evaluated in this script.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    required_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "split",
        "is_model_ready",
        "y_next_hour",
        *RAW_FEATURES,
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

    if data["forecast_origin_utc"].duplicated().any():
        raise ValueError(
            "Duplicate forecast origins were detected."
        )

    time_differences = (
        data["forecast_origin_utc"]
        .diff()
        .dropna()
    )

    if not time_differences.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "The complete hourly time grid is irregular."
        )

    evaluation = data.loc[
        data["split"].isin(
            ["train", "validation"]
        )
        & data["is_model_ready"].astype(bool)
    ]

    counts = (
        evaluation["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        actual_count = int(
            counts.get(split_name, 0)
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected {split_name} count: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    numeric_columns = (
        RAW_FEATURES + ["y_next_hour"]
    )

    if evaluation[
        numeric_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing values were detected among "
            "model-ready observations."
        )

    if not np.isfinite(
        evaluation[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite values were detected."
        )

    return data


# --------------------------------------------------
# Training-based standardization
# --------------------------------------------------

def standardize_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate feature means and standard deviations from
    model-ready training observations only.

    Apply the same transformation to all periods.
    """
    data = data.copy()

    training = data.loc[
        (data["split"] == "train")
        & data["is_model_ready"].astype(bool)
    ]

    parameter_rows = []

    for raw_feature, standardized_feature in zip(
        RAW_FEATURES,
        STANDARDIZED_FEATURES,
    ):
        training_values = training[
            raw_feature
        ].astype(float)

        training_mean = float(
            training_values.mean()
        )

        training_std = float(
            training_values.std(ddof=0)
        )

        if (
            training_std <= 0
            or not np.isfinite(training_std)
        ):
            raise RuntimeError(
                f"Invalid training standard deviation "
                f"for {raw_feature}."
            )

        data[standardized_feature] = (
            data[raw_feature] - training_mean
        ) / training_std

        parameter_rows.append(
            {
                "raw_feature": raw_feature,
                "standardized_feature": (
                    standardized_feature
                ),
                "training_mean": training_mean,
                "training_std_ddof_0": training_std,
                "estimated_from_split": "train",
                "estimated_from_model_ready_only": True,
            }
        )

    parameters = pd.DataFrame(
        parameter_rows
    )

    standardized_training = data.loc[
        (data["split"] == "train")
        & data["is_model_ready"].astype(bool),
        STANDARDIZED_FEATURES,
    ]

    training_means = (
        standardized_training.mean()
    )

    training_stds = (
        standardized_training.std(ddof=0)
    )

    if not np.allclose(
        training_means.to_numpy(dtype=float),
        0.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Standardized training means are not zero."
        )

    if not np.allclose(
        training_stds.to_numpy(dtype=float),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Standardized training standard deviations "
            "are not one."
        )

    return data, parameters


# --------------------------------------------------
# Fixed HAR-X estimation
# --------------------------------------------------

def fit_fixed_harx(
    data: pd.DataFrame,
):
    """
    Fit the fixed-parameter HAR-X model by OLS using
    training data only.
    """
    training = (
        data.loc[
            (data["split"] == "train")
            & data["is_model_ready"].astype(bool)
        ]
        .copy()
    )

    validation = (
        data.loc[
            (data["split"] == "validation")
            & data["is_model_ready"].astype(bool)
        ]
        .copy()
    )

    x_train = sm.add_constant(
        training[
            STANDARDIZED_FEATURES
        ].astype(float),
        has_constant="add",
    )

    y_train = training[
        "y_next_hour"
    ].astype(float)

    fitted = sm.OLS(
        y_train,
        x_train,
    ).fit()

    # Newey-West/HAC inference.
    # Predictions still come from the original OLS coefficients.
    hac_result = fitted.get_robustcov_results(
        cov_type="HAC",
        maxlags=HAC_MAX_LAGS,
    )

    x_validation = sm.add_constant(
        validation[
            STANDARDIZED_FEATURES
        ].astype(float),
        has_constant="add",
    )

    training["y_pred_fixed_harx"] = np.asarray(
        fitted.predict(x_train),
        dtype=float,
    )

    validation["y_pred_fixed_harx"] = np.asarray(
        fitted.predict(x_validation),
        dtype=float,
    )

    predictions = pd.concat(
        [training, validation],
        ignore_index=True,
    )

    predictions = (
        predictions
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    predictions["y_true"] = (
        predictions["y_next_hour"]
    )

    predictions["residual"] = (
        predictions["y_true"]
        - predictions["y_pred_fixed_harx"]
    )

    predictions["absolute_log_error"] = (
        predictions["residual"].abs()
    )

    predictions["squared_log_error"] = (
        predictions["residual"] ** 2
    )

    predictions["qlike"] = calculate_qlike(
        y_true=predictions[
            "y_true"
        ].to_numpy(dtype=float),
        y_pred=predictions[
            "y_pred_fixed_harx"
        ].to_numpy(dtype=float),
    )

    return predictions, fitted, hac_result


def build_metric_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate training and validation metrics.
    """
    rows = []

    for split_name in [
        "train",
        "validation",
    ]:
        split_data = predictions.loc[
            predictions["split"] == split_name
        ]

        metrics = calculate_metrics(
            y_true=split_data[
                "y_true"
            ].to_numpy(dtype=float),
            y_pred=split_data[
                "y_pred_fixed_harx"
            ].to_numpy(dtype=float),
        )

        rows.append(
            {
                "model": "Fixed HAR-X",
                "split": split_name,
                "n_observations": len(split_data),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def build_coefficient_table(
    fitted,
    hac_result,
) -> pd.DataFrame:
    """
    Save standardized coefficients with HAC(24)
    standard errors and p-values.
    """
    term_names = fitted.model.exog_names

    confidence_intervals = (
        hac_result.conf_int(alpha=0.05)
    )

    coefficient_table = pd.DataFrame(
        {
            "term": term_names,
            "coefficient": np.asarray(
                fitted.params,
                dtype=float,
            ),
            "hac24_standard_error": np.asarray(
                hac_result.bse,
                dtype=float,
            ),
            "hac24_t_statistic": np.asarray(
                hac_result.tvalues,
                dtype=float,
            ),
            "hac24_p_value": np.asarray(
                hac_result.pvalues,
                dtype=float,
            ),
            "hac24_ci_lower": (
                confidence_intervals[:, 0]
            ),
            "hac24_ci_upper": (
                confidence_intervals[:, 1]
            ),
        }
    )

    return coefficient_table


# --------------------------------------------------
# Residual diagnostics
# --------------------------------------------------

def find_longest_contiguous_segment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the longest sequence of observations separated
    by exactly one hour.
    """
    data = (
        data
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
        .copy()
    )

    new_segment = (
        data["forecast_origin_utc"]
        .diff()
        .ne(pd.Timedelta(hours=1))
    )

    data["_segment_id"] = (
        new_segment.cumsum()
    )

    segment_sizes = (
        data
        .groupby("_segment_id")
        .size()
    )

    longest_segment_id = (
        segment_sizes.idxmax()
    )

    longest_segment = (
        data.loc[
            data["_segment_id"]
            == longest_segment_id
        ]
        .drop(columns="_segment_id")
        .reset_index(drop=True)
    )

    return longest_segment


def run_residual_diagnostics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnose residual autocorrelation on the longest
    continuous model-ready training segment.
    """
    training_predictions = (
        predictions.loc[
            predictions["split"] == "train"
        ]
        .copy()
    )

    continuous_segment = (
        find_longest_contiguous_segment(
            training_predictions
        )
    )

    residuals = continuous_segment[
        "residual"
    ].to_numpy(dtype=float)

    ljung_box = acorr_ljungbox(
        residuals,
        lags=LJUNG_BOX_LAGS,
        model_df=0,
        return_df=True,
    )

    ljung_box = (
        ljung_box
        .reset_index()
        .rename(
            columns={
                "index": "lag",
                "lb_stat": (
                    "ljung_box_statistic"
                ),
                "lb_pvalue": "p_value",
            }
        )
    )

    residual_summary = pd.DataFrame(
        [
            {
                "continuous_segment_start": (
                    continuous_segment[
                        "forecast_origin_utc"
                    ].iloc[0]
                ),
                "continuous_segment_end": (
                    continuous_segment[
                        "forecast_origin_utc"
                    ].iloc[-1]
                ),
                "n_residuals": len(residuals),
                "residual_mean": float(
                    np.mean(residuals)
                ),
                "residual_std": float(
                    np.std(
                        residuals,
                        ddof=1,
                    )
                ),
                "residual_lag1_autocorrelation": (
                    float(
                        pd.Series(residuals)
                        .autocorr(lag=1)
                    )
                ),
                "durbin_watson": float(
                    durbin_watson(residuals)
                ),
            }
        ]
    )

    return residual_summary, ljung_box


# --------------------------------------------------
# Baseline comparison
# --------------------------------------------------

def load_validation_comparison(
    harx_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare validation performance for Naive, selected AR,
    and fixed HAR-X.
    """
    if not NAIVE_METRIC_PATH.exists():
        raise FileNotFoundError(
            f"Naive metrics not found:\n"
            f"{NAIVE_METRIC_PATH}"
        )

    if not AR_SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"AR selection results not found:\n"
            f"{AR_SELECTION_PATH}"
        )

    naive = pd.read_csv(
        NAIVE_METRIC_PATH
    )

    naive_validation = naive.loc[
        naive["split"] == "validation"
    ]

    if len(naive_validation) != 1:
        raise RuntimeError(
            "Expected one Naive validation row."
        )

    naive_validation = (
        naive_validation.iloc[0]
    )

    ar_results = pd.read_csv(
        AR_SELECTION_PATH
    )

    if "selected" not in ar_results.columns:
        raise RuntimeError(
            "The AR selection file has no selected column."
        )

    selected_mask = (
        ar_results["selected"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    selected_ar = ar_results.loc[
        selected_mask
    ]

    if len(selected_ar) != 1:
        raise RuntimeError(
            "Expected exactly one selected AR row."
        )

    selected_ar = selected_ar.iloc[0]

    harx_validation = harx_metrics.loc[
        harx_metrics["split"] == "validation"
    ]

    if len(harx_validation) != 1:
        raise RuntimeError(
            "Expected one HAR-X validation row."
        )

    harx_validation = (
        harx_validation.iloc[0]
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "Naive",
                "mae_log_rv": float(
                    naive_validation[
                        "mae_log_rv"
                    ]
                ),
                "rmse_log_rv": float(
                    naive_validation[
                        "rmse_log_rv"
                    ]
                ),
                "qlike": float(
                    naive_validation["qlike"]
                ),
            },
            {
                "model": (
                    f"AR({int(selected_ar['ar_order'])})"
                ),
                "mae_log_rv": float(
                    selected_ar[
                        "validation_mae_log_rv"
                    ]
                ),
                "rmse_log_rv": float(
                    selected_ar[
                        "validation_rmse_log_rv"
                    ]
                ),
                "qlike": float(
                    selected_ar[
                        "validation_qlike"
                    ]
                ),
            },
            {
                "model": "Fixed HAR-X",
                "mae_log_rv": float(
                    harx_validation[
                        "mae_log_rv"
                    ]
                ),
                "rmse_log_rv": float(
                    harx_validation[
                        "rmse_log_rv"
                    ]
                ),
                "qlike": float(
                    harx_validation["qlike"]
                ),
            },
        ]
    )

    naive_row = comparison.loc[
        comparison["model"] == "Naive"
    ].iloc[0]

    ar_row = comparison.loc[
        comparison["model"].str.startswith(
            "AR("
        )
    ].iloc[0]

    comparison[
        "qlike_improvement_vs_naive_percent"
    ] = (
        (
            naive_row["qlike"]
            - comparison["qlike"]
        )
        / naive_row["qlike"]
        * 100.0
    )

    comparison[
        "qlike_improvement_vs_ar_percent"
    ] = (
        (
            ar_row["qlike"]
            - comparison["qlike"]
        )
        / ar_row["qlike"]
        * 100.0
    )

    return comparison


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def save_outputs(
    data: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    residual_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """
    Save standardized data and all fixed HAR-X results.
    """
    STANDARDIZED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_data_path = (
        STANDARDIZED_OUTPUT_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_data_path.exists():
        temporary_data_path.unlink()

    data.to_parquet(
        temporary_data_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_data_path.replace(
        STANDARDIZED_OUTPUT_PATH
    )

    PARAMETER_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters.to_csv(
        PARAMETER_OUTPUT_PATH,
        index=False,
    )

    metrics.to_csv(
        METRIC_OUTPUT_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENT_OUTPUT_PATH,
        index=False,
    )

    residual_summary.to_csv(
        RESIDUAL_SUMMARY_PATH,
        index=False,
    )

    ljung_box.to_csv(
        LJUNG_BOX_OUTPUT_PATH,
        index=False,
    )

    prediction_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "split",
        "y_true",
        "y_pred_fixed_harx",
        "residual",
        "absolute_log_error",
        "squared_log_error",
        "qlike",
    ]

    prediction_output = predictions[
        prediction_columns
    ].copy()

    PREDICTION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_prediction_path = (
        PREDICTION_OUTPUT_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_prediction_path.exists():
        temporary_prediction_path.unlink()

    prediction_output.to_parquet(
        temporary_prediction_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_prediction_path.replace(
        PREDICTION_OUTPUT_PATH
    )

    if prediction_output[
        "split"
    ].eq("test").any():
        raise RuntimeError(
            "Test predictions were saved unexpectedly."
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Fixed HAR-X Estimation, Validation, and Residual Diagnostics")
    print("=" * 88)
    print(f"Input file: {INPUT_PATH}")
    print("Test-set performance remains sealed.")
    print()

    data = load_data()

    data, parameters = (
        standardize_features(data)
    )

    (
        predictions,
        fitted,
        hac_result,
    ) = fit_fixed_harx(data)

    metrics = build_metric_table(
        predictions
    )

    coefficients = build_coefficient_table(
        fitted=fitted,
        hac_result=hac_result,
    )

    (
        residual_summary,
        ljung_box,
    ) = run_residual_diagnostics(
        predictions
    )

    comparison = load_validation_comparison(
        metrics
    )

    print("Training-based standardization parameters:")
    print(
        parameters[
            [
                "raw_feature",
                "training_mean",
                "training_std_ddof_0",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Fixed HAR-X metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Validation model comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Fixed HAR-X standardized coefficients with HAC(24) inference:")
    print(
        coefficients.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Fixed HAR-X residual summary:")
    print(
        residual_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Fixed HAR-X Ljung-Box tests:")
    print(
        ljung_box.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    save_outputs(
        data=data,
        parameters=parameters,
        metrics=metrics,
        coefficients=coefficients,
        residual_summary=residual_summary,
        ljung_box=ljung_box,
        predictions=predictions,
    )

    print()
    print(
        f"Saved standardized model table to: "
        f"{STANDARDIZED_OUTPUT_PATH}"
    )
    print(
        f"Saved predictions to: "
        f"{PREDICTION_OUTPUT_PATH}"
    )
    print(
        f"Saved result tables to: "
        f"{METRIC_OUTPUT_PATH.parent}"
    )

    print()
    print("[Fixed HAR-X Analysis] PASS")
    print(
        "[Success] The fixed HAR-X model was standardized "
        "using training parameters, evaluated on validation, "
        "and diagnosed without accessing test performance."
    )


if __name__ == "__main__":
    main()