from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from run_tvp_harx_analysis import (
    STATE_TERMS,
    STANDARDIZED_FEATURES,
    TVPHARXModel,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_standardized_2022_2025.parquet"
)

TABLE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

PREDICTION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

STATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "states"
)

AR_SELECTION_PATH = (
    TABLE_OUTPUT_DIR
    / "ar_order_selection_train_validation.csv"
)

FIXED_HARX_COEFFICIENT_PATH = (
    TABLE_OUTPUT_DIR
    / "fixed_harx_coefficients_hac24.csv"
)

TRAINING_TVP_PARAMETER_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_variance_parameters.csv"
)

FINAL_METRIC_PATH = (
    TABLE_OUTPUT_DIR
    / "final_test_model_metrics.csv"
)

FINAL_COMPARISON_PATH = (
    TABLE_OUTPUT_DIR
    / "final_test_model_comparison.csv"
)

LOSS_DIFFERENTIAL_PATH = (
    TABLE_OUTPUT_DIR
    / "final_test_hac_loss_differential_tests.csv"
)

FINAL_AR_COEFFICIENT_PATH = (
    TABLE_OUTPUT_DIR
    / "final_refit_ar_coefficients.csv"
)

FINAL_FIXED_HARX_COEFFICIENT_PATH = (
    TABLE_OUTPUT_DIR
    / "final_refit_fixed_harx_coefficients.csv"
)

FINAL_TVP_PARAMETER_PATH = (
    TABLE_OUTPUT_DIR
    / "final_refit_tvp_harx_variance_parameters.csv"
)

FINAL_TVP_OPTIMIZATION_PATH = (
    TABLE_OUTPUT_DIR
    / "final_refit_tvp_harx_optimization_runs.csv"
)

FINAL_PREDICTION_PATH = (
    PREDICTION_OUTPUT_DIR
    / "final_test_predictions_all_models.parquet"
)

FINAL_TVP_STATE_PATH = (
    STATE_OUTPUT_DIR
    / "final_test_tvp_harx_state_paths.parquet"
)

EXPECTED_DEVELOPMENT_ROWS = 25283
EXPECTED_TEST_ROWS = 8422

INITIAL_STATE_VARIANCE = 0.25
HAC_MAX_LAGS = 24


# --------------------------------------------------
# Loss functions
# --------------------------------------------------

def calculate_qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate QLIKE from log realized variance.

    delta = log(RV_true) - log(RV_pred)

    QLIKE = exp(delta) - delta - 1
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
            np.mean(
                np.abs(errors)
            )
        ),
        "rmse_log_rv": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
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
# Load data and selected model structure
# --------------------------------------------------

def read_selected_ar_order() -> int:
    """
    Read the AR order selected using validation QLIKE.
    """
    if not AR_SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"AR selection file not found:\n"
            f"{AR_SELECTION_PATH}"
        )

    ar_results = pd.read_csv(
        AR_SELECTION_PATH
    )

    if "selected" not in ar_results.columns:
        raise RuntimeError(
            "The AR selection table has no selected column."
        )

    selected_mask = (
        ar_results["selected"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    selected = ar_results.loc[
        selected_mask
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one selected AR order."
        )

    return int(
        selected["ar_order"].iloc[0]
    )


def load_data(
    ar_order: int,
) -> pd.DataFrame:
    """
    Load the complete standardized hourly table and construct
    exact AR lags before filtering observations.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "split",
        "is_model_ready",
        "y_current",
        "y_next_hour",
        *STANDARDIZED_FEATURES,
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

    if data[
        "forecast_origin_utc"
    ].duplicated().any():
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
            "The complete hourly grid is irregular."
        )

    for lag_number in range(
        1,
        ar_order + 1,
    ):
        data[f"ar_lag_{lag_number}"] = (
            data["y_current"]
            .shift(lag_number - 1)
        )

    development_mask = (
        data["split"].isin(
            [
                "train",
                "validation",
            ]
        )
        & data["is_model_ready"].astype(bool)
    )

    test_mask = (
        data["split"].eq("test")
        & data["is_model_ready"].astype(bool)
    )

    development = data.loc[
        development_mask
    ]

    test = data.loc[
        test_mask
    ]

    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError(
            "Unexpected development-sample size: "
            f"expected {EXPECTED_DEVELOPMENT_ROWS}, "
            f"found {len(development)}."
        )

    if len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected test-sample size: "
            f"expected {EXPECTED_TEST_ROWS}, "
            f"found {len(test)}."
        )

    lag_columns = [
        f"ar_lag_{lag_number}"
        for lag_number in range(
            1,
            ar_order + 1,
        )
    ]

    numeric_columns = [
        "y_current",
        "y_next_hour",
        *lag_columns,
        *STANDARDIZED_FEATURES,
    ]

    for sample_name, sample in [
        ("development", development),
        ("test", test),
    ]:
        if sample[
            numeric_columns
        ].isna().any().any():
            raise RuntimeError(
                f"Missing model values were detected in "
                f"the {sample_name} sample."
            )

        if not np.isfinite(
            sample[
                numeric_columns
            ].to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                f"Non-finite model values were detected in "
                f"the {sample_name} sample."
            )

    return data


# --------------------------------------------------
# Naive predictions
# --------------------------------------------------

def build_naive_predictions(
    test: pd.DataFrame,
) -> np.ndarray:
    """
    Naive persistence forecast:

        y_hat_(t+1|t) = y_t
    """
    return test[
        "y_current"
    ].to_numpy(dtype=float)


# --------------------------------------------------
# Final AR refit
# --------------------------------------------------

def fit_final_ar(
    development: pd.DataFrame,
    test: pd.DataFrame,
    ar_order: int,
):
    """
    Refit the selected AR order using train and validation,
    then generate test predictions.
    """
    lag_columns = [
        f"ar_lag_{lag_number}"
        for lag_number in range(
            1,
            ar_order + 1,
        )
    ]

    x_development = sm.add_constant(
        development[
            lag_columns
        ].astype(float),
        has_constant="add",
    )

    y_development = development[
        "y_next_hour"
    ].astype(float)

    fitted = sm.OLS(
        y_development,
        x_development,
    ).fit()

    x_test = sm.add_constant(
        test[
            lag_columns
        ].astype(float),
        has_constant="add",
    )

    predictions = np.asarray(
        fitted.predict(x_test),
        dtype=float,
    )

    return (
        fitted,
        predictions,
    )


# --------------------------------------------------
# Final fixed HAR-X refit
# --------------------------------------------------

def fit_final_fixed_harx(
    development: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Refit the fixed HAR-X model using train and validation,
    then generate test predictions.

    Standardized features retain the original training-based
    scaling parameters.
    """
    x_development = sm.add_constant(
        development[
            STANDARDIZED_FEATURES
        ].astype(float),
        has_constant="add",
    )

    y_development = development[
        "y_next_hour"
    ].astype(float)

    fitted = sm.OLS(
        y_development,
        x_development,
    ).fit()

    x_test = sm.add_constant(
        test[
            STANDARDIZED_FEATURES
        ].astype(float),
        has_constant="add",
    )

    predictions = np.asarray(
        fitted.predict(x_test),
        dtype=float,
    )

    return (
        fitted,
        predictions,
    )


# --------------------------------------------------
# TVP-HAR-X utilities
# --------------------------------------------------

def load_original_initial_state() -> np.ndarray:
    """
    Use the fixed HAR-X coefficients estimated from the original
    training period as the initial TVP state.

    This avoids using future development observations to define
    the state at the beginning of the historical filter.
    """
    if not FIXED_HARX_COEFFICIENT_PATH.exists():
        raise FileNotFoundError(
            f"Fixed HAR-X coefficient file not found:\n"
            f"{FIXED_HARX_COEFFICIENT_PATH}"
        )

    coefficients = pd.read_csv(
        FIXED_HARX_COEFFICIENT_PATH
    )

    required_columns = [
        "term",
        "coefficient",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in coefficients.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing coefficient columns: {missing_columns}"
        )

    coefficient_series = (
        coefficients
        .set_index("term")
        ["coefficient"]
    )

    missing_terms = [
        term
        for term in STATE_TERMS
        if term not in coefficient_series.index
    ]

    if missing_terms:
        raise RuntimeError(
            "Missing initial-state terms: "
            f"{missing_terms}"
        )

    initial_state = (
        coefficient_series
        .reindex(STATE_TERMS)
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        initial_state
    ).all():
        raise RuntimeError(
            "The initial TVP state is non-finite."
        )

    return initial_state


def load_training_tvp_parameters() -> np.ndarray:
    """
    Load the previously estimated training-only parameters for
    use as optimization starting values.
    """
    if not TRAINING_TVP_PARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Training TVP parameter file not found:\n"
            f"{TRAINING_TVP_PARAMETER_PATH}"
        )

    parameters = pd.read_csv(
        TRAINING_TVP_PARAMETER_PATH
    )

    parameter_mapping = (
        parameters
        .set_index("parameter")
        ["value"]
        .to_dict()
    )

    required_parameters = [
        "observation_variance",
        "intercept_state_variance",
        "slope_state_variance",
    ]

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if parameter not in parameter_mapping
    ]

    if missing_parameters:
        raise RuntimeError(
            "Missing training TVP parameters: "
            f"{missing_parameters}"
        )

    values = np.array(
        [
            parameter_mapping[
                "observation_variance"
            ],
            parameter_mapping[
                "intercept_state_variance"
            ],
            parameter_mapping[
                "slope_state_variance"
            ],
        ],
        dtype=float,
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "The saved TVP parameters are non-finite."
        )

    if not (
        values > 0
    ).all():
        raise RuntimeError(
            "The saved TVP parameters are non-positive."
        )

    return values


def prepare_calendar(
    data: pd.DataFrame,
    included_splits: list[str],
) -> pd.DataFrame:
    """
    Retain every clock hour for the requested calendar period.

    Model-unready rows remain on the time grid with missing
    endogenous observations.
    """
    calendar = (
        data.loc[
            data["split"].isin(
                included_splits
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    calendar["endog_for_filter"] = np.where(
        calendar[
            "is_model_ready"
        ].astype(bool),
        calendar["y_next_hour"],
        np.nan,
    )

    return calendar


def build_tvp_design_matrix(
    calendar: pd.DataFrame,
) -> np.ndarray:
    """
    Construct the TVP observation design matrix.

    Missing feature rows correspond only to missing endogenous
    observations and therefore do not contribute to the
    likelihood.
    """
    ready_mask = (
        calendar[
            "is_model_ready"
        ].astype(bool)
    )

    if calendar.loc[
        ready_mask,
        STANDARDIZED_FEATURES,
    ].isna().any().any():
        raise RuntimeError(
            "A model-ready row has missing TVP regressors."
        )

    feature_matrix = (
        calendar[
            STANDARDIZED_FEATURES
        ]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    constant = np.ones(
        shape=(len(calendar), 1),
        dtype=float,
    )

    design = np.concatenate(
        [
            constant,
            feature_matrix,
        ],
        axis=1,
    )

    if design.shape[1] != len(
        STATE_TERMS
    ):
        raise RuntimeError(
            "Incorrect TVP design-matrix dimension."
        )

    return design


def fit_final_tvp_parameters(
    development_calendar: pd.DataFrame,
    initial_state: np.ndarray,
    training_parameters: np.ndarray,
):
    """
    Re-estimate TVP variance parameters using the combined
    train-validation development period.

    Multiple starting values are used to assess numerical
    stability.
    """
    design_matrix = build_tvp_design_matrix(
        development_calendar
    )

    initial_covariance = (
        np.eye(
            len(STATE_TERMS)
        )
        * INITIAL_STATE_VARIANCE
    )

    starting_values = [
        training_parameters.copy(),
        np.array(
            [
                training_parameters[0],
                training_parameters[1] * 0.5,
                training_parameters[2] * 0.5,
            ],
            dtype=float,
        ),
        np.array(
            [
                training_parameters[0],
                training_parameters[1] * 2.0,
                training_parameters[2] * 2.0,
            ],
            dtype=float,
        ),
    ]

    successful_results = []
    optimization_rows = []

    for run_number, start_params in enumerate(
        starting_values,
        start=1,
    ):
        try:
            model = TVPHARXModel(
                endog=development_calendar[
                    "endog_for_filter"
                ].to_numpy(dtype=float),
                design_matrix=design_matrix,
                initial_state=initial_state,
                initial_covariance=initial_covariance,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )

                result = model.fit(
                    start_params=start_params,
                    method="lbfgs",
                    maxiter=300,
                    disp=False,
                )

            converged = bool(
                result.mle_retvals.get(
                    "converged",
                    False,
                )
            )

            finite_result = bool(
                np.isfinite(
                    result.llf
                )
                and np.isfinite(
                    result.params
                ).all()
                and (
                    result.params > 0
                ).all()
            )

            optimization_rows.append(
                {
                    "run": run_number,
                    "start_observation_variance": (
                        start_params[0]
                    ),
                    "start_intercept_state_variance": (
                        start_params[1]
                    ),
                    "start_slope_state_variance": (
                        start_params[2]
                    ),
                    "converged": converged,
                    "log_likelihood": float(
                        result.llf
                    ),
                    "observation_variance": float(
                        result.params[0]
                    ),
                    "intercept_state_variance": float(
                        result.params[1]
                    ),
                    "slope_state_variance": float(
                        result.params[2]
                    ),
                    "error": "",
                }
            )

            if finite_result:
                successful_results.append(
                    result
                )

        except Exception as error:
            optimization_rows.append(
                {
                    "run": run_number,
                    "start_observation_variance": (
                        start_params[0]
                    ),
                    "start_intercept_state_variance": (
                        start_params[1]
                    ),
                    "start_slope_state_variance": (
                        start_params[2]
                    ),
                    "converged": False,
                    "log_likelihood": np.nan,
                    "observation_variance": np.nan,
                    "intercept_state_variance": np.nan,
                    "slope_state_variance": np.nan,
                    "error": str(error),
                }
            )

    optimization_table = pd.DataFrame(
        optimization_rows
    )

    if not successful_results:
        raise RuntimeError(
            "All final TVP optimization runs failed."
        )

    converged_results = [
        result
        for result in successful_results
        if bool(
            result.mle_retvals.get(
                "converged",
                False,
            )
        )
    ]

    candidate_results = (
        converged_results
        if converged_results
        else successful_results
    )

    best_result = max(
        candidate_results,
        key=lambda result: result.llf,
    )

    return (
        best_result,
        optimization_table,
    )


def build_first_test_state(
    development_result,
    variance_parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the predicted state distribution for the first
    test forecast.

    For a random-walk state:

        a_(t+1|t) = a_(t|t)
        P_(t+1|t) = P_(t|t) + Q
    """
    final_filtered_state = (
        development_result
        .filter_results
        .filtered_state[:, -1]
    )

    final_filtered_covariance = (
        development_result
        .filter_results
        .filtered_state_cov[:, :, -1]
    )

    state_variances = np.full(
        len(STATE_TERMS),
        variance_parameters[2],
        dtype=float,
    )

    state_variances[0] = (
        variance_parameters[1]
    )

    state_covariance = np.diag(
        state_variances
    )

    first_test_state = (
        final_filtered_state
    )

    first_test_covariance = (
        final_filtered_covariance
        + state_covariance
    )

    if not np.isfinite(
        first_test_state
    ).all():
        raise RuntimeError(
            "The first test state is non-finite."
        )

    if not np.isfinite(
        first_test_covariance
    ).all():
        raise RuntimeError(
            "The first test state covariance is non-finite."
        )

    return (
        first_test_state,
        first_test_covariance,
    )


def filter_test_tvp(
    test_calendar: pd.DataFrame,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    variance_parameters: np.ndarray,
):
    """
    Generate sequential one-hour-ahead test forecasts.

    Each observed test outcome updates the state only after its
    corresponding forecast has been produced.
    """
    design_matrix = build_tvp_design_matrix(
        test_calendar
    )

    model = TVPHARXModel(
        endog=test_calendar[
            "endog_for_filter"
        ].to_numpy(dtype=float),
        design_matrix=design_matrix,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )

    return model.filter(
        variance_parameters
    )


def extract_test_tvp_predictions(
    test_calendar: pd.DataFrame,
    result,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract test forecasts and test-period state paths.
    """
    forecasts = (
        result
        .filter_results
        .forecasts[0]
    )

    ready_mask = (
        test_calendar[
            "is_model_ready"
        ].to_numpy(dtype=bool)
    )

    ready_positions = np.flatnonzero(
        ready_mask
    )

    test_predictions = (
        forecasts[
            ready_positions
        ]
    )

    if len(test_predictions) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected number of TVP test predictions."
        )

    if not np.isfinite(
        test_predictions
    ).all():
        raise RuntimeError(
            "Non-finite TVP test predictions were detected."
        )

    predicted_states = (
        result
        .filter_results
        .predicted_state[:, :-1]
        .T
    )

    filtered_states = (
        result
        .filter_results
        .filtered_state
        .T
    )

    state_table = test_calendar[
        [
            "forecast_origin_utc",
            "is_model_ready",
        ]
    ].copy()

    for state_index, state_name in enumerate(
        STATE_TERMS
    ):
        state_table[
            f"predicted_{state_name}"
        ] = predicted_states[
            :,
            state_index,
        ]

        state_table[
            f"filtered_{state_name}"
        ] = filtered_states[
            :,
            state_index,
        ]

    return (
        test_predictions,
        state_table,
    )


# --------------------------------------------------
# Prediction table and metrics
# --------------------------------------------------

def build_test_prediction_table(
    test: pd.DataFrame,
    naive_predictions: np.ndarray,
    ar_predictions: np.ndarray,
    fixed_predictions: np.ndarray,
    tvp_predictions: np.ndarray,
    ar_order: int,
) -> pd.DataFrame:
    """
    Combine all four model predictions on identical timestamps.
    """
    output = test[
        [
            "forecast_origin_utc",
            "target_hour_start_utc",
            "target_hour_end_utc",
            "y_current",
            "y_next_hour",
        ]
    ].copy()

    output = output.rename(
        columns={
            "y_next_hour": "y_true",
        }
    )

    output["y_pred_naive"] = (
        naive_predictions
    )

    output[
        f"y_pred_ar_{ar_order}"
    ] = ar_predictions

    output["y_pred_fixed_harx"] = (
        fixed_predictions
    )

    output["y_pred_tvp_harx"] = (
        tvp_predictions
    )

    model_prediction_columns = {
        "Naive": "y_pred_naive",
        f"AR({ar_order})": (
            f"y_pred_ar_{ar_order}"
        ),
        "Fixed HAR-X": (
            "y_pred_fixed_harx"
        ),
        "TVP-HAR-X": (
            "y_pred_tvp_harx"
        ),
    }

    for model_name, prediction_column in (
        model_prediction_columns.items()
    ):
        safe_name = (
            model_name
            .lower()
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(" ", "_")
        )

        error = (
            output["y_true"]
            - output[prediction_column]
        )

        output[
            f"absolute_error_{safe_name}"
        ] = np.abs(
            error
        )

        output[
            f"squared_error_{safe_name}"
        ] = error ** 2

        output[
            f"qlike_{safe_name}"
        ] = calculate_qlike(
            y_true=output[
                "y_true"
            ].to_numpy(dtype=float),
            y_pred=output[
                prediction_column
            ].to_numpy(dtype=float),
        )

    numeric_columns = [
        column
        for column in output.columns
        if column not in [
            "forecast_origin_utc",
            "target_hour_start_utc",
            "target_hour_end_utc",
        ]
    ]

    if not np.isfinite(
        output[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite values were detected in the "
            "final test prediction table."
        )

    return output


def build_final_metric_table(
    predictions: pd.DataFrame,
    ar_order: int,
) -> pd.DataFrame:
    """
    Calculate final test metrics for all models.
    """
    model_columns = {
        "Naive": "y_pred_naive",
        f"AR({ar_order})": (
            f"y_pred_ar_{ar_order}"
        ),
        "Fixed HAR-X": (
            "y_pred_fixed_harx"
        ),
        "TVP-HAR-X": (
            "y_pred_tvp_harx"
        ),
    }

    rows = []

    y_true = predictions[
        "y_true"
    ].to_numpy(dtype=float)

    for model_name, prediction_column in (
        model_columns.items()
    ):
        y_pred = predictions[
            prediction_column
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
        )

        rows.append(
            {
                "model": model_name,
                "split": "test",
                "n_observations": len(
                    predictions
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_final_comparison(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate percentage improvements relative to Naive,
    AR and fixed HAR-X.
    """
    comparison = metrics.copy()

    naive = comparison.loc[
        comparison["model"]
        == "Naive"
    ].iloc[0]

    ar = comparison.loc[
        comparison["model"]
        .str.startswith("AR(")
    ].iloc[0]

    fixed = comparison.loc[
        comparison["model"]
        == "Fixed HAR-X"
    ].iloc[0]

    for metric in [
        "mae_log_rv",
        "rmse_log_rv",
        "qlike",
    ]:
        comparison[
            f"{metric}_improvement_vs_naive_percent"
        ] = (
            (
                naive[metric]
                - comparison[metric]
            )
            / naive[metric]
            * 100.0
        )

        comparison[
            f"{metric}_improvement_vs_ar_percent"
        ] = (
            (
                ar[metric]
                - comparison[metric]
            )
            / ar[metric]
            * 100.0
        )

        comparison[
            f"{metric}_improvement_vs_fixed_harx_percent"
        ] = (
            (
                fixed[metric]
                - comparison[metric]
            )
            / fixed[metric]
            * 100.0
        )

    return comparison


# --------------------------------------------------
# HAC loss-differential tests
# --------------------------------------------------

def hac_loss_differential_test(
    benchmark_loss: np.ndarray,
    tvp_loss: np.ndarray,
    benchmark_name: str,
    loss_name: str,
) -> dict[str, float | int | str | bool]:
    """
    HAC Diebold-Mariano-style loss-differential test.

    Define:

        d_t = loss_benchmark,t - loss_TVP,t

    Positive E[d_t] means TVP-HAR-X has lower expected loss.

    Null:
        E[d_t] <= 0

    Alternative:
        E[d_t] > 0
    """
    loss_difference = (
        benchmark_loss
        - tvp_loss
    )

    design = np.ones(
        shape=(
            len(loss_difference),
            1,
        ),
        dtype=float,
    )

    fitted = sm.OLS(
        loss_difference,
        design,
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": HAC_MAX_LAGS,
            "use_correction": True,
        },
    )

    mean_difference = float(
        fitted.params[0]
    )

    hac_standard_error = float(
        fitted.bse[0]
    )

    test_statistic = float(
        fitted.tvalues[0]
    )

    one_sided_p_value = float(
        1.0
        - norm.cdf(
            test_statistic
        )
    )

    two_sided_p_value = float(
        2.0
        * (
            1.0
            - norm.cdf(
                abs(
                    test_statistic
                )
            )
        )
    )

    benchmark_mean_loss = float(
        np.mean(
            benchmark_loss
        )
    )

    tvp_mean_loss = float(
        np.mean(
            tvp_loss
        )
    )

    improvement_percent = (
        (
            benchmark_mean_loss
            - tvp_mean_loss
        )
        / benchmark_mean_loss
        * 100.0
    )

    return {
        "benchmark": benchmark_name,
        "competing_model": "TVP-HAR-X",
        "loss": loss_name,
        "n_observations": len(
            loss_difference
        ),
        "hac_max_lags": HAC_MAX_LAGS,
        "benchmark_mean_loss": (
            benchmark_mean_loss
        ),
        "tvp_mean_loss": (
            tvp_mean_loss
        ),
        "mean_loss_difference": (
            mean_difference
        ),
        "improvement_percent": (
            improvement_percent
        ),
        "hac_standard_error": (
            hac_standard_error
        ),
        "test_statistic": (
            test_statistic
        ),
        "one_sided_p_value": (
            one_sided_p_value
        ),
        "two_sided_p_value": (
            two_sided_p_value
        ),
        "tvp_significantly_better_at_5_percent": (
            mean_difference > 0
            and one_sided_p_value < 0.05
        ),
    }


def build_loss_differential_tests(
    predictions: pd.DataFrame,
    ar_order: int,
) -> pd.DataFrame:
    """
    Compare TVP-HAR-X with Naive, AR and fixed HAR-X using
    absolute error, squared error and QLIKE.

    QLIKE against Fixed HAR-X is the pre-specified primary test.
    """
    model_safe_names = {
        "Naive": "naive",
        f"AR({ar_order})": (
            f"ar{ar_order}"
        ),
        "Fixed HAR-X": (
            "fixed_har_x"
        ),
        "TVP-HAR-X": (
            "tvp_har_x"
        ),
    }

    loss_prefixes = {
        "Absolute error": (
            "absolute_error"
        ),
        "Squared error": (
            "squared_error"
        ),
        "QLIKE": "qlike",
    }

    tvp_safe_name = (
        model_safe_names[
            "TVP-HAR-X"
        ]
    )

    rows = []

    for benchmark_name in [
        "Naive",
        f"AR({ar_order})",
        "Fixed HAR-X",
    ]:
        benchmark_safe_name = (
            model_safe_names[
                benchmark_name
            ]
        )

        for loss_name, prefix in (
            loss_prefixes.items()
        ):
            benchmark_column = (
                f"{prefix}_"
                f"{benchmark_safe_name}"
            )

            tvp_column = (
                f"{prefix}_"
                f"{tvp_safe_name}"
            )

            result = (
                hac_loss_differential_test(
                    benchmark_loss=predictions[
                        benchmark_column
                    ].to_numpy(dtype=float),
                    tvp_loss=predictions[
                        tvp_column
                    ].to_numpy(dtype=float),
                    benchmark_name=benchmark_name,
                    loss_name=loss_name,
                )
            )

            rows.append(
                result
            )

    return pd.DataFrame(
        rows
    )


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def save_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    loss_tests: pd.DataFrame,
    ar_fitted,
    fixed_fitted,
    ar_order: int,
    tvp_result,
    optimization_table: pd.DataFrame,
    tvp_state_table: pd.DataFrame,
) -> None:
    """
    Save all final test outputs.
    """
    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_prediction_path = (
        FINAL_PREDICTION_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_prediction_path.exists():
        temporary_prediction_path.unlink()

    predictions.to_parquet(
        temporary_prediction_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_prediction_path.replace(
        FINAL_PREDICTION_PATH
    )

    metrics.to_csv(
        FINAL_METRIC_PATH,
        index=False,
    )

    comparison.to_csv(
        FINAL_COMPARISON_PATH,
        index=False,
    )

    loss_tests.to_csv(
        LOSS_DIFFERENTIAL_PATH,
        index=False,
    )

    ar_coefficients = pd.DataFrame(
        {
            "model": f"AR({ar_order})",
            "term": ar_fitted.params.index,
            "coefficient": (
                ar_fitted.params
                .to_numpy(dtype=float)
            ),
            "estimated_from": (
                "train_and_validation"
            ),
        }
    )

    ar_coefficients.to_csv(
        FINAL_AR_COEFFICIENT_PATH,
        index=False,
    )

    fixed_coefficients = pd.DataFrame(
        {
            "model": "Fixed HAR-X",
            "term": fixed_fitted.params.index,
            "coefficient": (
                fixed_fitted.params
                .to_numpy(dtype=float)
            ),
            "estimated_from": (
                "train_and_validation"
            ),
        }
    )

    fixed_coefficients.to_csv(
        FINAL_FIXED_HARX_COEFFICIENT_PATH,
        index=False,
    )

    tvp_parameters = pd.DataFrame(
        [
            {
                "parameter": (
                    "observation_variance"
                ),
                "value": float(
                    tvp_result.params[0]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        tvp_result.params[0]
                    )
                ),
                "estimated_from": (
                    "train_and_validation"
                ),
            },
            {
                "parameter": (
                    "intercept_state_variance"
                ),
                "value": float(
                    tvp_result.params[1]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        tvp_result.params[1]
                    )
                ),
                "estimated_from": (
                    "train_and_validation"
                ),
            },
            {
                "parameter": (
                    "slope_state_variance"
                ),
                "value": float(
                    tvp_result.params[2]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        tvp_result.params[2]
                    )
                ),
                "estimated_from": (
                    "train_and_validation"
                ),
            },
        ]
    )

    tvp_parameters.to_csv(
        FINAL_TVP_PARAMETER_PATH,
        index=False,
    )

    optimization_table.to_csv(
        FINAL_TVP_OPTIMIZATION_PATH,
        index=False,
    )

    temporary_state_path = (
        FINAL_TVP_STATE_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_state_path.exists():
        temporary_state_path.unlink()

    tvp_state_table.to_parquet(
        temporary_state_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_state_path.replace(
        FINAL_TVP_STATE_PATH
    )

    saved_predictions = pd.read_parquet(
        FINAL_PREDICTION_PATH
    )

    if len(saved_predictions) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Saved final prediction count is incorrect."
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 92)
    print("Final Out-of-Sample Test Evaluation for All Volatility Models")
    print("=" * 92)
    print(f"Input file: {INPUT_PATH}")
    print(
        "Model selection is complete. "
        "The 2025 test set is now evaluated once."
    )
    print()

    ar_order = read_selected_ar_order()

    print(f"Locked AR specification: AR({ar_order})")
    print("Locked HAR-X feature set: 7 predictors")
    print(
        "Locked TVP structure: random-walk intercept "
        "and shared random-walk slope variance"
    )
    print()

    data = load_data(
        ar_order=ar_order
    )

    development = (
        data.loc[
            data["split"].isin(
                [
                    "train",
                    "validation",
                ]
            )
            & data[
                "is_model_ready"
            ].astype(bool)
        ]
        .copy()
    )

    test = (
        data.loc[
            data["split"].eq("test")
            & data[
                "is_model_ready"
            ].astype(bool)
        ]
        .copy()
    )

    # ----------------------------------------------
    # Naive
    # ----------------------------------------------

    naive_predictions = (
        build_naive_predictions(
            test
        )
    )

    # ----------------------------------------------
    # AR
    # ----------------------------------------------

    (
        ar_fitted,
        ar_predictions,
    ) = fit_final_ar(
        development=development,
        test=test,
        ar_order=ar_order,
    )

    # ----------------------------------------------
    # Fixed HAR-X
    # ----------------------------------------------

    (
        fixed_fitted,
        fixed_predictions,
    ) = fit_final_fixed_harx(
        development=development,
        test=test,
    )

    # ----------------------------------------------
    # TVP-HAR-X development refit
    # ----------------------------------------------

    initial_state = (
        load_original_initial_state()
    )

    training_parameters = (
        load_training_tvp_parameters()
    )

    development_calendar = (
        prepare_calendar(
            data=data,
            included_splits=[
                "train",
                "validation",
            ],
        )
    )

    (
        final_tvp_result,
        optimization_table,
    ) = fit_final_tvp_parameters(
        development_calendar=(
            development_calendar
        ),
        initial_state=initial_state,
        training_parameters=(
            training_parameters
        ),
    )

    # ----------------------------------------------
    # Begin test filter from end-of-development state
    # ----------------------------------------------

    (
        first_test_state,
        first_test_covariance,
    ) = build_first_test_state(
        development_result=(
            final_tvp_result
        ),
        variance_parameters=(
            final_tvp_result.params
        ),
    )

    test_calendar = prepare_calendar(
        data=data,
        included_splits=["test"],
    )

    test_tvp_result = filter_test_tvp(
        test_calendar=test_calendar,
        initial_state=first_test_state,
        initial_covariance=(
            first_test_covariance
        ),
        variance_parameters=(
            final_tvp_result.params
        ),
    )

    (
        tvp_predictions,
        tvp_state_table,
    ) = extract_test_tvp_predictions(
        test_calendar=test_calendar,
        result=test_tvp_result,
    )

    # ----------------------------------------------
    # Final prediction and metric tables
    # ----------------------------------------------

    predictions = (
        build_test_prediction_table(
            test=test,
            naive_predictions=(
                naive_predictions
            ),
            ar_predictions=(
                ar_predictions
            ),
            fixed_predictions=(
                fixed_predictions
            ),
            tvp_predictions=(
                tvp_predictions
            ),
            ar_order=ar_order,
        )
    )

    metrics = build_final_metric_table(
        predictions=predictions,
        ar_order=ar_order,
    )

    comparison = build_final_comparison(
        metrics
    )

    loss_tests = (
        build_loss_differential_tests(
            predictions=predictions,
            ar_order=ar_order,
        )
    )

    # ----------------------------------------------
    # Print results
    # ----------------------------------------------

    print("Final TVP-HAR-X optimization runs:")
    print(
        optimization_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10g}"
            ),
        )
    )

    print()
    print("Final TVP-HAR-X variance parameters:")
    print(
        f"Observation variance:       "
        f"{final_tvp_result.params[0]:.12e}"
    )
    print(
        f"Intercept state variance:   "
        f"{final_tvp_result.params[1]:.12e}"
    )
    print(
        f"Slope state variance:       "
        f"{final_tvp_result.params[2]:.12e}"
    )
    print(
        f"Development log likelihood: "
        f"{final_tvp_result.llf:.10f}"
    )
    print(
        f"Optimizer converged:         "
        f"{bool(final_tvp_result.mle_retvals.get('converged', False))}"
    )

    print()
    print("Final test metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print("Final test percentage improvements:")
    display_comparison_columns = [
        "model",
        "mae_log_rv_improvement_vs_naive_percent",
        "rmse_log_rv_improvement_vs_naive_percent",
        "qlike_improvement_vs_naive_percent",
        "qlike_improvement_vs_ar_percent",
        "qlike_improvement_vs_fixed_harx_percent",
    ]

    print(
        comparison[
            display_comparison_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "HAC loss-differential tests: "
        "TVP-HAR-X versus benchmarks"
    )

    print(
        loss_tests[
            [
                "benchmark",
                "loss",
                "benchmark_mean_loss",
                "tvp_mean_loss",
                "improvement_percent",
                "test_statistic",
                "one_sided_p_value",
                "tvp_significantly_better_at_5_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "Primary significance result "
        "(TVP-HAR-X versus Fixed HAR-X under QLIKE):"
    )

    primary_result = loss_tests.loc[
        (
            loss_tests["benchmark"]
            == "Fixed HAR-X"
        )
        & (
            loss_tests["loss"]
            == "QLIKE"
        )
    ]

    if len(primary_result) != 1:
        raise RuntimeError(
            "The primary QLIKE test result is missing."
        )

    print(
        primary_result.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    save_outputs(
        predictions=predictions,
        metrics=metrics,
        comparison=comparison,
        loss_tests=loss_tests,
        ar_fitted=ar_fitted,
        fixed_fitted=fixed_fitted,
        ar_order=ar_order,
        tvp_result=final_tvp_result,
        optimization_table=(
            optimization_table
        ),
        tvp_state_table=(
            tvp_state_table
        ),
    )

    print()
    print(
        f"Saved final metrics to:     "
        f"{FINAL_METRIC_PATH}"
    )
    print(
        f"Saved final predictions to: "
        f"{FINAL_PREDICTION_PATH}"
    )
    print(
        f"Saved significance tests to:"
        f" {LOSS_DIFFERENTIAL_PATH}"
    )
    print(
        f"Saved final state paths to: "
        f"{FINAL_TVP_STATE_PATH}"
    )

    print()
    print("[Final Test Evaluation] PASS")
    print(
        "[Success] All locked models were evaluated once "
        "on the common 2025 test sample, and predictive-loss "
        "differences were tested using HAC standard errors."
    )


if __name__ == "__main__":
    main()