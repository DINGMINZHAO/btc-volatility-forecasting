from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


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

NAIVE_METRIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "naive_baseline_metrics_train_validation.csv"
)

TABLE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "ar_selected_predictions_train_validation.parquet"
)

MIN_AR_ORDER = 1
MAX_AR_ORDER = 24

ROLLING_WINDOW = 168
CORRELATION_LAGS = 168

EXPECTED_COUNTS = {
    "train": 16673,
    "validation": 8610,
}


# --------------------------------------------------
# General utilities
# --------------------------------------------------

def calculate_qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate QLIKE using log realized variance.

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
    Calculate MAE and RMSE on the log-RV scale and QLIKE
    on the original realized-variance scale.
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


def find_longest_contiguous_segment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the longest section whose forecast origins remain
    exactly one hour apart.
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

    data["_segment_id"] = new_segment.cumsum()

    segment_sizes = (
        data
        .groupby("_segment_id")
        .size()
    )

    longest_segment_id = segment_sizes.idxmax()

    longest_segment = (
        data.loc[
            data["_segment_id"] == longest_segment_id
        ]
        .drop(columns="_segment_id")
        .reset_index(drop=True)
    )

    return longest_segment


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load the full hourly table so that lags and gaps remain
    defined on the original clock-time grid.
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
        "is_valid_target",
        "is_model_ready",
        "y_current",
        "y_next_hour",
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

    return data


# --------------------------------------------------
# Stationarity analysis
# --------------------------------------------------

def prepare_stationarity_sample(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create:

    1. Full training calendar sample for plotting with gaps.
    2. Longest continuous valid training segment for formal
       stationarity and autocorrelation analysis.
    """
    training_calendar = (
        data.loc[
            data["split"] == "train",
            [
                "forecast_origin_utc",
                "is_valid_target",
                "y_current",
            ],
        ]
        .copy()
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    valid_training = (
        training_calendar.loc[
            training_calendar[
                "is_valid_target"
            ].astype(bool)
            & training_calendar["y_current"].notna()
        ]
        .copy()
    )

    longest_segment = find_longest_contiguous_segment(
        valid_training
    )

    if len(longest_segment) < 1000:
        raise RuntimeError(
            "The longest continuous training segment "
            "is unexpectedly short."
        )

    return training_calendar, longest_segment


def run_stationarity_tests(
    series: pd.Series,
) -> pd.DataFrame:
    """
    Run constant-only ADF and KPSS tests.

    ADF null:
        Unit root / nonstationarity.

    KPSS null:
        Level stationarity.
    """
    values = series.to_numpy(dtype=float)

    adf_result = adfuller(
        values,
        regression="c",
        autolag="AIC",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kpss_result = kpss(
            values,
            regression="c",
            nlags="auto",
        )

    adf_statistic = float(adf_result[0])
    adf_p_value = float(adf_result[1])
    adf_lags = int(adf_result[2])
    adf_nobs = int(adf_result[3])

    kpss_statistic = float(kpss_result[0])
    kpss_p_value = float(kpss_result[1])
    kpss_lags = int(kpss_result[2])

    rows = [
        {
            "test": "ADF",
            "null_hypothesis": (
                "unit root / nonstationary"
            ),
            "statistic": adf_statistic,
            "p_value": adf_p_value,
            "lags": adf_lags,
            "n_observations": adf_nobs,
            "decision_at_5_percent": (
                "reject unit root"
                if adf_p_value < 0.05
                else "do not reject unit root"
            ),
        },
        {
            "test": "KPSS",
            "null_hypothesis": (
                "level stationary"
            ),
            "statistic": kpss_statistic,
            "p_value": kpss_p_value,
            "lags": kpss_lags,
            "n_observations": len(values),
            "decision_at_5_percent": (
                "reject level stationarity"
                if kpss_p_value < 0.05
                else "do not reject level stationarity"
            ),
        },
    ]

    return pd.DataFrame(rows)


def save_stationarity_figures(
    training_calendar: pd.DataFrame,
    continuous_segment: pd.DataFrame,
) -> None:
    """
    Save the time-series, rolling-statistics, ACF, and PACF
    figures.
    """
    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = training_calendar.copy()

    plot_data["rolling_mean_168h"] = (
        plot_data["y_current"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .mean()
    )

    plot_data["rolling_std_168h"] = (
        plot_data["y_current"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .std()
    )

    figure = plt.figure(figsize=(13, 5))
    plt.plot(
        plot_data["forecast_origin_utc"],
        plot_data["y_current"],
        linewidth=0.5,
        label="Log realized variance",
    )
    plt.plot(
        plot_data["forecast_origin_utc"],
        plot_data["rolling_mean_168h"],
        linewidth=1.2,
        label="168-hour rolling mean",
    )
    plt.title(
        "Training Log Realized Variance and Rolling Mean"
    )
    plt.xlabel("Forecast origin (UTC)")
    plt.ylabel("Log realized variance")
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "training_log_rv_and_rolling_mean.png",
        dpi=200,
    )

    plt.close(figure)

    figure = plt.figure(figsize=(13, 4))
    plt.plot(
        plot_data["forecast_origin_utc"],
        plot_data["rolling_std_168h"],
        linewidth=1.0,
    )
    plt.title(
        "Training 168-Hour Rolling Standard Deviation"
    )
    plt.xlabel("Forecast origin (UTC)")
    plt.ylabel("Rolling standard deviation")
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "training_log_rv_rolling_std.png",
        dpi=200,
    )

    plt.close(figure)

    figure = plt.figure(figsize=(11, 4))
    plot_acf(
        continuous_segment["y_current"],
        lags=CORRELATION_LAGS,
        zero=False,
        ax=plt.gca(),
    )
    plt.title(
        "ACF of Training Log Realized Variance"
    )
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "training_log_rv_acf.png",
        dpi=200,
    )

    plt.close(figure)

    figure = plt.figure(figsize=(11, 4))
    plot_pacf(
        continuous_segment["y_current"],
        lags=CORRELATION_LAGS,
        zero=False,
        method="ywm",
        ax=plt.gca(),
    )
    plt.title(
        "PACF of Training Log Realized Variance"
    )
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "training_log_rv_pacf.png",
        dpi=200,
    )

    plt.close(figure)


# --------------------------------------------------
# AR lag construction and estimation
# --------------------------------------------------

def construct_exact_ar_lags(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct exact clock-time lags on the complete hourly grid.

    For forecasting y_(t+1):

        ar_lag_1 = y_t
        ar_lag_2 = y_(t-1)
        ...
    """
    data = data.copy()

    for lag_number in range(
        MIN_AR_ORDER,
        MAX_AR_ORDER + 1,
    ):
        data[f"ar_lag_{lag_number}"] = (
            data["y_current"]
            .shift(lag_number - 1)
        )

    evaluation = data.loc[
        data["split"].isin(
            ["train", "validation"]
        )
        & data["is_model_ready"].astype(bool)
    ]

    actual_counts = (
        evaluation["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        actual_count = int(
            actual_counts.get(split_name, 0)
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected {split_name} sample size: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    lag_columns = [
        f"ar_lag_{lag_number}"
        for lag_number in range(
            1,
            MAX_AR_ORDER + 1,
        )
    ]

    required_numeric = (
        lag_columns
        + ["y_next_hour"]
    )

    if evaluation[
        required_numeric
    ].isna().any().any():
        raise RuntimeError(
            "Missing exact AR lags were found among "
            "model-ready observations."
        )

    return data


def fit_ar_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    order: int,
):
    """
    Estimate a fixed-parameter AR(p) by OLS using training data.
    """
    lag_columns = [
        f"ar_lag_{lag_number}"
        for lag_number in range(
            1,
            order + 1,
        )
    ]

    x_train = sm.add_constant(
        training[lag_columns].astype(float),
        has_constant="add",
    )

    y_train = training[
        "y_next_hour"
    ].astype(float)

    fitted = sm.OLS(
        y_train,
        x_train,
    ).fit()

    train_predictions = np.asarray(
        fitted.predict(x_train),
        dtype=float,
    )

    x_validation = sm.add_constant(
        validation[lag_columns].astype(float),
        has_constant="add",
    )

    validation_predictions = np.asarray(
        fitted.predict(x_validation),
        dtype=float,
    )

    return (
        fitted,
        train_predictions,
        validation_predictions,
    )


def evaluate_ar_orders(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit AR(1) through AR(24) and evaluate them using the same
    training and validation observations.
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

    y_train = training[
        "y_next_hour"
    ].to_numpy(dtype=float)

    y_validation = validation[
        "y_next_hour"
    ].to_numpy(dtype=float)

    result_rows = []

    for order in range(
        MIN_AR_ORDER,
        MAX_AR_ORDER + 1,
    ):
        (
            fitted,
            train_predictions,
            validation_predictions,
        ) = fit_ar_model(
            training=training,
            validation=validation,
            order=order,
        )

        train_metrics = calculate_metrics(
            y_true=y_train,
            y_pred=train_predictions,
        )

        validation_metrics = calculate_metrics(
            y_true=y_validation,
            y_pred=validation_predictions,
        )

        result_rows.append(
            {
                "ar_order": order,
                "n_parameters": len(fitted.params),
                "train_aic": float(fitted.aic),
                "train_bic": float(fitted.bic),
                "train_mae_log_rv": (
                    train_metrics["mae_log_rv"]
                ),
                "train_rmse_log_rv": (
                    train_metrics["rmse_log_rv"]
                ),
                "train_qlike": (
                    train_metrics["qlike"]
                ),
                "validation_mae_log_rv": (
                    validation_metrics["mae_log_rv"]
                ),
                "validation_rmse_log_rv": (
                    validation_metrics["rmse_log_rv"]
                ),
                "validation_qlike": (
                    validation_metrics["qlike"]
                ),
            }
        )

    return pd.DataFrame(result_rows)


def select_ar_order(
    results: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    """
    Primary criterion:
        Validation QLIKE.

    Tie-breakers:
        Validation RMSE,
        validation MAE,
        lower order.
    """
    ranked = (
        results
        .sort_values(
            [
                "validation_qlike",
                "validation_rmse_log_rv",
                "validation_mae_log_rv",
                "ar_order",
            ]
        )
        .reset_index(drop=True)
    )

    selected_order = int(
        ranked.loc[0, "ar_order"]
    )

    return selected_order, ranked


# --------------------------------------------------
# Selected AR diagnostics
# --------------------------------------------------

def calculate_ar_roots(
    fitted,
    order: int,
) -> pd.DataFrame:
    """
    Calculate roots of:

        1 - phi_1 z - ... - phi_p z^p = 0

    A fixed AR model is stable when every root lies outside
    the unit circle.
    """
    lag_columns = [
        f"ar_lag_{lag_number}"
        for lag_number in range(
            1,
            order + 1,
        )
    ]

    coefficients = (
        fitted.params[lag_columns]
        .to_numpy(dtype=float)
    )

    polynomial = np.concatenate(
        [
            -coefficients[::-1],
            [1.0],
        ]
    )

    roots = np.roots(polynomial)

    root_table = pd.DataFrame(
        {
            "root_real": roots.real,
            "root_imaginary": roots.imag,
            "root_modulus": np.abs(roots),
        }
    )

    root_table["outside_unit_circle"] = (
        root_table["root_modulus"] > 1.0
    )

    return root_table


def build_selected_predictions(
    data: pd.DataFrame,
    selected_order: int,
):
    """
    Refit and store predictions for the selected model.
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

    (
        fitted,
        train_predictions,
        validation_predictions,
    ) = fit_ar_model(
        training=training,
        validation=validation,
        order=selected_order,
    )

    training["y_pred_ar"] = train_predictions
    validation["y_pred_ar"] = validation_predictions

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
        - predictions["y_pred_ar"]
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
            "y_pred_ar"
        ].to_numpy(dtype=float),
    )

    predictions["ar_order"] = selected_order

    return predictions, fitted


def run_residual_diagnostics(
    predictions: pd.DataFrame,
    selected_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run residual diagnostics on the longest continuous section
    of model-ready training predictions.
    """
    training_predictions = (
        predictions.loc[
            predictions["split"] == "train"
        ]
        .copy()
    )

    continuous_predictions = (
        find_longest_contiguous_segment(
            training_predictions
        )
    )

    residuals = continuous_predictions[
        "residual"
    ].to_numpy(dtype=float)

    ljung_box_lags = sorted(
        {
            max(24, selected_order + 5),
            max(48, selected_order + 12),
            max(168, selected_order + 24),
        }
    )

    ljung_box = acorr_ljungbox(
        residuals,
        lags=ljung_box_lags,
        model_df=selected_order,
        return_df=True,
    )

    ljung_box = (
        ljung_box
        .reset_index()
        .rename(
            columns={
                "index": "lag",
                "lb_stat": "ljung_box_statistic",
                "lb_pvalue": "p_value",
            }
        )
    )

    residual_summary = pd.DataFrame(
        [
            {
                "continuous_segment_start": (
                    continuous_predictions[
                        "forecast_origin_utc"
                    ].iloc[0]
                ),
                "continuous_segment_end": (
                    continuous_predictions[
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
                "residual_lag1_autocorrelation": float(
                    pd.Series(residuals)
                    .autocorr(lag=1)
                ),
            }
        ]
    )

    figure = plt.figure(figsize=(11, 4))

    plot_acf(
        residuals,
        lags=CORRELATION_LAGS,
        zero=False,
        ax=plt.gca(),
    )

    plt.title(
        f"Residual ACF of Selected AR({selected_order})"
    )
    plt.tight_layout()

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "selected_ar_residual_acf.png",
        dpi=200,
    )

    plt.close(figure)

    return residual_summary, ljung_box


def compare_with_naive(
    selected_result: pd.Series,
) -> pd.DataFrame:
    """
    Compare validation performance with the saved Naive baseline.
    """
    if not NAIVE_METRIC_PATH.exists():
        raise FileNotFoundError(
            f"Naive metric file not found:\n"
            f"{NAIVE_METRIC_PATH}"
        )

    naive = pd.read_csv(NAIVE_METRIC_PATH)

    naive_validation = naive.loc[
        naive["split"] == "validation"
    ]

    if len(naive_validation) != 1:
        raise RuntimeError(
            "Expected one Naive validation result."
        )

    naive_validation = (
        naive_validation.iloc[0]
    )

    mappings = {
        "MAE": (
            "mae_log_rv",
            "validation_mae_log_rv",
        ),
        "RMSE": (
            "rmse_log_rv",
            "validation_rmse_log_rv",
        ),
        "QLIKE": (
            "qlike",
            "validation_qlike",
        ),
    }

    rows = []

    for metric, (
        naive_column,
        ar_column,
    ) in mappings.items():
        naive_value = float(
            naive_validation[naive_column]
        )

        ar_value = float(
            selected_result[ar_column]
        )

        rows.append(
            {
                "metric": metric,
                "naive_validation": naive_value,
                "selected_ar_validation": ar_value,
                "improvement_percent": (
                    (naive_value - ar_value)
                    / naive_value
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Save results
# --------------------------------------------------

def save_outputs(
    stationarity_tests: pd.DataFrame,
    ar_results: pd.DataFrame,
    selected_order: int,
    fitted,
    root_table: pd.DataFrame,
    residual_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """
    Save all AR analysis outputs.
    """
    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stationarity_tests.to_csv(
        TABLE_OUTPUT_DIR
        / "training_stationarity_tests.csv",
        index=False,
    )

    ar_results = ar_results.copy()

    ar_results["selected"] = (
        ar_results["ar_order"]
        == selected_order
    )

    ar_results.to_csv(
        TABLE_OUTPUT_DIR
        / "ar_order_selection_train_validation.csv",
        index=False,
    )

    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "coefficient": (
                fitted.params.to_numpy(dtype=float)
            ),
        }
    )

    coefficients.to_csv(
        TABLE_OUTPUT_DIR
        / "selected_ar_coefficients.csv",
        index=False,
    )

    root_table.to_csv(
        TABLE_OUTPUT_DIR
        / "selected_ar_roots.csv",
        index=False,
    )

    residual_summary.to_csv(
        TABLE_OUTPUT_DIR
        / "selected_ar_residual_summary.csv",
        index=False,
    )

    ljung_box.to_csv(
        TABLE_OUTPUT_DIR
        / "selected_ar_ljung_box.csv",
        index=False,
    )

    prediction_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "split",
        "ar_order",
        "y_current",
        "y_true",
        "y_pred_ar",
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

    temporary_path = (
        PREDICTION_OUTPUT_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    prediction_output.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_path.replace(
        PREDICTION_OUTPUT_PATH
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("Complete AR Stationarity, Order Selection, and Diagnostic Analysis")
    print("=" * 88)
    print(f"Input file: {INPUT_PATH}")
    print("Test-set performance remains sealed.")
    print()

    data = load_data()

    # ----------------------------------------------
    # Stationarity
    # ----------------------------------------------

    (
        training_calendar,
        continuous_segment,
    ) = prepare_stationarity_sample(data)

    stationarity_tests = run_stationarity_tests(
        continuous_segment["y_current"]
    )

    save_stationarity_figures(
        training_calendar=training_calendar,
        continuous_segment=continuous_segment,
    )

    print("Stationarity analysis sample:")
    print(
        f"Continuous segment start: "
        f"{continuous_segment['forecast_origin_utc'].iloc[0]}"
    )
    print(
        f"Continuous segment end:   "
        f"{continuous_segment['forecast_origin_utc'].iloc[-1]}"
    )
    print(
        f"Observations:             "
        f"{len(continuous_segment)}"
    )

    print()
    print("ADF and KPSS results:")
    print(
        stationarity_tests.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    # ----------------------------------------------
    # AR order selection
    # ----------------------------------------------

    data = construct_exact_ar_lags(data)

    ar_results = evaluate_ar_orders(data)

    selected_order, ranked = select_ar_order(
        ar_results
    )

    selected_result = ranked.iloc[0]

    predictions, fitted = (
        build_selected_predictions(
            data=data,
            selected_order=selected_order,
        )
    )

    print()
    print("Top ten AR orders by validation QLIKE:")
    print(
        ranked[
            [
                "ar_order",
                "train_aic",
                "train_bic",
                "validation_mae_log_rv",
                "validation_rmse_log_rv",
                "validation_qlike",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Selected AR order:")
    print(f"AR({selected_order})")

    print()
    print("Selected AR metrics:")
    print(
        pd.DataFrame(
            [
                {
                    "ar_order": selected_order,
                    "train_mae_log_rv": (
                        selected_result[
                            "train_mae_log_rv"
                        ]
                    ),
                    "train_rmse_log_rv": (
                        selected_result[
                            "train_rmse_log_rv"
                        ]
                    ),
                    "train_qlike": (
                        selected_result[
                            "train_qlike"
                        ]
                    ),
                    "validation_mae_log_rv": (
                        selected_result[
                            "validation_mae_log_rv"
                        ]
                    ),
                    "validation_rmse_log_rv": (
                        selected_result[
                            "validation_rmse_log_rv"
                        ]
                    ),
                    "validation_qlike": (
                        selected_result[
                            "validation_qlike"
                        ]
                    ),
                }
            ]
        ).to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    # ----------------------------------------------
    # Stability and residual diagnostics
    # ----------------------------------------------

    root_table = calculate_ar_roots(
        fitted=fitted,
        order=selected_order,
    )

    minimum_root_modulus = float(
        root_table["root_modulus"].min()
    )

    stable = bool(
        root_table[
            "outside_unit_circle"
        ].all()
    )

    residual_summary, ljung_box = (
        run_residual_diagnostics(
            predictions=predictions,
            selected_order=selected_order,
        )
    )

    print()
    print("AR stability diagnostics:")
    print(
        f"Minimum characteristic-root modulus: "
        f"{minimum_root_modulus:.10f}"
    )
    print(
        f"All roots outside unit circle: {stable}"
    )

    print()
    print("Residual summary:")
    print(
        residual_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("Ljung-Box residual tests:")
    print(
        ljung_box.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    # ----------------------------------------------
    # Naive comparison
    # ----------------------------------------------

    naive_comparison = compare_with_naive(
        selected_result=selected_result,
    )

    print()
    print("Validation comparison with Naive:")
    print(
        naive_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    save_outputs(
        stationarity_tests=stationarity_tests,
        ar_results=ar_results,
        selected_order=selected_order,
        fitted=fitted,
        root_table=root_table,
        residual_summary=residual_summary,
        ljung_box=ljung_box,
        predictions=predictions,
    )

    print()
    print(f"Tables saved to:  {TABLE_OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")
    print(
        f"Predictions saved to: "
        f"{PREDICTION_OUTPUT_PATH}"
    )

    print()
    print("[Complete AR Analysis] PASS")
    print(
        "[Success] Stationarity, AR order selection, "
        "stability, and residual diagnostics were completed "
        "without accessing test-set performance."
    )


if __name__ == "__main__":
    main()