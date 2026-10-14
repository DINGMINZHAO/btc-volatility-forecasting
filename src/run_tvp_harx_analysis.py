from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.mlemodel import MLEModel


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

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "tvp_harx_predictions_train_validation.parquet"
)

STATE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "states"
    / "tvp_harx_state_paths_train_validation.parquet"
)

NAIVE_METRIC_PATH = (
    TABLE_OUTPUT_DIR
    / "naive_baseline_metrics_train_validation.csv"
)

AR_SELECTION_PATH = (
    TABLE_OUTPUT_DIR
    / "ar_order_selection_train_validation.csv"
)

FIXED_HARX_METRIC_PATH = (
    TABLE_OUTPUT_DIR
    / "fixed_harx_metrics_train_validation.csv"
)

TVP_METRIC_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_metrics_train_validation.csv"
)

TVP_PARAMETER_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_variance_parameters.csv"
)

OPTIMIZATION_OUTPUT_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_optimization_runs.csv"
)

STATE_SUMMARY_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_validation_state_summary.csv"
)

LJUNG_BOX_OUTPUT_PATH = (
    TABLE_OUTPUT_DIR
    / "tvp_harx_ljung_box.csv"
)

MODEL_COMPARISON_PATH = (
    TABLE_OUTPUT_DIR
    / "validation_model_comparison.csv"
)

STANDARDIZED_FEATURES = [
    "z_har_1h",
    "z_har_6h",
    "z_har_24h",
    "z_har_168h",
    "z_scaled_log_downside_semivariance",
    "z_log1p_number_of_trades",
    "z_aggressive_buy_proxy",
]

STATE_TERMS = [
    "const",
    *STANDARDIZED_FEATURES,
]

EXPECTED_COUNTS = {
    "train": 16673,
    "validation": 8610,
}

INITIAL_STATE_VARIANCE = 0.25

LJUNG_BOX_LAGS = [
    24,
    48,
    168,
]


# --------------------------------------------------
# TVP-HAR-X state-space model
# --------------------------------------------------

class TVPHARXModel(MLEModel):
    """
    Dynamic regression with random-walk coefficients.

    Observation equation:
        y_t = x_t' beta_t + epsilon_t

    State equation:
        beta_(t+1) = beta_t + eta_t

    State covariance:
        diag(q_intercept, q_slope, ..., q_slope)
    """

    def __init__(
        self,
        endog: np.ndarray,
        design_matrix: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.design_matrix = np.asarray(
            design_matrix,
            dtype=float,
        )

        self.number_of_states = (
            self.design_matrix.shape[1]
        )

        super().__init__(
            endog=np.asarray(
                endog,
                dtype=float,
            ),
            k_states=self.number_of_states,
            k_posdef=self.number_of_states,
        )

        # Shape:
        # (k_endog, k_states, n_observations)
        self.ssm["design"] = (
            self.design_matrix.T[
                np.newaxis,
                :,
                :,
            ]
        )

        self.ssm["transition"] = np.eye(
            self.number_of_states
        )

        self.ssm["selection"] = np.eye(
            self.number_of_states
        )

        self.initialize_known(
            np.asarray(
                initial_state,
                dtype=float,
            ),
            np.asarray(
                initial_covariance,
                dtype=float,
            ),
        )

    @property
    def param_names(self) -> list[str]:
        return [
            "observation_variance",
            "intercept_state_variance",
            "slope_state_variance",
        ]

    @property
    def start_params(self) -> np.ndarray:
        return np.array(
            [
                1.0,
                1e-4,
                1e-5,
            ],
            dtype=float,
        )

    def transform_params(
        self,
        unconstrained: np.ndarray,
    ) -> np.ndarray:
        """
        Exponential transformation ensures positive variances.
        """
        return np.exp(
            unconstrained
        )

    def untransform_params(
        self,
        constrained: np.ndarray,
    ) -> np.ndarray:
        """
        Convert positive variance parameters back to the
        unconstrained optimization scale.
        """
        return np.log(
            constrained
        )

    def update(
        self,
        params: np.ndarray,
        transformed: bool = True,
        includes_fixed: bool = False,
        complex_step: bool = False,
    ) -> np.ndarray:
        """
        Update observation and state covariance matrices.

        The parameter dtype is preserved so that statsmodels'
        complex-step numerical differentiation works without
        discarding imaginary components.
        """
        params = super().update(
            params,
            transformed=transformed,
            includes_fixed=includes_fixed,
            complex_step=complex_step,
        )

        parameter_dtype = np.asarray(
            params
        ).dtype

        observation_variance = params[0]
        intercept_state_variance = params[1]
        slope_state_variance = params[2]

        observation_covariance = np.zeros(
            shape=(1, 1),
            dtype=parameter_dtype,
        )

        observation_covariance[0, 0] = (
            observation_variance
        )

        self.ssm["obs_cov"] = (
            observation_covariance
        )

        state_variances = np.empty(
            self.number_of_states,
            dtype=parameter_dtype,
        )

        state_variances[0] = (
            intercept_state_variance
        )

        state_variances[1:] = (
            slope_state_variance
        )

        self.ssm["state_cov"] = np.diag(
            state_variances
        )

        return params


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_qlike(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate QLIKE using log realized variance.

    If:
        delta = log(RV_true) - log(RV_pred)

    then:
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
    on the realized-variance scale.
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
# Load and prepare data
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load and validate the standardized model table.
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
                f"Unexpected {split_name} count: "
                f"expected {expected_count}, "
                f"found {actual_count}."
            )

    model_ready_values = evaluation[
        STANDARDIZED_FEATURES
        + ["y_next_hour"]
    ]

    if model_ready_values.isna().any().any():
        raise RuntimeError(
            "Missing values were detected among "
            "model-ready observations."
        )

    if not np.isfinite(
        model_ready_values.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite model-ready values were detected."
        )

    return data


def prepare_model_calendar(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retain every clock hour from the first model-ready training
    observation through the end of validation.

    Invalid observations remain missing endogenous values, so
    calendar time is never compressed.
    """
    first_model_origin = data.loc[
        (data["split"] == "train")
        & data["is_model_ready"].astype(bool),
        "forecast_origin_utc",
    ].min()

    model_calendar = (
        data.loc[
            (
                data["forecast_origin_utc"]
                >= first_model_origin
            )
            & data["split"].isin(
                ["train", "validation"]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    model_calendar["endog_for_filter"] = np.where(
        model_calendar[
            "is_model_ready"
        ].astype(bool),
        model_calendar["y_next_hour"],
        np.nan,
    )

    return model_calendar


def build_design_matrix(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Construct the observation design matrix:

        [1, x_1, ..., x_7]

    Regressors are filled with zero only on observations whose
    endogenous variable is missing. Such rows contribute no
    observation likelihood.
    """
    ready_mask = (
        data["is_model_ready"].astype(bool)
    )

    if data.loc[
        ready_mask,
        STANDARDIZED_FEATURES,
    ].isna().any().any():
        raise RuntimeError(
            "A model-ready observation has missing regressors."
        )

    feature_matrix = (
        data[STANDARDIZED_FEATURES]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    constant = np.ones(
        shape=(len(data), 1),
        dtype=float,
    )

    design_matrix = np.concatenate(
        [
            constant,
            feature_matrix,
        ],
        axis=1,
    )

    if design_matrix.shape[1] != len(
        STATE_TERMS
    ):
        raise RuntimeError(
            "The design-matrix dimension is incorrect."
        )

    return design_matrix


# --------------------------------------------------
# Initial fixed coefficients
# --------------------------------------------------

def estimate_initial_state(
    model_calendar: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """
    Estimate a fixed HAR-X model on training observations.

    The fixed coefficients initialize the time-varying state.
    """
    training = model_calendar.loc[
        (model_calendar["split"] == "train")
        & model_calendar[
            "is_model_ready"
        ].astype(bool)
    ]

    x_training = sm.add_constant(
        training[
            STANDARDIZED_FEATURES
        ].astype(float),
        has_constant="add",
    )

    y_training = training[
        "y_next_hour"
    ].astype(float)

    fitted = sm.OLS(
        y_training,
        x_training,
    ).fit()

    initial_state = (
        fitted.params
        .reindex(STATE_TERMS)
        .to_numpy(dtype=float)
    )

    residual_variance = float(
        np.mean(
            np.asarray(
                fitted.resid,
                dtype=float,
            ) ** 2
        )
    )

    if not np.isfinite(
        initial_state
    ).all():
        raise RuntimeError(
            "Initial fixed HAR-X coefficients are invalid."
        )

    if (
        residual_variance <= 0
        or not np.isfinite(
            residual_variance
        )
    ):
        raise RuntimeError(
            "Initial observation variance is invalid."
        )

    return (
        initial_state,
        residual_variance,
    )


# --------------------------------------------------
# Maximum-likelihood estimation
# --------------------------------------------------

def fit_training_tvp_model(
    model_calendar: pd.DataFrame,
    initial_state: np.ndarray,
    initial_observation_variance: float,
):
    """
    Estimate the observation variance, intercept-state variance
    and shared slope-state variance using training data only.

    Multiple starting values are tried to check numerical
    stability.
    """
    training_calendar = (
        model_calendar.loc[
            model_calendar["split"] == "train"
        ]
        .copy()
        .reset_index(drop=True)
    )

    training_design = build_design_matrix(
        training_calendar
    )

    initial_covariance = (
        np.eye(
            len(STATE_TERMS)
        )
        * INITIAL_STATE_VARIANCE
    )

    model = TVPHARXModel(
        endog=training_calendar[
            "endog_for_filter"
        ].to_numpy(dtype=float),
        design_matrix=training_design,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )

    starting_values = [
        np.array(
            [
                initial_observation_variance,
                1e-5,
                1e-6,
            ],
            dtype=float,
        ),
        np.array(
            [
                initial_observation_variance,
                1e-4,
                1e-5,
            ],
            dtype=float,
        ),
        np.array(
            [
                initial_observation_variance,
                1e-3,
                1e-4,
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
                np.isfinite(result.llf)
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
            "All TVP-HAR-X optimization runs failed."
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

    if not np.isfinite(
        best_result.params
    ).all():
        raise RuntimeError(
            "The selected variance parameters are non-finite."
        )

    if not (
        best_result.params > 0
    ).all():
        raise RuntimeError(
            "The selected variance parameters are not positive."
        )

    return (
        best_result,
        optimization_table,
    )


# --------------------------------------------------
# Full sequential filtering
# --------------------------------------------------

def filter_train_validation(
    model_calendar: pd.DataFrame,
    initial_state: np.ndarray,
    fitted_parameters: np.ndarray,
):
    """
    Apply the training-estimated variance parameters to the
    complete training-validation calendar.

    At each observation, the Kalman filter first generates the
    one-step-ahead forecast and then uses the realized outcome
    to update the state for the following hour.
    """
    design_matrix = build_design_matrix(
        model_calendar
    )

    initial_covariance = (
        np.eye(
            len(STATE_TERMS)
        )
        * INITIAL_STATE_VARIANCE
    )

    model = TVPHARXModel(
        endog=model_calendar[
            "endog_for_filter"
        ].to_numpy(dtype=float),
        design_matrix=design_matrix,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )

    result = model.filter(
        fitted_parameters
    )

    return result


# --------------------------------------------------
# Predictions and state paths
# --------------------------------------------------

def build_prediction_table(
    model_calendar: pd.DataFrame,
    result,
) -> pd.DataFrame:
    """
    Extract genuine one-step-ahead Kalman forecasts.
    """
    forecasts = (
        result
        .filter_results
        .forecasts[0]
    )

    forecast_variances = (
        result
        .filter_results
        .forecasts_error_cov[0, 0]
    )

    standardized_innovations = (
        result
        .filter_results
        .standardized_forecasts_error[0]
    )

    ready_mask = (
        model_calendar[
            "is_model_ready"
        ].to_numpy(dtype=bool)
    )

    ready_positions = np.flatnonzero(
        ready_mask
    )

    output = (
        model_calendar.loc[
            ready_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    output["y_true"] = (
        output["y_next_hour"]
    )

    output["y_pred_tvp_harx"] = (
        forecasts[ready_positions]
    )

    output["forecast_variance"] = (
        forecast_variances[
            ready_positions
        ]
    )

    output["innovation"] = (
        output["y_true"]
        - output["y_pred_tvp_harx"]
    )

    output["standardized_innovation"] = (
        standardized_innovations[
            ready_positions
        ]
    )

    output["absolute_log_error"] = (
        output["innovation"].abs()
    )

    output["squared_log_error"] = (
        output["innovation"] ** 2
    )

    output["qlike"] = calculate_qlike(
        y_true=output[
            "y_true"
        ].to_numpy(dtype=float),
        y_pred=output[
            "y_pred_tvp_harx"
        ].to_numpy(dtype=float),
    )

    validation_columns = [
        "y_true",
        "y_pred_tvp_harx",
        "forecast_variance",
        "innovation",
        "standardized_innovation",
        "qlike",
    ]

    if output[
        validation_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing TVP forecast values were detected."
        )

    if not np.isfinite(
        output[
            validation_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite TVP forecast values were detected."
        )

    if (
        output["forecast_variance"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive forecast variances were detected."
        )

    actual_counts = (
        output["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_COUNTS.items()
    ):
        actual_count = int(
            actual_counts.get(
                split_name,
                0,
            )
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"Unexpected TVP prediction count for "
                f"{split_name}: expected {expected_count}, "
                f"found {actual_count}."
            )

    return output


def build_state_path_table(
    model_calendar: pd.DataFrame,
    result,
) -> pd.DataFrame:
    """
    Save predicted states used for forecasting and filtered
    states obtained after observing each outcome.
    """
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

    if predicted_states.shape != filtered_states.shape:
        raise RuntimeError(
            "Predicted and filtered state arrays "
            "have inconsistent dimensions."
        )

    if predicted_states.shape[0] != len(
        model_calendar
    ):
        raise RuntimeError(
            "State-path length does not match the calendar."
        )

    state_table = model_calendar[
        [
            "forecast_origin_utc",
            "split",
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

    state_value_columns = [
        column
        for column in state_table.columns
        if (
            column.startswith("predicted_")
            or column.startswith("filtered_")
        )
    ]

    if not np.isfinite(
        state_table[
            state_value_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite state estimates were detected."
        )

    return state_table


def build_metric_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate training and validation forecast metrics.
    """
    rows = []

    for split_name in [
        "train",
        "validation",
    ]:
        split_data = predictions.loc[
            predictions["split"]
            == split_name
        ]

        metrics = calculate_metrics(
            y_true=split_data[
                "y_true"
            ].to_numpy(dtype=float),
            y_pred=split_data[
                "y_pred_tvp_harx"
            ].to_numpy(dtype=float),
        )

        rows.append(
            {
                "model": "TVP-HAR-X",
                "split": split_name,
                "n_observations": len(
                    split_data
                ),
                **metrics,
            }
        )

    metric_table = pd.DataFrame(
        rows
    )

    if not np.isfinite(
        metric_table[
            [
                "mae_log_rv",
                "rmse_log_rv",
                "qlike",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite TVP metrics were detected."
        )

    return metric_table


def build_state_summary(
    state_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize forecast-time coefficient variation during the
    validation period.
    """
    validation = state_table.loc[
        (
            state_table["split"]
            == "validation"
        )
        & state_table[
            "is_model_ready"
        ].astype(bool)
    ]

    rows = []

    for state_name in STATE_TERMS:
        column = (
            f"predicted_{state_name}"
        )

        values = validation[
            column
        ].astype(float)

        rows.append(
            {
                "state": state_name,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
                "first": values.iloc[0],
                "last": values.iloc[-1],
            }
        )

    return pd.DataFrame(
        rows
    )


# --------------------------------------------------
# Innovation diagnostics
# --------------------------------------------------

def find_longest_contiguous_segment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the longest group of observations whose forecast
    origins remain exactly one hour apart.
    """
    data = (
        data
        .sort_values(
            "forecast_origin_utc"
        )
        .reset_index(drop=True)
        .copy()
    )

    new_segment = (
        data["forecast_origin_utc"]
        .diff()
        .ne(
            pd.Timedelta(hours=1)
        )
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
        .drop(
            columns="_segment_id"
        )
        .reset_index(drop=True)
    )

    return longest_segment


def run_innovation_diagnostics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run Ljung-Box tests on standardized one-step-ahead
    innovations from the longest continuous training segment.
    """
    training = predictions.loc[
        predictions["split"]
        == "train"
    ].copy()

    continuous = (
        find_longest_contiguous_segment(
            training
        )
    )

    innovations = continuous[
        "standardized_innovation"
    ].to_numpy(dtype=float)

    ljung_box = acorr_ljungbox(
        innovations,
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

    ljung_box[
        "continuous_segment_start"
    ] = continuous[
        "forecast_origin_utc"
    ].iloc[0]

    ljung_box[
        "continuous_segment_end"
    ] = continuous[
        "forecast_origin_utc"
    ].iloc[-1]

    ljung_box[
        "n_innovations"
    ] = len(
        innovations
    )

    ljung_box[
        "innovation_mean"
    ] = float(
        np.mean(
            innovations
        )
    )

    ljung_box[
        "innovation_std"
    ] = float(
        np.std(
            innovations,
            ddof=1,
        )
    )

    ljung_box[
        "innovation_lag1_autocorrelation"
    ] = float(
        pd.Series(
            innovations
        ).autocorr(lag=1)
    )

    return ljung_box


# --------------------------------------------------
# Baseline comparison
# --------------------------------------------------

def read_selected_ar_result() -> pd.Series:
    """
    Read the previously selected AR order and validation metrics.
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
            "The AR selection file has no selected column."
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
            "Expected exactly one selected AR result."
        )

    return selected.iloc[0]


def build_validation_comparison(
    tvp_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare validation results across all four models.
    """
    required_files = [
        NAIVE_METRIC_PATH,
        FIXED_HARX_METRIC_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required metric file not found:\n{path}"
            )

    naive = pd.read_csv(
        NAIVE_METRIC_PATH
    )

    fixed = pd.read_csv(
        FIXED_HARX_METRIC_PATH
    )

    selected_ar = (
        read_selected_ar_result()
    )

    naive_validation = naive.loc[
        naive["split"]
        == "validation"
    ]

    fixed_validation = fixed.loc[
        fixed["split"]
        == "validation"
    ]

    tvp_validation = tvp_metrics.loc[
        tvp_metrics["split"]
        == "validation"
    ]

    if len(naive_validation) != 1:
        raise RuntimeError(
            "Expected one Naive validation result."
        )

    if len(fixed_validation) != 1:
        raise RuntimeError(
            "Expected one fixed HAR-X validation result."
        )

    if len(tvp_validation) != 1:
        raise RuntimeError(
            "Expected one TVP-HAR-X validation result."
        )

    naive_validation = (
        naive_validation.iloc[0]
    )

    fixed_validation = (
        fixed_validation.iloc[0]
    )

    tvp_validation = (
        tvp_validation.iloc[0]
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
                    naive_validation[
                        "qlike"
                    ]
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
                    fixed_validation[
                        "mae_log_rv"
                    ]
                ),
                "rmse_log_rv": float(
                    fixed_validation[
                        "rmse_log_rv"
                    ]
                ),
                "qlike": float(
                    fixed_validation[
                        "qlike"
                    ]
                ),
            },
            {
                "model": "TVP-HAR-X",
                "mae_log_rv": float(
                    tvp_validation[
                        "mae_log_rv"
                    ]
                ),
                "rmse_log_rv": float(
                    tvp_validation[
                        "rmse_log_rv"
                    ]
                ),
                "qlike": float(
                    tvp_validation[
                        "qlike"
                    ]
                ),
            },
        ]
    )

    naive_qlike = float(
        comparison.loc[
            comparison["model"]
            == "Naive",
            "qlike",
        ].iloc[0]
    )

    fixed_qlike = float(
        comparison.loc[
            comparison["model"]
            == "Fixed HAR-X",
            "qlike",
        ].iloc[0]
    )

    comparison[
        "qlike_improvement_vs_naive_percent"
    ] = (
        (
            naive_qlike
            - comparison["qlike"]
        )
        / naive_qlike
        * 100.0
    )

    comparison[
        "qlike_improvement_vs_fixed_harx_percent"
    ] = (
        (
            fixed_qlike
            - comparison["qlike"]
        )
        / fixed_qlike
        * 100.0
    )

    return comparison


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def save_outputs(
    fitted_result,
    optimization_table: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    state_table: pd.DataFrame,
    state_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
) -> None:
    """
    Save all TVP-HAR-X estimation, prediction and diagnostic
    outputs.
    """
    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_table = pd.DataFrame(
        [
            {
                "parameter": (
                    "observation_variance"
                ),
                "value": float(
                    fitted_result.params[0]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        fitted_result.params[0]
                    )
                ),
                "estimated_from": "train",
            },
            {
                "parameter": (
                    "intercept_state_variance"
                ),
                "value": float(
                    fitted_result.params[1]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        fitted_result.params[1]
                    )
                ),
                "estimated_from": "train",
            },
            {
                "parameter": (
                    "slope_state_variance"
                ),
                "value": float(
                    fitted_result.params[2]
                ),
                "standard_deviation": float(
                    np.sqrt(
                        fitted_result.params[2]
                    )
                ),
                "estimated_from": "train",
            },
        ]
    )

    parameter_table.to_csv(
        TVP_PARAMETER_PATH,
        index=False,
    )

    optimization_table.to_csv(
        OPTIMIZATION_OUTPUT_PATH,
        index=False,
    )

    metrics.to_csv(
        TVP_METRIC_PATH,
        index=False,
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    state_summary.to_csv(
        STATE_SUMMARY_PATH,
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
        "y_pred_tvp_harx",
        "forecast_variance",
        "innovation",
        "standardized_innovation",
        "absolute_log_error",
        "squared_log_error",
        "qlike",
    ]

    prediction_output = predictions[
        prediction_columns
    ].copy()

    if prediction_output[
        "split"
    ].eq("test").any():
        raise RuntimeError(
            "Test predictions were included unexpectedly."
        )

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

    STATE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_state_path = (
        STATE_OUTPUT_PATH.with_suffix(
            ".tmp.parquet"
        )
    )

    if temporary_state_path.exists():
        temporary_state_path.unlink()

    state_table.to_parquet(
        temporary_state_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    temporary_state_path.replace(
        STATE_OUTPUT_PATH
    )

    saved_predictions = pd.read_parquet(
        PREDICTION_OUTPUT_PATH,
        columns=[
            "split",
            "y_true",
            "y_pred_tvp_harx",
        ],
    )

    if len(saved_predictions) != len(
        prediction_output
    ):
        raise RuntimeError(
            "Saved prediction row count is incorrect."
        )

    saved_states = pd.read_parquet(
        STATE_OUTPUT_PATH
    )

    if len(saved_states) != len(
        state_table
    ):
        raise RuntimeError(
            "Saved state-path row count is incorrect."
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 88)
    print("TVP-HAR-X State-Space Estimation and Sequential Validation")
    print("=" * 88)
    print(f"Input file: {INPUT_PATH}")
    print(
        "Variance parameters are estimated using "
        "training data only."
    )
    print(
        "Test-set performance remains sealed."
    )
    print()

    data = load_data()

    model_calendar = prepare_model_calendar(
        data
    )

    (
        initial_state,
        initial_observation_variance,
    ) = estimate_initial_state(
        model_calendar
    )

    (
        fitted_result,
        optimization_table,
    ) = fit_training_tvp_model(
        model_calendar=model_calendar,
        initial_state=initial_state,
        initial_observation_variance=(
            initial_observation_variance
        ),
    )

    full_result = filter_train_validation(
        model_calendar=model_calendar,
        initial_state=initial_state,
        fitted_parameters=fitted_result.params,
    )

    predictions = build_prediction_table(
        model_calendar=model_calendar,
        result=full_result,
    )

    state_table = build_state_path_table(
        model_calendar=model_calendar,
        result=full_result,
    )

    metrics = build_metric_table(
        predictions
    )

    state_summary = build_state_summary(
        state_table
    )

    ljung_box = run_innovation_diagnostics(
        predictions
    )

    comparison = build_validation_comparison(
        metrics
    )

    print("Optimization runs:")
    print(
        optimization_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10g}"
            ),
        )
    )

    print()
    print("Selected variance parameters:")
    print(
        f"Observation variance:       "
        f"{fitted_result.params[0]:.12e}"
    )
    print(
        f"Intercept state variance:   "
        f"{fitted_result.params[1]:.12e}"
    )
    print(
        f"Slope state variance:       "
        f"{fitted_result.params[2]:.12e}"
    )
    print(
        f"Intercept state std:        "
        f"{np.sqrt(fitted_result.params[1]):.12e}"
    )
    print(
        f"Slope state std:            "
        f"{np.sqrt(fitted_result.params[2]):.12e}"
    )
    print(
        f"Training log likelihood:    "
        f"{fitted_result.llf:.10f}"
    )
    print(
        f"Optimizer converged:         "
        f"{bool(fitted_result.mle_retvals.get('converged', False))}"
    )

    print()
    print("TVP-HAR-X metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print("Validation model comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "Validation time-varying coefficient summary:"
    )
    print(
        state_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    print()
    print(
        "TVP-HAR-X standardized innovation "
        "Ljung-Box tests:"
    )
    print(
        ljung_box.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10f}"
            ),
        )
    )

    save_outputs(
        fitted_result=fitted_result,
        optimization_table=optimization_table,
        metrics=metrics,
        comparison=comparison,
        predictions=predictions,
        state_table=state_table,
        state_summary=state_summary,
        ljung_box=ljung_box,
    )

    print()
    print(
        f"Saved metrics to:     "
        f"{TVP_METRIC_PATH}"
    )
    print(
        f"Saved predictions to: "
        f"{PREDICTION_OUTPUT_PATH}"
    )
    print(
        f"Saved state paths to: "
        f"{STATE_OUTPUT_PATH}"
    )

    print()
    print("[TVP-HAR-X Analysis] PASS")
    print(
        "[Success] The TVP-HAR-X variance parameters were "
        "estimated using training data and evaluated through "
        "sequential one-hour-ahead validation forecasts "
        "without accessing test performance."
    )


if __name__ == "__main__":
    main()