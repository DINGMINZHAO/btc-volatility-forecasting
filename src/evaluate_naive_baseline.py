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
    / "btcusdt_model_table_transformed_2022_2025.parquet"
)

PREDICTION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "naive_baseline_predictions_train_validation.parquet"
)

METRIC_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "naive_baseline_metrics_train_validation.csv"
)

EVALUATION_SPLITS = [
    "train",
    "validation",
]

EXPECTED_SPLIT_COUNTS = {
    "train": 16673,
    "validation": 8610,
}


def load_evaluation_sample() -> pd.DataFrame:
    """
    Load the common model-ready training and validation sample.

    Test observations are deliberately excluded at this stage.
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

    evaluation = (
        data.loc[
            data["split"].isin(EVALUATION_SPLITS)
            & data["is_model_ready"].astype(bool),
            required_columns,
        ]
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )

    actual_counts = (
        evaluation["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_SPLIT_COUNTS.items()
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

    if evaluation["split"].eq("test").any():
        raise RuntimeError(
            "Test observations were included unexpectedly."
        )

    numeric_columns = [
        "y_current",
        "y_next_hour",
    ]

    if evaluation[numeric_columns].isna().any().any():
        raise RuntimeError(
            "Missing target values were detected in the "
            "evaluation sample."
        )

    if not np.isfinite(
        evaluation[numeric_columns]
        .to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite target values were detected."
        )

    time_difference = (
        evaluation["target_hour_end_utc"]
        - evaluation["forecast_origin_utc"]
    )

    if not time_difference.eq(
        pd.Timedelta(hours=1)
    ).all():
        raise RuntimeError(
            "At least one prediction target is not exactly "
            "one hour ahead."
        )

    if not evaluation[
        "target_hour_start_utc"
    ].eq(
        evaluation["forecast_origin_utc"]
    ).all():
        raise RuntimeError(
            "At least one target hour does not begin at "
            "the forecast origin."
        )

    return evaluation


def construct_naive_predictions(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the persistence forecast:

        y_hat_(t+1|t) = y_t
    """
    predictions = data.copy()

    predictions = predictions.rename(
        columns={
            "y_next_hour": "y_true",
        }
    )

    predictions["y_pred_naive"] = (
        predictions["y_current"]
    )

    predictions["log_error"] = (
        predictions["y_true"]
        - predictions["y_pred_naive"]
    )

    predictions["absolute_log_error"] = (
        predictions["log_error"].abs()
    )

    predictions["squared_log_error"] = (
        predictions["log_error"] ** 2
    )

    # QLIKE on the variance scale:
    #
    # actual_rv / predicted_rv
    #     - log(actual_rv / predicted_rv)
    #     - 1
    #
    # Since y = log(RV):
    #
    # log(actual_rv / predicted_rv)
    #     = y_true - y_pred
    #
    # expm1(delta) - delta is more numerically stable than
    # exp(delta) - delta - 1 when delta is close to zero.
    delta = predictions["log_error"].to_numpy(
        dtype=float
    )

    qlike = np.expm1(delta) - delta

    # Remove possible tiny negative values caused only by
    # floating-point rounding.
    predictions["qlike"] = np.maximum(
        qlike,
        0.0,
    )

    predictions["rv_true"] = np.exp(
        predictions["y_true"]
    )

    predictions["rv_pred_naive"] = np.exp(
        predictions["y_pred_naive"]
    )

    return predictions


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:
    """
    Validate the Naive prediction formula and loss values.
    """
    if not np.allclose(
        predictions["y_pred_naive"]
        .to_numpy(dtype=float),
        predictions["y_current"]
        .to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "Naive predictions do not equal y_current."
        )

    validation_columns = [
        "y_true",
        "y_pred_naive",
        "log_error",
        "absolute_log_error",
        "squared_log_error",
        "qlike",
        "rv_true",
        "rv_pred_naive",
    ]

    if predictions[
        validation_columns
    ].isna().any().any():
        raise RuntimeError(
            "Missing values were detected in the "
            "Naive prediction output."
        )

    if not np.isfinite(
        predictions[
            validation_columns
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite values were detected in the "
            "Naive prediction output."
        )

    if (
        predictions["absolute_log_error"] < 0
    ).any():
        raise RuntimeError(
            "Negative absolute errors were detected."
        )

    if (
        predictions["squared_log_error"] < 0
    ).any():
        raise RuntimeError(
            "Negative squared errors were detected."
        )

    if (
        predictions["qlike"] < 0
    ).any():
        raise RuntimeError(
            "Negative QLIKE values were detected."
        )

    if (
        predictions["rv_true"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive realized variances were detected."
        )

    if (
        predictions["rv_pred_naive"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive predicted variances were detected."
        )


def calculate_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, and QLIKE separately for training
    and validation.
    """
    metric_rows: list[dict[str, float | int | str]] = []

    for split_name in EVALUATION_SPLITS:
        split_data = predictions.loc[
            predictions["split"] == split_name
        ]

        if split_data.empty:
            raise RuntimeError(
                f"No observations found for {split_name}."
            )

        mae = float(
            split_data[
                "absolute_log_error"
            ].mean()
        )

        rmse = float(
            np.sqrt(
                split_data[
                    "squared_log_error"
                ].mean()
            )
        )

        qlike = float(
            split_data["qlike"].mean()
        )

        metric_rows.append(
            {
                "model": "Naive",
                "split": split_name,
                "n_observations": len(split_data),
                "mae_log_rv": mae,
                "rmse_log_rv": rmse,
                "qlike": qlike,
            }
        )

    metrics = pd.DataFrame(metric_rows)

    if not np.isfinite(
        metrics[
            [
                "mae_log_rv",
                "rmse_log_rv",
                "qlike",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite metric values were detected."
        )

    return metrics


def save_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """
    Save prediction-level results and aggregate metrics.
    """
    prediction_columns = [
        "forecast_origin_utc",
        "target_hour_start_utc",
        "target_hour_end_utc",
        "split",
        "y_current",
        "y_true",
        "y_pred_naive",
        "rv_true",
        "rv_pred_naive",
        "log_error",
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

    METRIC_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        METRIC_OUTPUT_PATH,
        index=False,
    )

    saved_predictions = pd.read_parquet(
        PREDICTION_OUTPUT_PATH,
        columns=[
            "split",
            "y_true",
            "y_pred_naive",
            "qlike",
        ],
    )

    if len(saved_predictions) != len(prediction_output):
        raise RuntimeError(
            "Saved prediction row count is incorrect."
        )

    if saved_predictions["split"].eq("test").any():
        raise RuntimeError(
            "Test predictions were saved unexpectedly."
        )


def main() -> None:
    print("=" * 80)
    print("Evaluate Naive One-Hour-Ahead Volatility Baseline")
    print("=" * 80)
    print(f"Input file:      {INPUT_PATH}")
    print(f"Prediction file: {PREDICTION_OUTPUT_PATH}")
    print(f"Metric file:     {METRIC_OUTPUT_PATH}")
    print()

    evaluation = load_evaluation_sample()

    print("Evaluation sample:")
    print(
        evaluation["split"]
        .value_counts()
        .reindex(EVALUATION_SPLITS)
        .to_string()
    )

    predictions = construct_naive_predictions(
        evaluation
    )

    validate_predictions(predictions)

    metrics = calculate_metrics(predictions)

    print()
    print("Naive baseline metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    print()
    print("First three validation predictions:")
    print(
        predictions.loc[
            predictions["split"] == "validation",
            [
                "forecast_origin_utc",
                "target_hour_end_utc",
                "y_current",
                "y_true",
                "y_pred_naive",
                "absolute_log_error",
                "qlike",
            ],
        ]
        .head(3)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.10f}",
        )
    )

    save_outputs(
        predictions=predictions,
        metrics=metrics,
    )

    print()
    print(f"Saved predictions to: {PREDICTION_OUTPUT_PATH}")
    print(f"Saved metrics to:     {METRIC_OUTPUT_PATH}")

    print()
    print("[Naive Baseline] PASS")
    print(
        "[Success] The persistence baseline was evaluated on "
        "the common training and validation samples without "
        "accessing test-set performance."
    )


if __name__ == "__main__":
    main()