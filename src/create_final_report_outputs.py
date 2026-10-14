from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "final_test_predictions_all_models.parquet"
)

METRIC_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_test_model_metrics.csv"
)

LOSS_TEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "final_test_hac_loss_differential_tests.csv"
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

SUMMARY_OUTPUT_PATH = (
    TABLE_OUTPUT_DIR
    / "final_report_model_summary.csv"
)

MONTHLY_OUTPUT_PATH = (
    TABLE_OUTPUT_DIR
    / "final_test_monthly_metrics.csv"
)

MODEL_ORDER = [
    "Naive",
    "AR(24)",
    "Fixed HAR-X",
    "TVP-HAR-X",
]

MODEL_COLUMNS = {
    "Naive": {
        "prediction": "y_pred_naive",
        "qlike": "qlike_naive",
    },
    "AR(24)": {
        "prediction": "y_pred_ar_24",
        "qlike": "qlike_ar24",
    },
    "Fixed HAR-X": {
        "prediction": "y_pred_fixed_harx",
        "qlike": "qlike_fixed_har_x",
    },
    "TVP-HAR-X": {
        "prediction": "y_pred_tvp_harx",
        "qlike": "qlike_tvp_har_x",
    },
}

ROLLING_WINDOW = 168


# --------------------------------------------------
# Load
# --------------------------------------------------

def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    required_paths = [
        PREDICTION_PATH,
        METRIC_PATH,
        LOSS_TEST_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    predictions = pd.read_parquet(
        PREDICTION_PATH
    )

    metrics = pd.read_csv(
        METRIC_PATH
    )

    loss_tests = pd.read_csv(
        LOSS_TEST_PATH
    )

    predictions["forecast_origin_utc"] = (
        pd.to_datetime(
            predictions["forecast_origin_utc"],
            utc=True,
        )
    )

    required_prediction_columns = [
        "forecast_origin_utc",
        "y_true",
    ]

    for model_information in MODEL_COLUMNS.values():
        required_prediction_columns.extend(
            [
                model_information["prediction"],
                model_information["qlike"],
            ]
        )

    missing_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    if len(predictions) != 8422:
        raise RuntimeError(
            "Unexpected final test sample size: "
            f"{len(predictions)}"
        )

    return predictions, metrics, loss_tests


# --------------------------------------------------
# Summary tables
# --------------------------------------------------

def build_summary_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        metrics
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
        .copy()
    )

    naive_row = summary.loc[
        summary["model"] == "Naive"
    ].iloc[0]

    for metric in [
        "mae_log_rv",
        "rmse_log_rv",
        "qlike",
    ]:
        summary[
            f"{metric}_improvement_vs_naive_percent"
        ] = (
            (
                naive_row[metric]
                - summary[metric]
            )
            / naive_row[metric]
            * 100.0
        )

        summary[
            f"{metric}_rank"
        ] = (
            summary[metric]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

    return summary


def build_monthly_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    data = predictions.copy()

    data["month"] = (
        data["forecast_origin_utc"]
        .dt.strftime("%Y-%m")
    )

    rows = []

    for month, month_data in data.groupby(
        "month",
        sort=True,
    ):
        y_true = month_data[
            "y_true"
        ].to_numpy(dtype=float)

        for model_name in MODEL_ORDER:
            prediction_column = (
                MODEL_COLUMNS[
                    model_name
                ]["prediction"]
            )

            qlike_column = (
                MODEL_COLUMNS[
                    model_name
                ]["qlike"]
            )

            y_pred = month_data[
                prediction_column
            ].to_numpy(dtype=float)

            errors = y_true - y_pred

            rows.append(
                {
                    "month": month,
                    "model": model_name,
                    "n_observations": len(
                        month_data
                    ),
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
                        month_data[
                            qlike_column
                        ].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------
# Figure helpers
# --------------------------------------------------

def save_metric_bar_chart(
    summary: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    values = (
        summary
        .set_index("model")
        .reindex(MODEL_ORDER)[
            metric_column
        ]
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        values.index,
        values.values,
    )

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)

    upper_limit = float(
        values.max()
    ) * 1.15

    plt.ylim(
        0,
        upper_limit,
    )

    for bar, value in zip(
        bars,
        values.values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / filename,
        dpi=200,
    )

    plt.close(figure)


def save_monthly_qlike_chart(
    monthly_metrics: pd.DataFrame,
) -> None:
    pivot = (
        monthly_metrics
        .pivot(
            index="month",
            columns="model",
            values="qlike",
        )
        .reindex(
            columns=MODEL_ORDER
        )
    )

    figure = plt.figure(
        figsize=(11, 5)
    )

    for model_name in MODEL_ORDER:
        plt.plot(
            pivot.index,
            pivot[model_name],
            marker="o",
            linewidth=1.5,
            label=model_name,
        )

    plt.title(
        "Monthly Test QLIKE by Model"
    )
    plt.xlabel("Month")
    plt.ylabel("Mean QLIKE")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / "final_test_monthly_qlike.png",
        dpi=200,
    )

    plt.close(figure)


def save_cumulative_qlike_difference(
    predictions: pd.DataFrame,
) -> None:
    """
    Positive values mean cumulative TVP-HAR-X QLIKE is lower
    than cumulative Fixed HAR-X QLIKE.
    """
    data = predictions.copy()

    data["qlike_difference"] = (
        data["qlike_fixed_har_x"]
        - data["qlike_tvp_har_x"]
    )

    data[
        "cumulative_qlike_difference"
    ] = (
        data["qlike_difference"]
        .cumsum()
    )

    figure = plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        data["forecast_origin_utc"],
        data[
            "cumulative_qlike_difference"
        ],
        linewidth=1.2,
    )

    plt.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    plt.title(
        "Cumulative QLIKE Difference: "
        "Fixed HAR-X Minus TVP-HAR-X"
    )
    plt.xlabel("Forecast origin (UTC)")
    plt.ylabel(
        "Cumulative loss difference\n"
        "(positive means TVP-HAR-X is better)"
    )
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / (
            "final_test_cumulative_qlike_"
            "difference_fixed_vs_tvp.png"
        ),
        dpi=200,
    )

    plt.close(figure)


def save_rolling_qlike_difference(
    predictions: pd.DataFrame,
) -> None:
    """
    Plot a 168-hour rolling QLIKE difference.

    Positive values indicate lower rolling QLIKE for TVP-HAR-X.
    """
    data = predictions.copy()

    data["qlike_difference"] = (
        data["qlike_fixed_har_x"]
        - data["qlike_tvp_har_x"]
    )

    data[
        "rolling_168h_qlike_difference"
    ] = (
        data["qlike_difference"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        )
        .mean()
    )

    figure = plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        data["forecast_origin_utc"],
        data[
            "rolling_168h_qlike_difference"
        ],
        linewidth=1.0,
    )

    plt.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    plt.title(
        "168-Hour Rolling QLIKE Difference: "
        "Fixed HAR-X Minus TVP-HAR-X"
    )
    plt.xlabel("Forecast origin (UTC)")
    plt.ylabel(
        "Rolling loss difference\n"
        "(positive means TVP-HAR-X is better)"
    )
    plt.tight_layout()

    figure.savefig(
        FIGURE_OUTPUT_DIR
        / (
            "final_test_rolling_168h_qlike_"
            "difference_fixed_vs_tvp.png"
        ),
        dpi=200,
    )

    plt.close(figure)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 84)
    print("Create Final Report Tables and Figures")
    print("=" * 84)
    print(f"Prediction file: {PREDICTION_PATH}")
    print(f"Metric file:     {METRIC_PATH}")
    print()

    (
        predictions,
        metrics,
        loss_tests,
    ) = load_inputs()

    summary = build_summary_table(
        metrics
    )

    monthly_metrics = (
        build_monthly_metrics(
            predictions
        )
    )

    TABLE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    monthly_metrics.to_csv(
        MONTHLY_OUTPUT_PATH,
        index=False,
    )

    save_metric_bar_chart(
        summary=summary,
        metric_column="mae_log_rv",
        ylabel="MAE on log realized variance",
        title="Final Test MAE by Model",
        filename="final_test_mae_comparison.png",
    )

    save_metric_bar_chart(
        summary=summary,
        metric_column="rmse_log_rv",
        ylabel="RMSE on log realized variance",
        title="Final Test RMSE by Model",
        filename="final_test_rmse_comparison.png",
    )

    save_metric_bar_chart(
        summary=summary,
        metric_column="qlike",
        ylabel="QLIKE",
        title="Final Test QLIKE by Model",
        filename="final_test_qlike_comparison.png",
    )

    save_monthly_qlike_chart(
        monthly_metrics
    )

    save_cumulative_qlike_difference(
        predictions
    )

    save_rolling_qlike_difference(
        predictions
    )

    primary_test = loss_tests.loc[
        (
            loss_tests["benchmark"]
            == "Fixed HAR-X"
        )
        & (
            loss_tests["loss"]
            == "QLIKE"
        )
    ]

    if len(primary_test) != 1:
        raise RuntimeError(
            "Primary Fixed HAR-X versus TVP-HAR-X "
            "QLIKE test was not found."
        )

    print("Final report model summary:")
    print(
        summary[
            [
                "model",
                "mae_log_rv",
                "rmse_log_rv",
                "qlike",
                "mae_log_rv_rank",
                "rmse_log_rv_rank",
                "qlike_rank",
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
        "Primary QLIKE test: "
        "TVP-HAR-X versus Fixed HAR-X"
    )
    print(
        primary_test[
            [
                "mean_loss_difference",
                "improvement_percent",
                "test_statistic",
                "one_sided_p_value",
                "two_sided_p_value",
                (
                    "tvp_significantly_better_"
                    "at_5_percent"
                ),
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
        f"Saved final summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )
    print(
        f"Saved monthly metrics to: "
        f"{MONTHLY_OUTPUT_PATH}"
    )
    print(
        f"Saved figures to: "
        f"{FIGURE_OUTPUT_DIR}"
    )

    print()
    print("[Final Report Outputs] PASS")


if __name__ == "__main__":
    main()