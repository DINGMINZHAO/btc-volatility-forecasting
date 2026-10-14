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
    / "btcusdt_model_table_with_splits_2022_2025.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_model_table_transformed_2022_2025.parquet"
)

PARAMETER_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "feature_transformation_parameters.csv"
)

RAW_FEATURE = "downside_semivariance"
TRANSFORMED_FEATURE = "scaled_log_downside_semivariance"


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the model table before applying the training-based
    downside-semivariance transformation.
    """
    required_columns = [
        "forecast_origin_utc",
        "split",
        "is_model_ready",
        RAW_FEATURE,
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

    nonmissing_raw = data[RAW_FEATURE].dropna()

    if not np.isfinite(
        nonmissing_raw.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite downside-semivariance values "
            "were detected."
        )

    if (nonmissing_raw < 0).any():
        raise ValueError(
            "Negative downside semivariance was detected."
        )

    return data


def estimate_training_scale(
    data: pd.DataFrame,
) -> float:
    """
    Estimate the transformation scale using positive,
    model-ready training observations only.
    """
    training_positive = data.loc[
        (data["split"] == "train")
        & data["is_model_ready"].astype(bool)
        & data[RAW_FEATURE].gt(0),
        RAW_FEATURE,
    ]

    if training_positive.empty:
        raise RuntimeError(
            "No positive model-ready training values were found."
        )

    scale = float(training_positive.median())

    if scale <= 0 or not np.isfinite(scale):
        raise RuntimeError(
            "The training transformation scale is invalid."
        )

    return scale


def apply_transformation(
    data: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    """
    Apply the training-derived transformation to every split.

    Missing raw observations remain missing.
    """
    data = data.copy()

    data[TRANSFORMED_FEATURE] = np.nan

    available_mask = data[RAW_FEATURE].notna()

    data.loc[
        available_mask,
        TRANSFORMED_FEATURE,
    ] = np.log1p(
        data.loc[
            available_mask,
            RAW_FEATURE,
        ]
        / scale
    )

    return data


def validate_transformation(
    data: pd.DataFrame,
    scale: float,
) -> None:
    """
    Validate missing values, zeros, finiteness and the exact
    transformation formula.
    """
    raw_missing = data[RAW_FEATURE].isna()
    transformed_missing = data[TRANSFORMED_FEATURE].isna()

    if not raw_missing.equals(transformed_missing):
        raise RuntimeError(
            "The transformed feature does not preserve "
            "the raw feature's missing-value pattern."
        )

    available_mask = data[RAW_FEATURE].notna()

    transformed_values = data.loc[
        available_mask,
        TRANSFORMED_FEATURE,
    ]

    if not np.isfinite(
        transformed_values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Non-finite transformed values were detected."
        )

    zero_mask = data[RAW_FEATURE].eq(0)

    if not data.loc[
        zero_mask,
        TRANSFORMED_FEATURE,
    ].eq(0).all():
        raise RuntimeError(
            "Raw zero values were not transformed to zero."
        )

    expected_values = np.log1p(
        data.loc[
            available_mask,
            RAW_FEATURE,
        ].to_numpy(dtype=float)
        / scale
    )

    actual_values = data.loc[
        available_mask,
        TRANSFORMED_FEATURE,
    ].to_numpy(dtype=float)

    if not np.allclose(
        actual_values,
        expected_values,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The transformed feature does not match "
            "the required formula."
        )

    model_ready = data["is_model_ready"].astype(bool)

    if data.loc[
        model_ready,
        TRANSFORMED_FEATURE,
    ].isna().any():
        raise RuntimeError(
            "Missing transformed values were found among "
            "model-ready observations."
        )


def save_outputs(
    data: pd.DataFrame,
    scale: float,
) -> None:
    """
    Save the transformed model table and transformation parameter.
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

    parameter_table = pd.DataFrame(
        [
            {
                "feature": RAW_FEATURE,
                "transformation": (
                    "log1p(raw_value / training_positive_median)"
                ),
                "parameter_name": "training_positive_median",
                "parameter_value": scale,
                "estimated_from_split": "train",
                "estimated_from_model_ready_only": True,
            }
        ]
    )

    PARAMETER_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_table.to_csv(
        PARAMETER_OUTPUT_PATH,
        index=False,
    )

    verification = pd.read_parquet(
        OUTPUT_PATH,
        columns=[
            RAW_FEATURE,
            TRANSFORMED_FEATURE,
            "split",
            "is_model_ready",
        ],
    )

    if len(verification) != len(data):
        raise RuntimeError(
            "Saved row count does not match the source table."
        )

    if int(
        verification[
            TRANSFORMED_FEATURE
        ].isna().sum()
    ) != int(
        data[
            TRANSFORMED_FEATURE
        ].isna().sum()
    ):
        raise RuntimeError(
            "Saved transformed-feature missing count "
            "is incorrect."
        )


def main() -> None:
    print("=" * 80)
    print("Apply Training-Based Downside-Semivariance Transformation")
    print("=" * 80)
    print(f"Input file:     {INPUT_PATH}")
    print(f"Output file:    {OUTPUT_PATH}")
    print(f"Parameter file: {PARAMETER_OUTPUT_PATH}")
    print()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    data = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(data)}")

    data = validate_input(data)

    scale = estimate_training_scale(data)

    data = apply_transformation(
        data=data,
        scale=scale,
    )

    validate_transformation(
        data=data,
        scale=scale,
    )

    print()
    print("Transformation summary:")
    print(
        f"Training positive median: "
        f"{scale:.16e}"
    )
    print(
        f"Raw feature missing values: "
        f"{int(data[RAW_FEATURE].isna().sum())}"
    )
    print(
        f"Transformed feature missing values: "
        f"{int(data[TRANSFORMED_FEATURE].isna().sum())}"
    )
    print(
        f"Transformed zero values: "
        f"{int((data[TRANSFORMED_FEATURE] == 0).sum())}"
    )

    print()
    print("Transformed feature by split:")

    split_summary = (
        data.loc[
            data["is_model_ready"].astype(bool)
        ]
        .groupby("split")
        [TRANSFORMED_FEATURE]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "median",
                "max",
            ]
        )
    )

    print(
        split_summary.to_string(
            float_format=lambda value: f"{value:.6f}",
        )
    )

    save_outputs(
        data=data,
        scale=scale,
    )

    print()
    print(f"Saved transformed table to: {OUTPUT_PATH}")
    print(f"Saved parameter table to:   {PARAMETER_OUTPUT_PATH}")

    print()
    print("[Downside Transformation] PASS")
    print(
        "[Success] The training-derived downside transformation "
        "was applied consistently to training, validation, "
        "and test observations."
    )


if __name__ == "__main__":
    main()