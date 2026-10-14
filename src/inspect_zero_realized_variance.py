from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOURLY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "btcusdt_hourly_realized_variance_2022_2025.parquet"
)

RETURNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "btcusdt_5m_returns_2022_2025.parquet"
)

MAX_HOURS_TO_DISPLAY = 10


def main() -> None:
    print("=" * 75)
    print("Inspect Zero Hourly Realized Variance")
    print("=" * 75)
    print(f"Hourly file:  {HOURLY_PATH}")
    print(f"Returns file: {RETURNS_PATH}")
    print()

    if not HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n{HOURLY_PATH}"
        )

    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Five-minute return dataset not found:\n{RETURNS_PATH}"
        )

    hourly = pd.read_parquet(
        HOURLY_PATH,
        columns=[
            "hour_start_utc",
            "hour_end_utc",
            "n_returns",
            "realized_variance",
        ],
    )

    returns_5m = pd.read_parquet(
        RETURNS_PATH,
        columns=[
            "return_end_utc",
            "close",
            "log_return_5m",
        ],
    )

    # --------------------------------------------------
    # 1. Identify zero and positive realized variance
    # --------------------------------------------------

    zero_mask = hourly["realized_variance"] == 0
    negative_mask = hourly["realized_variance"] < 0
    positive_mask = hourly["realized_variance"] > 0

    zero_hours = (
        hourly.loc[zero_mask]
        .sort_values("hour_end_utc")
        .reset_index(drop=True)
    )

    positive_rv = hourly.loc[
        positive_mask,
        "realized_variance",
    ]

    print("Realized-variance counts:")
    print(f"Total hourly observations: {len(hourly)}")
    print(f"Zero RV hours:             {len(zero_hours)}")
    print(f"Positive RV hours:         {int(positive_mask.sum())}")
    print(f"Negative RV hours:         {int(negative_mask.sum())}")

    if negative_mask.any():
        raise ValueError(
            "Negative realized variance was detected."
        )

    if len(positive_rv) == 0:
        raise ValueError(
            "No positive realized-variance observations were found."
        )

    print()
    print("Small positive realized-variance values:")
    print(f"Minimum positive RV: {positive_rv.min():.16e}")
    print(
        f"0.01% quantile:      "
        f"{positive_rv.quantile(0.0001):.16e}"
    )
    print(
        f"0.10% quantile:      "
        f"{positive_rv.quantile(0.001):.16e}"
    )
    print(
        f"1.00% quantile:      "
        f"{positive_rv.quantile(0.01):.16e}"
    )

    if zero_hours.empty:
        print()
        print("[Zero RV Audit] No zero realized-variance hours found.")
        return

    print()
    print("Zero realized-variance hours:")
    print(
        zero_hours
        .head(MAX_HOURS_TO_DISPLAY)
        .to_string(index=False)
    )

    if len(zero_hours) > MAX_HOURS_TO_DISPLAY:
        print(
            f"... {len(zero_hours) - MAX_HOURS_TO_DISPLAY} "
            "additional zero-RV hours not displayed."
        )

    # --------------------------------------------------
    # 2. Verify the underlying five-minute returns
    # --------------------------------------------------

    audit_rows: list[dict[str, object]] = []

    for row in zero_hours.itertuples(index=False):
        hour_returns = returns_5m.loc[
            (
                returns_5m["return_end_utc"]
                > row.hour_start_utc
            )
            & (
                returns_5m["return_end_utc"]
                <= row.hour_end_utc
            )
        ].copy()

        n_rows = len(hour_returns)
        n_missing = int(
            hour_returns["log_return_5m"].isna().sum()
        )
        n_exact_zero = int(
            (hour_returns["log_return_5m"] == 0).sum()
        )

        reconstructed_rv = float(
            np.square(
                hour_returns["log_return_5m"]
                .to_numpy(dtype=float)
            ).sum()
        )

        audit_rows.append(
            {
                "hour_start_utc": row.hour_start_utc,
                "hour_end_utc": row.hour_end_utc,
                "n_5m_rows": n_rows,
                "n_missing_returns": n_missing,
                "n_exact_zero_returns": n_exact_zero,
                "unique_close_prices": int(
                    hour_returns["close"].nunique()
                ),
                "reconstructed_rv": reconstructed_rv,
            }
        )

    audit = pd.DataFrame(audit_rows)

    print()
    print("Underlying five-minute audit:")
    print(
        audit
        .head(MAX_HOURS_TO_DISPLAY)
        .to_string(index=False)
    )

    invalid_zero_hours = audit.loc[
        (audit["n_5m_rows"] != 12)
        | (audit["n_missing_returns"] != 0)
        | (audit["n_exact_zero_returns"] != 12)
        | (audit["reconstructed_rv"] != 0)
    ]

    if not invalid_zero_hours.empty:
        print()
        print("[Zero RV Audit] FAIL")
        print(
            invalid_zero_hours.to_string(index=False)
        )

        raise RuntimeError(
            "At least one zero-RV hour is not explained by "
            "12 exact zero five-minute returns."
        )

    # Show the underlying observations for the first zero hour.
    first_zero_hour = zero_hours.iloc[0]

    first_hour_rows = returns_5m.loc[
        (
            returns_5m["return_end_utc"]
            > first_zero_hour["hour_start_utc"]
        )
        & (
            returns_5m["return_end_utc"]
            <= first_zero_hour["hour_end_utc"]
        )
    ]

    print()
    print("Five-minute observations in the first zero-RV hour:")
    print(
        first_hour_rows.to_string(index=False)
    )

    print()
    print("[Zero RV Audit] PASS")
    print(
        "[Success] Every zero realized-variance hour is "
        "explained by 12 exact zero five-minute returns."
    )


if __name__ == "__main__":
    main()