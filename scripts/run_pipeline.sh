#!/usr/bin/env bash
set -euo pipefail

# Execute from any working directory.
cd "$(dirname "$0")/.."

# Reruns the full pipeline, including public-data download and model fitting.
# Some stages (particularly state-space estimation) can take significant time.

echo "==> Download the original monthly Binance archives"
python src/download_full_data.py

echo "==> Extract archived CSV files"
python src/extract_full_data.py

echo "==> Combine the 5-minute candlesticks"
python src/build_merged_kline_dataset.py

echo "==> Build 5-minute log returns"
python src/build_5m_returns.py

echo "==> Construct hourly realized variance"
python src/build_hourly_realized_variance.py

echo "==> Flag suspicious zero-activity intervals"
python src/build_hourly_quality_flags.py

echo "==> Build the hourly prediction target"
python src/build_hourly_target.py

echo "==> Align forecast origins and future targets"
python src/build_forecast_base_table.py

echo "==> HAR feature stage 1"
python src/build_har_features_stage1.py

echo "==> HAR feature stage 2"
python src/build_har_features_stage2.py

echo "==> HAR feature stage 3"
python src/build_har_features_stage3.py

echo "==> Downside realized semivariance"
python src/build_downside_semivariance_feature.py

echo "==> Trading activity"
python src/build_trade_count_feature.py

echo "==> Aggressive buying proxy"
python src/build_aggressive_buy_feature.py

echo "==> Assign train / validation / test"
python src/build_time_splits.py

echo "==> Apply training-fitted downside transformation"
python src/apply_downside_transform.py

echo "==> Persistence baseline"
python src/evaluate_naive_baseline.py

echo "==> AR benchmark and diagnostics"
python src/run_complete_ar_analysis.py

echo "==> Fixed HAR-X analysis"
python src/run_fixed_harx_analysis.py

echo "==> TVP-HAR-X analysis"
python src/run_tvp_harx_analysis.py

echo "==> Locked final 2025 test"
python src/run_final_test_evaluation.py

echo "==> Tables and figures"
python src/create_final_report_outputs.py

