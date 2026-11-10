"""Time-series forecasting utilities for daily revenue planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def load_revenue_series(path: Path) -> pd.Series:
    """Load a dated revenue series from a CSV file."""
    frame = pd.read_csv(path)
    normalized = {column.lower(): column for column in frame.columns}
    revenue_column = normalized.get("revenue")
    if revenue_column is None:
        raise ValueError("The CSV must include a Revenue column.")

    date_column = normalized.get("date") or normalized.get("day")
    values = pd.to_numeric(frame[revenue_column], errors="coerce")
    if values.isna().any():
        raise ValueError("Revenue must contain only numeric, nonmissing values.")

    if date_column is not None:
        index = pd.to_datetime(frame[date_column], errors="coerce")
        if index.isna().any():
            raise ValueError("Date values must be valid timestamps.")
        series = pd.Series(values.to_numpy(), index=index, name="revenue")
        series = series.sort_index()
    else:
        series = pd.Series(values.to_numpy(), name="revenue")

    validate_series(series)
    return series


def validate_series(series: pd.Series, minimum_observations: int = 60) -> None:
    """Validate the minimum conditions needed for forecasting."""
    if len(series) < minimum_observations:
        raise ValueError(f"At least {minimum_observations} observations are required.")
    if series.isna().any():
        raise ValueError("The revenue series cannot contain missing values.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("The revenue series must contain finite values.")
    if isinstance(series.index, pd.DatetimeIndex) and series.index.has_duplicates:
        raise ValueError("The revenue series cannot contain duplicate dates.")


def fit_arima(
    series: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
) -> ARIMAResults:
    """Fit an ARIMA model after validating the input series."""
    validate_series(series)
    return ARIMA(series.astype(float), order=order).fit()


def generate_forecast(model: ARIMAResults, steps: int = 90) -> pd.DataFrame:
    """Generate point forecasts and 95% confidence intervals."""
    if steps <= 0:
        raise ValueError("Forecast steps must be greater than zero.")
    prediction = model.get_forecast(steps=steps)
    interval = prediction.conf_int(alpha=0.05)
    result = pd.DataFrame(
        {
            "forecast": prediction.predicted_mean,
            "lower_95": interval.iloc[:, 0],
            "upper_95": interval.iloc[:, 1],
        }
    )
    return result.reset_index(names="period")


def evaluate_holdout(
    series: pd.Series,
    holdout: int = 90,
    order: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, float]:
    """Evaluate an ARIMA specification on the most recent holdout period."""
    validate_series(series, minimum_observations=holdout + 30)
    train = series.iloc[:-holdout]
    actual = series.iloc[-holdout:].to_numpy(dtype=float)
    model = fit_arima(train, order=order)
    predicted = np.asarray(model.forecast(steps=holdout), dtype=float)
    errors = actual - predicted
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero]))) if nonzero.any() else np.nan
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": mape,
    }


def parse_order(value: str) -> tuple[int, int, int]:
    """Parse a comma-separated ARIMA order."""
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 3 or any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("Order must contain three nonnegative integers.")
    return parts


def main() -> None:
    """Run validation, holdout evaluation, and a 90-day forecast."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="CSV containing Revenue and optional Date/Day columns.")
    parser.add_argument("--order", type=parse_order, default=(1, 1, 1))
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("outputs/forecast.csv"))
    args = parser.parse_args()

    series = load_revenue_series(args.data)
    metrics: dict[str, Any] = evaluate_holdout(series, holdout=min(args.steps, len(series) // 4), order=args.order)
    model = fit_arima(series, order=args.order)
    forecast = generate_forecast(model, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
