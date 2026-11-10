"""Tests for the revenue forecasting workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast import fit_arima, generate_forecast, load_revenue_series, validate_series


def test_load_revenue_series_orders_dates(tmp_path: Path) -> None:
    """CSV loading should parse and order dated revenue observations."""
    path = tmp_path / "revenue.csv"
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=60, freq="D")[::-1],
            "Revenue": np.linspace(100.0, 160.0, 60),
        }
    )
    frame.to_csv(path, index=False)
    series = load_revenue_series(path)
    assert series.index.is_monotonic_increasing
    assert len(series) == 60


def test_validate_series_rejects_short_input() -> None:
    """A short history should fail before model fitting."""
    with pytest.raises(ValueError, match="At least"):
        validate_series(pd.Series([1.0, 2.0, 3.0]))


def test_generate_forecast_returns_requested_horizon() -> None:
    """The forecast output should include point and interval estimates."""
    rng = np.random.default_rng(42)
    series = pd.Series(1000 + np.cumsum(rng.normal(0, 2, 90)))
    model = fit_arima(series)
    forecast = generate_forecast(model, steps=7)
    assert len(forecast) == 7
    assert {"period", "forecast", "lower_95", "upper_95"} == set(forecast.columns)
