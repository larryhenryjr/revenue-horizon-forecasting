# Revenue Horizon Forecasting

A reproducible 90-day revenue-forecasting workflow designed to translate daily performance history into planning signals for capacity, infrastructure, and resource allocation.

## Business question

What patterns are present in the first two years of daily revenue, and how can the next 90 days be forecast to support operational planning?

## Approach

- Validate temporal ordering, completeness, and numeric integrity.
- Inspect trend, seasonality, autocorrelation, and stationarity.
- Fit an ARIMA specification using a chronological training window.
- Evaluate the latest period as an out-of-time holdout.
- Export point forecasts, confidence intervals, and MAE/RMSE/MAPE metrics.

## Portfolio artifacts

- [Rendered analysis notebook](notebook/analysis.html)
- [Portfolio report](reports/revenue-horizon-forecasting-report.pdf)

The public repository includes the rendered notebook and report while intentionally excluding the working `.ipynb` file, restricted source data, local paths, and academic grading material.
## Repository structure

```text
├── data/README.md
├── src/forecast.py
├── tests/test_forecast.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
python src/forecast.py path/to/revenue.csv --steps 90 --order 1,1,1
pytest tests/ -v --tb=short
```

The input CSV must contain a numeric `Revenue` column and may contain either `Date` or `Day` for its time index.

## Decision value

Forecast intervals make uncertainty visible rather than presenting a single number as certainty. The holdout evaluation also separates in-sample fit from forward-looking performance, which is essential when the output will inform resource decisions.

## Portfolio

Case study: [larry-henry-jr.net](https://larry-henry-jr.net/#project/revenue-forecasting)
