"""Task 1 - Natural gas price analysis and forecasting.

The model captures:
1. a long-run linear time trend; and
2. recurring monthly seasonality through month dummy variables.

Run from the repository root with:
    python -m src.task1_gas_forecasting
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Nat_Gas.csv"
OUTPUT_DIR = ROOT / "outputs"

FEATURE_COLUMNS = ["trend_days"] + [f"month_{month}" for month in range(2, 13)]


def load_gas_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the monthly natural-gas price data."""
    data = pd.read_csv(path)
    required = {"Dates", "Prices"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["Dates"] = pd.to_datetime(data["Dates"], format="%m/%d/%y")
    data["Prices"] = pd.to_numeric(data["Prices"], errors="raise")
    data = data.sort_values("Dates").reset_index(drop=True)
    return data


def build_features(dates, base_date: pd.Timestamp) -> pd.DataFrame:
    """Create a linear time-trend feature and monthly seasonal dummies."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))

    features = pd.DataFrame(index=range(len(dates)))
    features["trend_days"] = (dates - base_date).days

    month_dummies = pd.get_dummies(
        pd.Series(dates.month, index=features.index),
        prefix="month",
        dtype=float,
    )

    # January is the reference month. Keep a stable feature schema for prediction.
    for column in [f"month_{month}" for month in range(2, 13)]:
        features[column] = month_dummies[column] if column in month_dummies else 0.0

    return features[FEATURE_COLUMNS].astype(float)


def fit_gas_price_model(data: pd.DataFrame):
    """Fit the trend + seasonality linear regression model."""
    base_date = data["Dates"].min()
    X = build_features(data["Dates"], base_date)
    y = data["Prices"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    return model, base_date


def predict_price(model, base_date: pd.Timestamp, date) -> float:
    """Predict the natural-gas price for one date."""
    X = build_features([pd.Timestamp(date)], base_date)
    return float(model.predict(X)[0])


def forecast_next_month_ends(
    model,
    base_date: pd.Timestamp,
    last_observed_date: pd.Timestamp,
    periods: int = 12,
) -> pd.DataFrame:
    """Forecast prices for the next `periods` month-end dates."""
    first_forecast = last_observed_date + pd.offsets.MonthEnd(1)
    dates = pd.date_range(first_forecast, periods=periods, freq="ME")
    X_future = build_features(dates, base_date)

    return pd.DataFrame(
        {
            "Dates": dates,
            "Predicted_Prices": model.predict(X_future),
        }
    )


def plot_history_and_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plot observed gas prices and the 12-month forecast."""
    plt.figure(figsize=(11, 5))
    plt.plot(history["Dates"], history["Prices"], marker="o", label="Historical")
    plt.plot(
        forecast["Dates"],
        forecast["Predicted_Prices"],
        marker="o",
        linestyle="--",
        label="Forecast",
    )
    plt.xlabel("Date")
    plt.ylabel("Natural gas price")
    plt.title("Natural Gas Prices - Historical Data and Seasonal Trend Forecast")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)

    plt.show()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_gas_data()
    model, base_date = fit_gas_price_model(data)
    forecast = forecast_next_month_ends(
        model,
        base_date,
        data["Dates"].max(),
        periods=12,
    )

    forecast.to_csv(OUTPUT_DIR / "gas_price_forecast.csv", index=False)

    print("12-month natural-gas price forecast:")
    print(forecast.to_string(index=False))

    plot_history_and_forecast(
        data,
        forecast,
        OUTPUT_DIR / "gas_price_forecast.png",
    )


if __name__ == "__main__":
    main()
