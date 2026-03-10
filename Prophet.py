import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from prophet import Prophet


def load_prophet_df(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    cols = {c.lower(): c for c in df.columns}

    date_col = cols.get("date", None)
    if date_col is None:
        raise KeyError(f"No date column. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    price_col = None
    for c in df.columns:
        if "price" in c.lower():
            price_col = c
            break
    if price_col is None:
        price_col = [c for c in df.columns if c != date_col][0]

    out = df[[date_col, price_col]].copy()
    out.columns = ["ds", "y"]
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna()
    return out


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    data = load_prophet_df("oil_prices_2426.csv")

    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    m.fit(train)

    future_test = test[["ds"]].copy()
    fc_test = m.predict(future_test)
    test_rmse = rmse(test["y"].values, fc_test["yhat"].values)
    print("Prophet Test RMSE:", round(test_rmse, 4))

    future = m.make_future_dataframe(periods=730, freq="D")
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    plt.title("Prophet Forecast (24 months) with 95% CI")
    plt.xlabel("Date")
    plt.ylabel("price (dollars)")
    plt.show()

    fig2 = m.plot_components(forecast)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(test["ds"], test["y"], label="Actual")
    plt.plot(fc_test["ds"], fc_test["yhat"], label="Predicted")
    plt.fill_between(fc_test["ds"], fc_test["yhat_lower"], fc_test["yhat_upper"], alpha=0.3)
    plt.title("Prophet Test Set Fit with 95% CI")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
