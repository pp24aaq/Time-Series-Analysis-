import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def load_ts(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    cols = {c.lower(): c for c in df.columns}

    if "date" not in cols:
        raise KeyError(f"No date column. Columns: {list(df.columns)}")

    date_col = cols["date"]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    price_col = None
    for c in df.columns:
        if "price" in c.lower():
            price_col = c
            break
    if price_col is None:
        price_col = df.columns[0]

    ts = pd.to_numeric(df[price_col], errors="coerce").dropna()
    ts = ts[~ts.index.duplicated(keep="first")].sort_index()
    ts = ts.asfreq("D")
    ts = ts.interpolate("time")
    ts.name = price_col
    return ts


def adf_p(series):
    return adfuller(series.dropna(), autolag="AIC")[1]


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def try_fit_aic(series, order):
    try:
        fit = ARIMA(series, order=order).fit()
        return fit.aic
    except:
        return np.inf


def grid_stage1(series):
    candidates = [0, 2, 4, 6, 8]
    best_aic, best_order = np.inf, None
    for p in candidates:
        for d in range(0, 3):
            for q in candidates:
                aic = try_fit_aic(series, (p, d, q))
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order = aic, (p, d, q)
    return best_order, best_aic


def grid_stage2(series, seed):
    p0, d0, q0 = seed
    p_range = range(max(0, p0 - 2), min(8, p0 + 2) + 1)
    q_range = range(max(0, q0 - 2), min(8, q0 + 2) + 1)
    d_range = range(0, 3)

    best_aic, best_order = np.inf, None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                aic = try_fit_aic(series, (p, d, q))
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order = aic, (p, d, q)
    return best_order, best_aic


def main():
    ts = load_ts("oil_prices_2426.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(ts)
    plt.title("Oil Price Time Series")
    plt.xlabel("Date")
    plt.ylabel(ts.name)
    plt.tight_layout()
    plt.show()

    p0 = adf_p(ts)
    d_used = 0
    if p0 > 0.05:
        for d in [1, 2]:
            if adf_p(ts.diff(d).dropna()) <= 0.05:
                d_used = d
                break
        if d_used == 0:
            d_used = 2

    ts_stat = ts.diff(d_used).dropna() if d_used > 0 else ts.copy()

    plot_acf(ts_stat, lags=40)
    plt.title("ACF (stationary)")
    plt.tight_layout()
    plt.show()

    plot_pacf(ts_stat, lags=40, method="ywm")
    plt.title("PACF (stationary)")
    plt.tight_layout()
    plt.show()

    seed_order, seed_aic = grid_stage1(ts)
    best_order, best_aic = grid_stage2(ts, seed_order)

    print("Stage1 best:", seed_order, "AIC:", round(seed_aic, 3))
    print("Final best:", best_order, "AIC:", round(best_aic, 3))

    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    fit = ARIMA(train, order=best_order).fit()
    pred = fit.forecast(steps=len(test))

    print("Test RMSE:", round(rmse(test.values, pred.values), 4))

    resid = fit.resid.dropna()

    plot_acf(resid, lags=40)
    plt.title("Residual ACF")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=30)
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.show()

    steps = 730
    final_fit = ARIMA(ts, order=best_order).fit()
    fc = final_fit.get_forecast(steps=steps)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int()

    plt.figure(figsize=(11, 4))
    plt.plot(ts, label="Observed")
    plt.plot(mean_fc, label="Forecast")
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.3)
    plt.title("ARIMA Forecast (24 months ~ 730 days) with 95% CI")
    plt.xlabel("Date")
    plt.ylabel(ts.name)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
