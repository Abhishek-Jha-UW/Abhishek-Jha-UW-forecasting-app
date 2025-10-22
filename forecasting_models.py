import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_forecast_index(last_date, steps, freq):
    """Safely generate forecast index from inferred frequency."""
    try:
        forecast_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=steps,
            freq=freq
        )
    except Exception:
        # Default to weekly if inference fails
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="7D")
    return forecast_index


class SimpleMA:
    def __init__(self, df, target_column, window=4, steps=6, freq="7D"):
        self.df = df.copy()
        self.target_column = target_column
        self.window = window
        self.steps = steps
        self.freq = freq

    def forecast(self):
        sma_series = self.df[self.target_column].rolling(window=self.window).mean()
        last_sma = sma_series.dropna().iloc[-1]
        forecast_values = [last_sma] * self.steps
        forecast_index = get_forecast_index(self.df.index[-1], self.steps, self.freq)
        return pd.Series(forecast_values, index=forecast_index, name="Forecast")


class ARIMAForecaster:
    def __init__(self, df, target_column, order=(1, 1, 1), steps=6, freq="7D"):
        self.series = df[target_column]
        self.order = order
        self.steps = steps
        self.freq = freq

    def forecast(self):
        model = ARIMA(self.series, order=self.order)
        fitted = model.fit()
        forecast_obj = fitted.get_forecast(steps=self.steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        forecast_index = get_forecast_index(self.series.index[-1], self.steps, self.freq)
        forecast.index = forecast_index
        conf_int.index = forecast_index
        return forecast, conf_int


class SARIMAForecaster:
    def __init__(self, df, target_column, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), steps=6, freq="7D"):
        self.series = df[target_column]
        self.order = order
        self.seasonal_order = seasonal_order
        self.steps = steps
        self.freq = freq

    def forecast(self):
        model = SARIMAX(self.series, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
        fitted = model.fit(disp=False)
        forecast_obj = fitted.get_forecast(steps=self.steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        forecast_index = get_forecast_index(self.series.index[-1], self.steps, self.freq)
        forecast.index = forecast_index
        conf_int.index = forecast_index
        return forecast, conf_int


class ETSForecaster:
    def __init__(self, df, target_column, trend='add', seasonal=None, steps=6, freq="7D"):
        self.series = df[target_column]
        self.trend = trend
        self.seasonal = seasonal
        self.steps = steps
        self.freq = freq

    def forecast(self):
        model = ExponentialSmoothing(self.series, trend=self.trend, seasonal=self.seasonal)
        fitted = model.fit()
        forecast = fitted.forecast(self.steps)
        forecast_index = get_forecast_index(self.series.index[-1], self.steps, self.freq)
        forecast.index = forecast_index
        return forecast, None


class XGBoostForecaster:
    def __init__(self, df, target_column, lags=6, steps=6, freq="7D"):
        self.df = df.copy()
        self.target_column = target_column
        self.lags = lags
        self.steps = steps
        self.freq = freq

    def create_features(self):
        for i in range(1, self.lags + 1):
            self.df[f'lag_{i}'] = self.df[self.target_column].shift(i)
        self.df.dropna(inplace=True)

    def forecast(self):
        self.create_features()
        X = self.df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y = self.df[self.target_column]
        model = XGBRegressor(verbosity=0)
        model.fit(X, y)
        last_row = X.iloc[-1].values.reshape(1, -1)
        preds = []
        for _ in range(self.steps):
            pred = model.predict(last_row)[0]
            preds.append(pred)
            last_row = np.roll(last_row, -1)
            last_row[0, -1] = pred
        forecast_index = get_forecast_index(self.df.index[-1], self.steps, self.freq)
        forecast = pd.Series(preds, index=forecast_index, name="Forecast")
        return forecast, None


def evaluate_forecast(true, pred):
    true, pred = np.array(true), np.array(pred)
    min_len = min(len(true), len(pred))
    true, pred = true[:min_len], pred[:min_len]
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape
