import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Utility — ensure DatetimeIndex
def ensure_datetime_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=[df.columns[0]])  # Drop rows where target is NA
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    df = df.sort_index()
    return df

# Simple Moving Average
class SimpleMA:
    def __init__(self, df, target_col, steps=6, window=3):
        self.df = ensure_datetime_index(df)
        self.target = target_col
        self.steps = steps
        self.window = window

    def forecast(self):
        y = self.df[self.target].dropna()
        last_date = y.index[-1]
        freq = pd.infer_freq(y.index) or "W"
        future_idx = pd.date_range(last_date, periods=self.steps + 1, freq=freq)[1:]
        forecast_value = y.rolling(self.window).mean().iloc[-1]
        forecast = pd.Series([forecast_value] * self.steps, index=future_idx)
        return forecast

# ARIMA Model
class ARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        model = ARIMA(y, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=self.steps)
        freq = pd.infer_freq(y.index) or "W"
        forecast.index = pd.date_range(y.index[-1], periods=self.steps + 1, freq=freq)[1:]
        return forecast

# SARIMA Model
class SARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=self.steps)
        freq = pd.infer_freq(y.index) or "W"
        forecast.index = pd.date_range(y.index[-1], periods=self.steps + 1, freq=freq)[1:]
        return forecast

# ETS Model
class ETSForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(self.steps)
        freq = pd.infer_freq(y.index) or "W"
        forecast.index = pd.date_range(y.index[-1], periods=self.steps + 1, freq=freq)[1:]
        return forecast

# XGBoost Model
class XGBoostForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        df = y.reset_index()
        df["time"] = np.arange(len(df))
        X, y_train = df[["time"]], df[self.target]
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y_train)

        future_time = np.arange(len(df), len(df) + self.steps).reshape(-1, 1)
        pred = model.predict(future_time)

        freq = pd.infer_freq(y.index) or "W"
        forecast_idx = pd.date_range(y.index[-1], periods=self.steps + 1, freq=freq)[1:]
        forecast = pd.Series(pred, index=forecast_idx)
        return forecast

# Evaluation metrics
def evaluate_forecast(true, pred):
    true, pred = np.array(true), np.array(pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / np.maximum(true, 1e-8))) * 100
    return rmse, mae, mape
