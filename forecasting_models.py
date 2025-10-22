import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# --- Utility Functions ---

def ensure_datetime_index(df, target_col):
    """Ensure dataframe index is DatetimeIndex and sorted."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(subset=[target_col])
    df = df.sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    return df

def safe_infer_freq(index):
    """Try to infer frequency safely, fallback to 'W' or 'D'."""
    freq = pd.infer_freq(index)
    if freq is None:
        deltas = np.diff(index.values).astype("timedelta64[D]").astype(int)
        median_gap = np.median(deltas) if len(deltas) > 0 else 7
        freq = "D" if median_gap <= 1 else "W"
    return freq

# --- Forecasting Models ---

class SimpleMA:
    def __init__(self, df, target_col, steps=6, window=3):
        self.df = ensure_datetime_index(df, target_col)
        self.target = target_col
        self.steps = steps
        self.window = window

    def forecast(self):
        y = self.df[self.target].dropna()
        freq = safe_infer_freq(y.index)
        offset = pd.tseries.frequencies.to_offset(freq)
        forecast_index = [y.index[-1] + (i + 1) * offset for i in range(self.steps)]
        forecast_value = y.rolling(self.window).mean().iloc[-1]
        return pd.Series([forecast_value] * self.steps, index=pd.to_datetime(forecast_index))

class ARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df, target_col)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        model = ARIMA(y, order=(1, 1, 1))
        fit = model.fit()
        freq = safe_infer_freq(y.index)
        offset = pd.tseries.frequencies.to_offset(freq)
        forecast = fit.forecast(steps=self.steps)
        forecast_index = [y.index[-1] + (i + 1) * offset for i in range(self.steps)]
        forecast.index = pd.to_datetime(forecast_index)
        return forecast

class SARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df, target_col)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)
        freq = safe_infer_freq(y.index)
        offset = pd.tseries.frequencies.to_offset(freq)
        forecast = fit.forecast(steps=self.steps)
        forecast_index = [y.index[-1] + (i + 1) * offset for i in range(self.steps)]
        forecast.index = pd.to_datetime(forecast_index)
        return forecast

class ETSForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df, target_col)
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target]
        freq = safe_infer_freq(y.index)
        offset = pd.tseries.frequencies.to_offset(freq)
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(self.steps)
        forecast_index = [y.index[-1] + (i + 1) * offset for i in range(self.steps)]
        forecast.index = pd.to_datetime(forecast_index)
        return forecast

class XGBoostForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = ensure_datetime_index(df, target_col)
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
        freq = safe_infer_freq(self.df.index)
        offset = pd.tseries.frequencies.to_offset(freq)
        forecast_index = [self.df.index[-1] + (i + 1) * offset for i in range(self.steps)]
        return pd.Series(pred, index=pd.to_datetime(forecast_index))

# --- Evaluation ---

def evaluate_forecast(true, pred):
    true, pred = np.array(true), np.array(pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / np.maximum(true, 1e-8))) * 100
    return rmse, mae, mape
