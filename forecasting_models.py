import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# --- Simple Moving Average ---
class SimpleMA:
    def __init__(self, df, target_col, steps=6, window=3):
        self.df = df.copy()
        self.target = target_col
        self.steps = steps
        self.window = window

    def forecast(self):
        y = self.df[self.target].dropna()
        last_sma = y.rolling(window=self.window).mean().iloc[-1]
        forecast_values = [last_sma] * self.steps
        return pd.Series(forecast_values)  # No datetime index

# --- ARIMA ---
class ARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = df.copy()
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target].dropna()
        model = ARIMA(y, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=self.steps)
        return forecast  # Uses default index from statsmodels

# --- SARIMA ---
class SARIMAForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = df.copy()
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target].dropna()
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=self.steps)
        return forecast  # No datetime index assigned

# --- ETS ---
class ETSForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = df.copy()
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target].dropna()
        model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(self.steps)
        return forecast  # No datetime index assigned

# --- XGBoost ---
class XGBoostForecaster:
    def __init__(self, df, target_col, steps=6):
        self.df = df.copy()
        self.target = target_col
        self.steps = steps

    def forecast(self):
        y = self.df[self.target].dropna()
        df = y.reset_index(drop=True)
        df = df.to_frame()
        df["time"] = np.arange(len(df))
        X = df[["time"]]
        y_train = df[self.target]
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y_train)
        future_time = np.arange(len(df), len(df) + self.steps).reshape(-1, 1)
        pred = model.predict(future_time)
        return pd.Series(pred)  # No datetime index

# --- Evaluation ---
def evaluate_forecast(true, pred):
    true, pred = np.array(true), np.array(pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / np.maximum(true, 1e-8))) * 100
    return rmse, mae, mape
