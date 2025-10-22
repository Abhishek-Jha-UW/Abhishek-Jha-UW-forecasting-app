import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 📊 Simple Moving Average
class SimpleMA:
    def __init__(self, df, target_column, window=4):
        self.df = df.copy()
        self.target_column = target_column
        self.window = window

    def apply(self):
        self.df['SMA'] = self.df[self.target_column].rolling(window=self.window).mean()
        return self.df

# 🔮 ARIMA Forecaster
class ARIMAForecaster:
    def __init__(self, df, target_column, order=(1,1,1), steps=6):
        self.series = df[target_column]
        self.order = order
        self.steps = steps

    def forecast(self):
        model = ARIMA(self.series, order=self.order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=self.steps)
        forecast.index = pd.date_range(start=self.series.index[-1], periods=self.steps+1, freq='W')[1:]
        return forecast

# 🌦️ SARIMA Forecaster
class SARIMAForecaster:
    def __init__(self, df, target_column, order=(1,1,1), seasonal_order=(1,1,1,12), steps=6):
        self.series = df[target_column]
        self.order = order
        self.seasonal_order = seasonal_order
        self.steps = steps

    def forecast(self):
        model = SARIMAX(self.series, order=self.order, seasonal_order=self.seasonal_order)
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=self.steps)
        forecast.index = pd.date_range(start=self.series.index[-1], periods=self.steps+1, freq='W')[1:]
        return forecast

# 📉 ETS Forecaster
class ETSForecaster:
    def __init__(self, df, target_column, steps=6):
        self.series = df[target_column]
        self.steps = steps

    def forecast(self):
        model = ExponentialSmoothing(self.series, trend='add', seasonal=None)
        fitted = model.fit()
        forecast = fitted.forecast(self.steps)
        forecast.index = pd.date_range(start=self.series.index[-1], periods=self.steps+1, freq='W')[1:]
        return forecast

# ⚙️ XGBoost Forecaster
class XGBoostForecaster:
    def __init__(self, df, target_column, lags=6, steps=6):
        self.df = df.copy()
        self.target_column = target_column
        self.lags = lags
        self.steps = steps

    def create_features(self):
        for i in range(1, self.lags + 1):
            self.df[f'lag_{i}'] = self.df[self.target_column].shift(i)
        self.df.dropna(inplace=True)

    def forecast(self):
        self.create_features()
        X = self.df[[f'lag_{i}' for i in range(1, self.lags + 1)]]
        y = self.df[self.target_column]
        model = XGBRegressor()
        model.fit(X, y)
        last_row = X.iloc[-1].values.reshape(1, -1)
        preds = []
        for _ in range(self.steps):
            pred = model.predict(last_row)[0]
            preds.append(pred)
            last_row = np.roll(last_row, -1)
            last_row[0, -1] = pred
        forecast_index = pd.date_range(start=self.df.index[-1], periods=self.steps+1, freq='W')[1:]
        return pd.Series(preds, index=forecast_index)

# 📏 Evaluation Metrics (Safe across environments)
def evaluate_forecast(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    if len(true) != len(pred):
        min_len = min(len(true), len(pred))
        true = true[:min_len]
        pred = pred[:min_len]
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape
