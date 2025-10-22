import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

# 📏 Evaluation Metrics
def evaluate_forecast(true, pred):
    rmse = mean_squared_error(true, pred, squared=False)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape
