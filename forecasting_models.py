import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# =====================================================
# Utility Functions
# =====================================================

def train_test_split_ts(series, test_size=0.2):
    split = int(len(series) * (1 - test_size))
    return series[:split], series[split:]


def create_lag_features(series, lags=5):
    df = pd.DataFrame(series)
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df.iloc[:, 0].shift(lag)
    df.dropna(inplace=True)
    return df


def evaluate_forecast(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / np.maximum(true, 1e-8))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# =====================================================
# Base Forecaster
# =====================================================

class BaseForecaster(ABC):
    def __init__(self, series, steps=6):
        self.series = series
        self.steps = steps

    @abstractmethod
    def fit_forecast(self):
        pass


# =====================================================
# ARIMA
# =====================================================

class ARIMAForecaster(BaseForecaster):
    def fit_forecast(self):
        train, test = train_test_split_ts(self.series)
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()

        forecast_test = fit.forecast(len(test))
        metrics = evaluate_forecast(test, forecast_test)

        future_forecast = fit.forecast(self.steps)

        return metrics, future_forecast


# =====================================================
# SARIMA
# =====================================================

class SARIMAForecaster(BaseForecaster):
    def fit_forecast(self):
        train, test = train_test_split_ts(self.series)
        model = SARIMAX(train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)

        forecast_test = fit.forecast(len(test))
        metrics = evaluate_forecast(test, forecast_test)

        future_forecast = fit.forecast(self.steps)

        return metrics, future_forecast


# =====================================================
# ETS
# =====================================================

class ETSForecaster(BaseForecaster):
    def fit_forecast(self):
        train, test = train_test_split_ts(self.series)
        model = ExponentialSmoothing(train,
                                     trend="add",
                                     seasonal="add",
                                     seasonal_periods=12)
        fit = model.fit()

        forecast_test = fit.forecast(len(test))
        metrics = evaluate_forecast(test, forecast_test)

        future_forecast = fit.forecast(self.steps)

        return metrics, future_forecast


# =====================================================
# XGBoost with Lag Features
# =====================================================

class XGBoostForecaster(BaseForecaster):
    def fit_forecast(self):
        df = create_lag_features(self.series, lags=6)

        train, test = train_test_split_ts(df)

        X_train = train.drop(columns=train.columns[0])
        y_train = train.iloc[:, 0]

        X_test = test.drop(columns=test.columns[0])
        y_test = test.iloc[:, 0]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        pred_test = model.predict(X_test)
        metrics = evaluate_forecast(y_test, pred_test)

        # Recursive future forecasting
        last_values = list(self.series[-6:])
        future_preds = []

        for _ in range(self.steps):
            X_input = scaler.transform([last_values[-6:]])
            next_pred = model.predict(X_input)[0]
            future_preds.append(next_pred)
            last_values.append(next_pred)

        return metrics, pd.Series(future_preds)
