import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def train_test_split_ts(series, test_size=0.2):
    split = int(len(series) * (1 - test_size))
    return series[:split], series[split:]


def create_lag_features(series, lags=6):
    df = pd.DataFrame(series)
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df.iloc[:, 0].shift(lag)
    df.dropna(inplace=True)
    return df


def evaluate_forecast(true, pred):
    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    mape = float(np.mean(np.abs((true - pred) / np.maximum(true, 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def infer_seasonal_period(inferred_freq, n_obs: int) -> int:
    """Pick a sensible seasonal length from pandas inferred frequency."""
    if n_obs < 8:
        return 2
    if inferred_freq is None:
        return int(max(2, min(12, n_obs // 6)))

    freq_token = str(inferred_freq).upper().split("-")[0].split("@")[0]

    if freq_token.startswith("W"):
        m = 52
    elif freq_token.startswith("M") or freq_token == "BM" or freq_token == "MS":
        m = 12
    elif freq_token.startswith("Q"):
        m = 4
    elif freq_token.startswith("D") or freq_token == "B":
        m = 7
    elif freq_token.startswith("H"):
        m = 24
    else:
        m = 12

    cap = max(2, n_obs // 4)
    return int(max(2, min(m, cap)))


def _residual_std(true, pred):
    e = np.asarray(true, dtype=float).ravel() - np.asarray(pred, dtype=float).ravel()
    return float(np.std(e)) if len(e) > 1 else float(np.abs(e[0])) if len(e) == 1 else 0.0


def walk_forward_eval(series, split_eval_fn, n_splits=3):
    """
    Average hold-out metrics over several chronological cut points.
    split_eval_fn(train, test) -> (metrics dict, pred Series aligned to test).
    """
    n = len(series)
    test_len = max(4, min(28, n // 10))
    starts = [int(n * f) for f in np.linspace(0.58, 0.84, n_splits)]
    metric_list = []
    resid_list = []
    for start in starts:
        if start + test_len > n:
            continue
        train = series.iloc[:start]
        test = series.iloc[start : start + test_len]
        if len(train) < max(18, test_len * 2):
            continue
        try:
            m, pred = split_eval_fn(train, test)
        except Exception:
            continue
        metric_list.append(m)
        resid_list.append(_residual_std(test, pred))

    if not metric_list:
        train, test = train_test_split_ts(series)
        m, pred = split_eval_fn(train, test)
        return m, _residual_std(test, pred), "single_holdout"

    agg = {k: float(np.mean([mm[k] for mm in metric_list])) for k in metric_list[0]}
    rs = float(np.mean(resid_list)) if resid_list else 0.0
    return agg, rs, "walk_forward"


class BaseForecaster(ABC):
    def __init__(self, series, steps=12, seasonal_period=12, walk_forward_folds=1):
        self.series = pd.Series(series, dtype=float).sort_index()
        self.steps = int(steps)
        self.seasonal_period = int(max(2, seasonal_period))
        self.walk_forward_folds = int(max(1, walk_forward_folds))

    @abstractmethod
    def fit_forecast(self):
        """Return (metrics dict, future forecast Series length steps, extras dict)."""
        pass


class NaiveForecaster(BaseForecaster):
    def _split_eval(self, train, test):
        last = float(train.iloc[-1])
        pred = pd.Series(np.full(len(test), last), index=test.index)
        return evaluate_forecast(test, pred), pred

    def _future(self):
        v = float(self.series.iloc[-1])
        return pd.Series(np.full(self.steps, v))

    def fit_forecast(self):
        wf = self.walk_forward_folds
        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, self._split_eval, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"
        return metrics, self._future(), {"residual_std": rs, "validation": mode}


class SeasonalNaiveForecaster(BaseForecaster):
    def _split_eval(self, train, test):
        m = min(self.seasonal_period, len(train))
        full = pd.concat([train, test]).sort_index()
        preds = []
        for t in test.index:
            pos = full.index.get_loc(t)
            j = pos - m
            preds.append(float(full.iloc[j]) if j >= 0 else float(train.iloc[-1]))
        pred = pd.Series(preds, index=test.index)
        return evaluate_forecast(test, pred), pred

    def _future(self):
        m = min(self.seasonal_period, len(self.series))
        hist = [float(x) for x in self.series.values]
        out = []
        for _ in range(self.steps):
            j = len(hist) - m
            nxt = float(hist[j]) if j >= 0 else float(hist[-1])
            out.append(nxt)
            hist.append(nxt)
        return pd.Series(out)

    def fit_forecast(self):
        wf = self.walk_forward_folds
        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, self._split_eval, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"
        return metrics, self._future(), {"residual_std": rs, "validation": mode}


class MovingAverageForecaster(BaseForecaster):
    """Flat forecast equal to the mean of the last `window` observations."""

    def _window(self, train):
        w = max(3, min(self.seasonal_period, max(3, len(train) // 6)))
        return min(w, len(train))

    def _split_eval(self, train, test):
        w = self._window(train)
        mu = float(train.iloc[-w:].mean())
        pred = pd.Series(np.full(len(test), mu), index=test.index)
        return evaluate_forecast(test, pred), pred

    def _future(self):
        w = self._window(self.series)
        mu = float(self.series.iloc[-w:].mean())
        return pd.Series(np.full(self.steps, mu))

    def fit_forecast(self):
        wf = self.walk_forward_folds
        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, self._split_eval, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"
        return metrics, self._future(), {"residual_std": rs, "validation": mode}


class ARIMAForecaster(BaseForecaster):
    def _split_eval(self, train, test):
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.forecast(len(test))
        pred = pd.Series(np.asarray(pred), index=test.index)
        return evaluate_forecast(test, pred), pred

    def fit_forecast(self):
        wf = self.walk_forward_folds

        def ev(tr, te):
            return self._split_eval(tr, te)

        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, ev, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"

        fit = ARIMA(self.series, order=(1, 1, 1)).fit()
        fc = pd.Series(np.asarray(fit.forecast(self.steps)))
        return metrics, fc, {"residual_std": rs, "validation": mode}


class SARIMAForecaster(BaseForecaster):
    def _seasonal_order(self, train_len: int):
        m = self.seasonal_period
        if train_len < 2 * m + 8:
            return (0, 0, 0, 0)
        return (1, 1, 1, m)

    def _split_eval(self, train, test):
        seasonal_order = self._seasonal_order(len(train))
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.forecast(len(test))
        pred = pd.Series(np.asarray(pred), index=test.index)
        return evaluate_forecast(test, pred), pred

    def fit_forecast(self):
        wf = self.walk_forward_folds

        def ev(tr, te):
            return self._split_eval(tr, te)

        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, ev, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"

        seasonal_order = self._seasonal_order(len(self.series))
        fit = SARIMAX(
            self.series,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fc = pd.Series(np.asarray(fit.forecast(self.steps)))
        return metrics, fc, {"residual_std": rs, "validation": mode}


class ETSForecaster(BaseForecaster):
    def _fit_hw(self, y, seasonal_periods):
        if seasonal_periods is None:
            return ExponentialSmoothing(y, trend="add", seasonal=None).fit()
        return ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit()

    def _effective_m(self, train_len: int):
        m = self.seasonal_period
        if train_len < 2 * m + 4:
            return None
        return m

    def _split_eval(self, train, test):
        m = self._effective_m(len(train))
        fit = self._fit_hw(train, m)
        pred = fit.forecast(len(test))
        pred = pd.Series(np.asarray(pred), index=test.index)
        return evaluate_forecast(test, pred), pred

    def fit_forecast(self):
        wf = self.walk_forward_folds

        def ev(tr, te):
            return self._split_eval(tr, te)

        if wf > 1 and len(self.series) >= 48:
            metrics, rs, mode = walk_forward_eval(self.series, ev, n_splits=wf)
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"

        m = self._effective_m(len(self.series))
        fit = self._fit_hw(self.series, m)
        fc = pd.Series(np.asarray(fit.forecast(self.steps)))
        return metrics, fc, {"residual_std": rs, "validation": mode}


class XGBoostForecaster(BaseForecaster):
    LAGS = 6

    def _split_eval(self, train, test):
        comb = pd.concat([train, test]).sort_index()
        df = create_lag_features(comb, lags=self.LAGS)
        if len(df) < 12:
            raise ValueError("insufficient rows for lags")
        te_start = test.index.min()
        train_df = df[df.index < te_start]
        test_df = df[(df.index >= te_start) & (df.index <= test.index.max())]
        if len(train_df) < 10 or len(test_df) < 2:
            raise ValueError("insufficient lag rows")

        X_tr = train_df.drop(columns=train_df.columns[0])
        y_tr = train_df.iloc[:, 0]
        X_te = test_df.drop(columns=test_df.columns[0])
        y_te = test_df.iloc[:, 0]

        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_trs, y_tr)
        pred = model.predict(X_tes)
        pred = pd.Series(pred, index=test_df.index).reindex(test.index)
        pred = pred.ffill().fillna(float(train.iloc[-1]))
        return evaluate_forecast(test, pred), pred

    def _future_recursive(self, scaler, model):
        last_values = list(self.series.iloc[-self.LAGS :].astype(float))
        future_preds = []
        for _ in range(self.steps):
            X_input = scaler.transform([last_values[-self.LAGS :]])
            nxt = float(model.predict(X_input)[0])
            future_preds.append(nxt)
            last_values.append(nxt)
        return pd.Series(future_preds)

    def fit_forecast(self):
        wf = self.walk_forward_folds

        def ev(tr, te):
            return self._split_eval(tr, te)

        if wf > 1 and len(self.series) >= 60:
            try:
                metrics, rs, mode = walk_forward_eval(self.series, ev, n_splits=wf)
            except Exception:
                train, test = train_test_split_ts(self.series)
                metrics, pred = self._split_eval(train, test)
                rs = _residual_std(test, pred)
                mode = "single_holdout"
        else:
            train, test = train_test_split_ts(self.series)
            metrics, pred = self._split_eval(train, test)
            rs = _residual_std(test, pred)
            mode = "single_holdout"

        df = create_lag_features(self.series, lags=self.LAGS)
        X = df.drop(columns=df.columns[0])
        y = df.iloc[:, 0]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            n_jobs=1,
        )
        model.fit(Xs, y)
        fc = self._future_recursive(scaler, model)
        return metrics, fc, {"residual_std": rs, "validation": mode}


def get_model_registry():
    return {
        "Naive": NaiveForecaster,
        "Seasonal Naive": SeasonalNaiveForecaster,
        "Moving Average": MovingAverageForecaster,
        "ARIMA": ARIMAForecaster,
        "SARIMA": SARIMAForecaster,
        "ETS": ETSForecaster,
        "XGBoost": XGBoostForecaster,
    }
