"""
backbone.py
Sliding-window forecasting backbone using multi-output Ridge regression.

Single Ridge with all horizon steps as outputs (much faster than 96 separate models).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class SlidingWindowRidge:
    """
    Multi-step forecasting via a single multi-output Ridge regression.
    Predicts all H horizon steps simultaneously.
    """

    def __init__(self, lookback: int = 96, horizon: int = 96, alpha: float = 1.0):
        self.lookback = lookback
        self.horizon = horizon
        self.alpha = alpha
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = Ridge(alpha=alpha, fit_intercept=True)

    def _make_windows(self, data: np.ndarray):
        T, D = data.shape
        X_list, y_list = [], []
        for t in range(self.lookback, T - self.horizon + 1):
            X_list.append(data[t - self.lookback: t, :].flatten())
            y_list.append(data[t: t + self.horizon, 0])   # target = col 0
        if not X_list:
            return np.empty((0, self.lookback * D)), np.empty((0, self.horizon))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def fit(self, train: np.ndarray):
        """train : (T_train, D)  target is column 0"""
        X, Y = self._make_windows(train)
        if len(X) == 0:
            raise ValueError("Not enough data for the given lookback/horizon.")
        X_s = self.scaler_x.fit_transform(X)
        Y_s = self.scaler_y.fit_transform(Y)
        self.model.fit(X_s, Y_s)

    def evaluate(self, test: np.ndarray):
        """test : (T_test, D) -> dict with mse, mae, n_samples"""
        X, Y = self._make_windows(test)
        if len(X) == 0:
            return {"mse": float("nan"), "mae": float("nan"), "n_samples": 0}
        X_s = self.scaler_x.transform(X)
        Y_hat_s = self.model.predict(X_s)
        Y_hat = self.scaler_y.inverse_transform(Y_hat_s)
        mse = float(np.mean((Y_hat - Y) ** 2))
        mae = float(np.mean(np.abs(Y_hat - Y)))
        return {"mse": round(mse, 6), "mae": round(mae, 6), "n_samples": len(X)}
