# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Optional
import numpy as np
from gluonts.time_feature import get_seasonality


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: np.ndarray,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """
    diff = np.mean(np.abs(target - forecast), axis=1)
    mase = diff / seasonal_error
    # if seasonal_error is 0, set mase to 0
    mase = mase.filled(0)  
    return np.mean(mase)

def calculate_seasonal_error(
    past_data: np.ndarray,
    freq: Optional[str] = None,
):
    r"""
    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency. See [HA21]_ for more details.
    """
    seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1
        
    y_t = past_data[:, :-forecast_freq]
    y_tm = past_data[:, forecast_freq:]

    mean_diff = np.mean(np.abs(y_t - y_tm), axis=1)
    mean_diff = np.expand_dims(mean_diff, axis=1)

    return mean_diff



def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray, eps: float = 1e-8) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    denom = (np.abs(target) + np.abs(forecast)) / 2.0
    return np.mean(np.abs(target - forecast) / (denom + eps))

def dtw_distance(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Dynamic Time Warping (DTW) distance with L1 local cost.

    If target/forecast are multivariate, compute DTW per variable and average.
    """
    def _dtw_1d(x: np.ndarray, y: np.ndarray) -> float:
        n, m = len(x), len(y)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.abs(x[i - 1] - y[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[n, m])

    x = target
    y = forecast
    if isinstance(x, np.ma.MaskedArray):
        x = np.ma.filled(x, 0.0)
    if isinstance(y, np.ma.MaskedArray):
        y = np.ma.filled(y, 0.0)
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        return _dtw_1d(x, y)
    if x.ndim == 2:
        # (T, D)
        return float(np.mean([_dtw_1d(x[:, d], y[:, d]) for d in range(x.shape[1])]))
    if x.ndim == 3:
        # (B, T, D) -> average over batch
        return float(np.mean([dtw_distance(x[b], y[b]) for b in range(x.shape[0])]))
    raise ValueError("dtw_distance expects 1D, 2D, or 3D arrays")

def extreme_mae(target: np.ndarray, forecast: np.ndarray, q: float = 0.9) -> float:
    r"""
    Mean absolute error on extreme targets (|Y| in the top q-quantile).
    """
    if isinstance(target, np.ma.MaskedArray):
        target = np.ma.filled(target, 0.0)
    if isinstance(forecast, np.ma.MaskedArray):
        forecast = np.ma.filled(forecast, 0.0)
    target = np.asarray(target)
    forecast = np.asarray(forecast)
    thresh = np.quantile(np.abs(target), q)
    mask = np.abs(target) >= thresh
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(target[mask] - forecast[mask])))

def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * ((Y <= \hat{Y}) - q)|)
    """
    return 2 * np.abs((forecast - target) * ((target <= forecast) - q))

def scaled_quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float, seasonal_error) -> np.ndarray:
    return quantile_loss(target, forecast, q) / seasonal_error

def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y < \hat{Y})
    """
    return np.mean(target < forecast)
