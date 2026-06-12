"""Statistical baselines: persistence, seasonal-naive, SARIMA.

Persistence and seasonal-naive operate directly on the window tensors
(missingness-aware by construction: they use the observation mask to find the
last observed value / the most recent same-hour value). SARIMA is fit per
station on a forward-fill+mean imputed univariate PM2.5 series, with a rolling
30-day context per forecast anchor to keep CPU cost bounded.

All predictions are produced in **scaled** space (consistent with the neural
models); inverse scaling happens once in evaluation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from src.data.dataset import AirQualityWindowDataset, StationArrays

logger = logging.getLogger(__name__)


def _predict_window_persistence(
    values: np.ndarray, mask: np.ndarray, t_cols: list[int], n_horizons: int
) -> np.ndarray:
    """Last observed value of each target variable, repeated for all horizons."""
    out = np.zeros((len(t_cols), n_horizons), dtype=np.float32)
    for ti, col in enumerate(t_cols):
        obs = np.flatnonzero(mask[:, col])
        last = values[obs[-1], col] if len(obs) else 0.0  # 0 = train mean (scaled)
        out[ti, :] = last
    return out


def _predict_window_seasonal_naive(
    values: np.ndarray, mask: np.ndarray, t_cols: list[int], horizons: list[int]
) -> np.ndarray:
    """Most recent observed value at the same hour-of-day as each target.

    For horizon ``h`` the target sits ``h`` hours after the anchor (last input
    row, index L-1). The same-hour candidates inside the window are at input
    indices ``L - 1 + h - 24k`` for k = ceil(h/24), ceil(h/24)+1, ...; the
    first observed one is used. Falls back to persistence, then to 0.
    """
    L = values.shape[0]
    out = np.zeros((len(t_cols), len(horizons)), dtype=np.float32)
    for ti, col in enumerate(t_cols):
        for hi, h in enumerate(horizons):
            pred = None
            k = -(-h // 24)  # ceil
            pos = L - 1 + h - 24 * k
            while pos >= 0:
                if mask[pos, col]:
                    pred = values[pos, col]
                    break
                pos -= 24
            if pred is None:  # persistence fallback
                obs = np.flatnonzero(mask[:, col])
                pred = values[obs[-1], col] if len(obs) else 0.0
            out[ti, hi] = pred
    return out


def predict_statistical(
    ds: AirQualityWindowDataset, cfg: dict[str, Any], method: str
) -> np.ndarray:
    """Run persistence or seasonal-naive over every window of ``ds``.

    Returns predictions of shape (n_windows, T, H), scaled space.
    """
    from src.data.dataset import feature_columns

    feats = feature_columns(cfg)
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    t_cols = [feats.index(t) for t in targets]
    L = ds.input_length

    preds = np.zeros((len(ds), len(targets), len(horizons)), dtype=np.float32)
    for i, (s_i, anchor) in enumerate(ds.index):
        st = ds.stations[s_i]
        sl = slice(anchor - L + 1, anchor + 1)
        values, mask = st.values[sl], st.mask[sl]
        if method == "persistence":
            preds[i] = _predict_window_persistence(values, mask, t_cols, len(horizons))
        elif method == "seasonal_naive":
            preds[i] = _predict_window_seasonal_naive(values, mask, t_cols, horizons)
        else:
            raise ValueError(f"unknown method {method!r}")
    logger.info("%s: predicted %d windows", method, len(ds))
    return preds


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def _ffill_mean_1d(x: np.ndarray) -> np.ndarray:
    """Forward-fill then 0-fill (0 = train mean in scaled space)."""
    out = x.copy()
    idx = np.where(np.isfinite(out), np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[~np.isfinite(out)] = 0.0
    return out


def select_sarima_order(
    series: np.ndarray, cfg: dict[str, Any]
) -> tuple[tuple, tuple, dict[str, float]]:
    """Small AIC grid search on one station's recent training data.

    Kept deliberately cheap (documented): 4 candidate (order, seasonal_order)
    combinations evaluated on the last 30 days of the series.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    candidates = [
        ((1, 0, 0), (1, 0, 1, 24)),
        ((1, 0, 1), (1, 0, 1, 24)),
        ((2, 0, 1), (1, 0, 1, 24)),
        ((1, 0, 1), (0, 1, 1, 24)),
    ]
    tail = series[-720:]
    aics: dict[str, float] = {}
    best, best_aic = None, np.inf
    for order, sorder in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(tail, order=order, seasonal_order=sorder).fit(
                    disp=False, maxiter=cfg["baselines"]["sarima"]["maxiter"]
                )
            aics[f"{order}x{sorder}"] = float(res.aic)
            if res.aic < best_aic:
                best, best_aic = (order, sorder), res.aic
        except Exception as exc:  # noqa: BLE001 - record and move on
            aics[f"{order}x{sorder}"] = float("nan")
            logger.warning("SARIMA order %sx%s failed: %s", order, sorder, exc)
    logger.info("SARIMA order selection AICs: %s -> best %s", aics, best)
    assert best is not None, "all SARIMA candidates failed"
    return best[0], best[1], aics


def predict_sarima(
    ds: AirQualityWindowDataset,
    stations: list[StationArrays],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Per-station SARIMA forecasts for PM2.5 at each test anchor.

    Strategy (documented in the paper): parameters are fit once per station on
    the last ``context_hours`` of the *training* period; at each forecast
    anchor the fitted parameters are applied (no refit) to a rolling
    ``context_hours`` context ending at the anchor via ``results.apply``, and
    ``forecast(max_horizon)`` yields the multi-step predictions. Only the
    PM2.5 row of the output is populated; other targets are NaN.
    """
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    from src.data.dataset import feature_columns

    scfg = cfg["baselines"]["sarima"]
    feats = feature_columns(cfg)
    targets = cfg["dataset"]["target_pollutants"]
    horizons = cfg["dataset"]["horizons"]
    pm_col = feats.index(cfg["dataset"]["primary_target"])
    pm_row = targets.index(cfg["dataset"]["primary_target"])
    ctx = int(scfg["context_hours"])
    max_h = max(horizons)
    train_end = pd.Timestamp(cfg["splits"]["train_end"])

    preds = np.full((len(ds), len(targets), len(horizons)), np.nan, dtype=np.float32)

    # group window indices by station to fit once per station
    by_station: dict[int, list[tuple[int, int]]] = {}
    for w_i, (s_i, anchor) in enumerate(ds.index):
        by_station.setdefault(int(s_i), []).append((w_i, int(anchor)))

    order = tuple(scfg["order"])
    sorder = tuple(scfg["seasonal_order"])
    for s_i, anchors in sorted(by_station.items()):
        st = stations[s_i]
        # scaled PM2.5 with NaNs restored from the mask, then ffill+mean imputed
        series = np.where(st.mask[:, pm_col] > 0, st.values[:, pm_col], np.nan)
        series = _ffill_mean_1d(series)

        train_mask = pd.DatetimeIndex(st.times) <= train_end
        train_series = series[train_mask][-ctx:]
        if len(train_series) < 200:
            logger.warning("SARIMA %s: only %d train hours, skipping",
                           st.station, len(train_series))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = SARIMAX(train_series, order=order, seasonal_order=sorder).fit(
                disp=False, maxiter=scfg["maxiter"]
            )
        for w_i, anchor in anchors:
            context = series[max(0, anchor + 1 - ctx): anchor + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fitted.apply(context, refit=False)
                fc = res.forecast(max_h)
            for hi, h in enumerate(horizons):
                preds[w_i, pm_row, hi] = fc[h - 1]
        logger.info("SARIMA %s: %d anchors forecast", st.station, len(anchors))
    return preds
