"""Broad imputation-techniques benchmark (reconstruction quality).

This module benchmarks a *representative method per taxonomy family* (~35 concrete
imputers drawn from the survey list in the project brief) by the same protocol the
paper already uses for its imputability axis: hide a sample of **observed** test-period
cells, reconstruct them, and score the error at the hidden cells. It is the engine
behind ``notebooks/imputation_benchmark.ipynb``.

Design (mirrors the existing code, does not reinvent it):

* Test-period ``(values, mask)`` station slices come from
  :func:`src.data.dataset.build_station_arrays` + :func:`split_ranges`.
* The cell-hiding + scoring contract generalises
  :func:`src.evaluate._impute_skill`: each method is a callable
  ``fn(x_in, m_in, ctx) -> recon`` over one station's ``(N, V)`` standardized array,
  where ``x_in`` is missing-zeroed and ``m_in`` is the *reduced* observation mask
  (observed minus the artificially-hidden holdout). The third ``ctx`` argument carries
  the per-slice ``times`` / spatial panel that calendar- and space-aware methods need
  (``_impute_skill`` only passes two args because its single method, SAITS, needs
  neither).
* Learned methods are **fit on the training period only** (``times <= train_end``),
  exactly like :func:`src.data.impute.impute_full_series`; the registry stores a
  closure over the fitted object. Forward-fill (:func:`src.data.impute.ffill_mean_impute`)
  is the anchor baseline; ``imputability = 1 - RMSE_method / RMSE_ffill``.
* Errors are accumulated per (method, pattern, variable). Physical PM2.5 RMSE/MAE
  (µg/m³) = standardized error × ``scalers[var][1]``; overall metrics are reported in
  standardized units so they are comparable across the three networks.

Optional third-party imputers (fancyimpute, pypots, minisom, scikit-fuzzy) are
imported lazily and guarded: a missing wheel demotes the method to ``skipped`` in the
coverage map rather than crashing the run.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.data.dataset import build_station_arrays, feature_columns, split_ranges
from src.data.impute import _calendar_features, ffill_mean_impute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-slice context + method record
# ---------------------------------------------------------------------------

@dataclass
class SliceCtx:
    """Side information for one station's test slice, passed to every method."""

    times: np.ndarray            # (N,) datetime64
    var_names: list[str]         # length V
    station_id: int
    # Spatial panel: for each variable index, an (N, S) array of the OTHER stations'
    # standardized observed values aligned to this slice's times (NaN where missing).
    spatial: "SpatialPanel | None" = None


ImputerFn = Callable[[np.ndarray, np.ndarray, SliceCtx], np.ndarray]


@dataclass
class Method:
    name: str
    family: str
    speed: str               # 'fast' | 'slow' | 'deep'
    fn: ImputerFn


# ---------------------------------------------------------------------------
# Simple per-slice imputers (no training)
# ---------------------------------------------------------------------------

def _impute_mean(x_in, m_in, ctx):
    """Global/train mean = 0 in standardized space (missing cells already 0)."""
    return x_in.copy()


def _impute_ffill(x_in, m_in, ctx):
    return ffill_mean_impute(x_in, m_in)


def _impute_bfill(x_in, m_in, ctx):
    flip = ffill_mean_impute(x_in[::-1], m_in[::-1])
    return flip[::-1].copy()


def _impute_last_next(x_in, m_in, ctx):
    """Last-and-next mean: average of forward- and backward-fill."""
    f = ffill_mean_impute(x_in, m_in)
    b = _impute_bfill(x_in, m_in, ctx)
    return ((f + b) / 2.0).astype(np.float32)


def _group_mean_fill(x_in, m_in, key):
    """Fill missing cells with the per-(group key, column) mean of observed cells."""
    out = x_in.copy()
    obs = m_in > 0
    V = x_in.shape[1]
    keys = np.asarray(key)
    for v in range(V):
        col_obs = obs[:, v]
        if not col_obs.any():
            continue
        df = pd.DataFrame({"k": keys[col_obs], "x": x_in[col_obs, v]})
        means = df.groupby("k")["x"].mean()
        miss = ~col_obs
        if miss.any():
            mapped = pd.Series(keys[miss]).map(means).to_numpy()
            mapped = np.where(np.isfinite(mapped), mapped, 0.0)
            out[miss, v] = mapped
    return out.astype(np.float32)


def _impute_hour_mean(x_in, m_in, ctx):
    hod = pd.DatetimeIndex(ctx.times).hour.to_numpy()
    return _group_mean_fill(x_in, m_in, hod)


def _impute_day_mean(x_in, m_in, ctx):
    day = pd.DatetimeIndex(ctx.times).floor("D").asi8
    return _group_mean_fill(x_in, m_in, day)


def _interp_fill(x_in, m_in, kind):
    out = x_in.copy().astype(np.float64)
    obs = m_in > 0
    N, V = x_in.shape
    idx = np.arange(N)
    for v in range(V):
        o = obs[:, v]
        if o.sum() < 2:
            out[:, v] = np.where(o, x_in[:, v], 0.0)
            continue
        s = pd.Series(np.where(o, x_in[:, v], np.nan))
        if kind == "nearest":
            s = s.interpolate(method="nearest", limit_direction="both")
        elif kind == "linear":
            s = s.interpolate(method="linear", limit_direction="both")
        elif kind == "cubic":
            # cubic spline needs >=4 points; fall back to linear otherwise
            try:
                s = s.interpolate(method="spline", order=3, limit_direction="both")
            except Exception:
                s = s.interpolate(method="linear", limit_direction="both")
        out[:, v] = s.to_numpy()
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _impute_linear(x_in, m_in, ctx):
    return _interp_fill(x_in, m_in, "linear")


def _impute_cubic(x_in, m_in, ctx):
    return _interp_fill(x_in, m_in, "cubic")


def _impute_nearest(x_in, m_in, ctx):
    return _interp_fill(x_in, m_in, "nearest")


def _impute_spatial(x_in, m_in, ctx):
    """Cross-station fill: hidden cell <- weighted mean of other stations at same time.

    Stand-in for IDW / optimal interpolation. Without station coordinates we weight
    every co-observing station equally (the other stations keep their real observed
    values; only this station's cells were hidden). Falls back to column mean (0).
    """
    if ctx.spatial is None:
        return _impute_linear(x_in, m_in, ctx)
    out = x_in.copy()
    obs = m_in > 0
    V = x_in.shape[1]
    for v in range(V):
        miss = ~obs[:, v]
        if not miss.any():
            continue
        donor = ctx.spatial.mean_other(ctx.station_id, v)  # (N,) NaN where none
        fill = donor[miss]
        out[miss, v] = np.where(np.isfinite(fill), fill, 0.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial panel (built once per dataset from the test-period slices)
# ---------------------------------------------------------------------------

class SpatialPanel:
    """Time-aligned standardized observed values of every station, per variable.

    ``mean_other(station_id, v)`` returns, for each row of the *reference* time grid,
    the mean of all OTHER stations' observed values of variable ``v`` (NaN if none).
    Slices share one reference grid (the union of test times), so a station's row
    selection is found by matching its times into the grid.
    """

    def __init__(self, grid_times: np.ndarray, panel: dict[int, np.ndarray],
                 row_of: dict[int, np.ndarray]):
        self.grid_times = grid_times
        self._panel = panel          # v -> (G, S) standardized obs, NaN missing
        self._row_of = row_of        # station_id -> (N,) row indices into the grid
        self._station_col: dict[int, int] = {}

    def set_station_columns(self, station_ids: list[int]):
        self._station_col = {sid: j for j, sid in enumerate(sorted(station_ids))}

    def mean_other(self, station_id: int, v: int) -> np.ndarray:
        full = self._panel[v]                      # (G, S)
        col = self._station_col[station_id]
        masked = full.copy()
        masked[:, col] = np.nan                    # exclude self
        with np.errstate(invalid="ignore"):
            grid_mean = np.nanmean(masked, axis=1)  # (G,)
        return grid_mean[self._row_of[station_id]]


def _build_spatial_panel(slices: list[dict]) -> SpatialPanel:
    grid = np.unique(np.concatenate([s["times"] for s in slices]))
    pos = {t: i for i, t in enumerate(grid.tolist())}
    V = slices[0]["vals"].shape[1]
    S = len(slices)
    panel = {v: np.full((len(grid), S), np.nan, dtype=np.float64) for v in range(V)}
    row_of = {}
    for j, s in enumerate(slices):
        rows = np.array([pos[t] for t in s["times"].tolist()])
        row_of[s["station_id"]] = rows
        vals, mask = s["vals"], s["mask"]
        for v in range(V):
            ov = mask[:, v] > 0
            panel[v][rows[ov], j] = vals[ov, v]
    sp = SpatialPanel(grid, panel, row_of)
    sp.set_station_columns([s["station_id"] for s in slices])
    return sp


# ---------------------------------------------------------------------------
# Fitted row-wise imputers (sklearn): fit on train, transform test slices
# ---------------------------------------------------------------------------

def _train_matrix(train_stations, cfg, scalers, subsample, seed):
    """Stacked (rows, V+4) train matrix with NaNs + calendar features, subsampled."""
    feats = feature_columns(cfg)
    train_end = pd.Timestamp(cfg["splits"]["train_end"]).to_datetime64()
    blocks = []
    for st in train_stations:
        sel = st.times <= train_end
        if sel.sum() == 0:
            continue
        vals = np.where(st.mask[sel] > 0, st.values[sel], np.nan).astype(np.float64)
        blocks.append(np.hstack([vals, _calendar_features(st.times[sel])]))
    X = np.vstack(blocks)
    rng = np.random.default_rng(seed)
    if subsample and len(X) > subsample:
        X = X[rng.choice(len(X), size=subsample, replace=False)]
    return X, len(feats)


def _make_sklearn_imputer(estimator_kind, cfg, train_stations, scalers, seed,
                          subsample, **kw):
    """Fit a sklearn imputer on train rows; return a per-slice transform closure."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer

    X, V = _train_matrix(train_stations, cfg, scalers, subsample, seed)

    if estimator_kind == "knn":
        imp = KNNImputer(n_neighbors=kw.get("k", 5), weights=kw.get("weights", "uniform"))
    elif estimator_kind == "mice":
        imp = IterativeImputer(max_iter=kw.get("max_iter", 10), random_state=seed,
                               sample_posterior=kw.get("posterior", False),
                               keep_empty_features=True)
    else:  # custom estimator inside IterativeImputer (RF / tree / SVR)
        est = kw["estimator"]
        imp = IterativeImputer(estimator=est, max_iter=kw.get("max_iter", 5),
                               random_state=seed, keep_empty_features=True)
    imp.fit(X)

    def _fn(x_in, m_in, ctx):
        vals = np.where(m_in > 0, x_in, np.nan).astype(np.float64)
        mat = np.hstack([vals, _calendar_features(ctx.times)])
        out = imp.transform(mat)[:, :V]
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    return _fn


# ---------------------------------------------------------------------------
# Transductive matrix completion (per-slice; no train needed)
# ---------------------------------------------------------------------------

def _ppca_impute(x_in, m_in, ctx, n_components=4, n_iter=30):
    """Probabilistic-PCA EM completion of one (N, V) slice (standardized)."""
    X = np.where(m_in > 0, x_in, np.nan).astype(np.float64)
    N, V = X.shape
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    Xf = np.where(np.isfinite(X), X, col_mean)
    q = min(n_components, V - 1) if V > 1 else 1
    for _ in range(n_iter):
        mu = Xf.mean(axis=0)
        Xc = Xf - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        recon = (U[:, :q] * S[:q]) @ Vt[:q] + mu
        miss = ~(m_in > 0)
        Xf = np.where(miss, recon, Xf)
    return np.nan_to_num(Xf, nan=0.0).astype(np.float32)


def _gaussian_em_impute(x_in, m_in, ctx, n_iter=20, ridge=1e-3, bootstrap=False,
                        seed=0):
    """Joint multivariate-normal EM imputation of one slice (Amelia-style if bootstrap)."""
    X = np.where(m_in > 0, x_in, np.nan).astype(np.float64)
    N, V = X.shape
    rng = np.random.default_rng(seed)
    src = X
    if bootstrap:
        src = X[rng.integers(0, N, size=N)]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(src, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xf = np.where(np.isfinite(X), X, mu)
    cov = np.cov(Xf.T) + ridge * np.eye(V)
    for _ in range(n_iter):
        mu = Xf.mean(axis=0)
        for i in range(N):
            miss = ~(m_in[i] > 0)
            if not miss.any() or miss.all():
                continue
            obs = ~miss
            Soo = cov[np.ix_(obs, obs)] + ridge * np.eye(obs.sum())
            Smo = cov[np.ix_(miss, obs)]
            Xf[i, miss] = mu[miss] + Smo @ np.linalg.solve(Soo, Xf[i, obs] - mu[obs])
        cov = np.cov(Xf.T) + ridge * np.eye(V)
    return np.nan_to_num(Xf, nan=0.0).astype(np.float32)


def _fancyimpute_method(name):
    """Return a per-slice closure around a fancyimpute completer, or None if absent."""
    try:
        import fancyimpute  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("fancyimpute unavailable (%s): skipping %s", exc, name)
        return None

    def _fn(x_in, m_in, ctx):
        from fancyimpute import IterativeSVD, MatrixFactorization, SoftImpute
        X = np.where(m_in > 0, x_in, np.nan).astype(np.float64)
        if (~np.isfinite(X)).all():
            return x_in.copy()
        try:
            if name == "softimpute":
                out = SoftImpute(verbose=False).fit_transform(X)
            elif name == "iterativesvd":
                out = IterativeSVD(rank=min(4, X.shape[1] - 1), verbose=False).fit_transform(X)
            else:  # matrix factorization (PMF)
                out = MatrixFactorization(rank=min(4, X.shape[1] - 1), verbose=False).fit_transform(X)
        except Exception as exc:
            logger.warning("%s failed on a slice (%s) -> mean fill", name, exc)
            out = np.nan_to_num(X, nan=0.0)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    return _fn


# ---------------------------------------------------------------------------
# Time-series per-column imputers (statsmodels): slow group
# ---------------------------------------------------------------------------

def _statespace_impute(x_in, m_in, ctx, kind):
    """Per-column state-space smoothing (ARIMA or local-level Kalman)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = x_in.copy().astype(np.float64)
    obs = m_in > 0
    N, V = x_in.shape
    for v in range(V):
        o = obs[:, v]
        if o.sum() < 10:
            out[:, v] = _interp_fill(x_in[:, [v]], m_in[:, [v]], "linear")[:, 0]
            continue
        y = np.where(o, x_in[:, v], np.nan)
        try:
            order = (2, 0, 1) if kind == "arima" else (0, 0, 0)
            trend = None if kind == "arima" else "n"
            model = SARIMAX(y, order=order, trend=trend,
                            measurement_error=(kind == "kalman"),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.filter(model.start_params) if kind == "kalman" else model.fit(disp=False, maxiter=20)
            pred = np.asarray(res.predict())
            lin = _interp_fill(x_in[:, [v]], m_in[:, [v]], "linear")[:, 0]
            # standardized data: |z|>8 (or non-finite) is a divergence -> use interp
            bad = ~np.isfinite(pred) | (np.abs(pred) > 8)
            pred = np.where(bad, lin, pred)
            out[:, v] = np.where(o, x_in[:, v], pred)
        except Exception:
            out[:, v] = _interp_fill(x_in[:, [v]], m_in[:, [v]], "linear")[:, 0]
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _ssa_impute(x_in, m_in, ctx, window=24, rank=3, n_iter=10):
    """Iterative Singular Spectrum Analysis gap filling, per column."""
    out = _interp_fill(x_in, m_in, "linear").astype(np.float64)
    obs = m_in > 0
    N, V = x_in.shape
    L = min(window, max(2, N // 2))
    K = N - L + 1
    if K < 2:
        return out.astype(np.float32)
    for v in range(V):
        if obs[:, v].sum() < L:
            continue
        series = out[:, v].copy()
        miss = ~obs[:, v]
        for _ in range(n_iter):
            traj = np.column_stack([series[i:i + L] for i in range(K)])
            U, S, Vt = np.linalg.svd(traj, full_matrices=False)
            r = min(rank, len(S))
            rec = (U[:, :r] * S[:r]) @ Vt[:r]
            # diagonal averaging back to a series
            recon = np.zeros(N)
            cnt = np.zeros(N)
            for i in range(K):
                recon[i:i + L] += rec[:, i]
                cnt[i:i + L] += 1
            recon /= np.maximum(cnt, 1)
            series[miss] = recon[miss]
        out[:, v] = series
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Clustering imputers
# ---------------------------------------------------------------------------

def _kmeans_impute(x_in, m_in, ctx, k=8, n_iter=10):
    """k-means style fill: assign rows to nearest centroid, fill with centroid value."""
    from sklearn.cluster import KMeans
    X = np.where(m_in > 0, x_in, 0.0).astype(np.float64)
    km = KMeans(n_clusters=min(k, max(2, len(X) // 50)), n_init=3, random_state=0)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    out = x_in.copy()
    miss = ~(m_in > 0)
    rows, cols = np.where(miss)
    out[rows, cols] = centers[labels[rows], cols]
    return out.astype(np.float32)


def _som_method():
    try:
        from minisom import MiniSom  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("minisom unavailable (%s): skipping SOM", exc)
        return None

    def _fn(x_in, m_in, ctx):
        from minisom import MiniSom
        X = np.where(m_in > 0, x_in, 0.0).astype(np.float64)
        V = X.shape[1]
        g = 6
        som = MiniSom(g, g, V, sigma=1.0, learning_rate=0.5, random_seed=0)
        som.train_random(X, 200)
        W = som.get_weights().reshape(-1, V)
        out = x_in.copy()
        for i in np.where((~(m_in > 0)).any(axis=1))[0]:
            o = m_in[i] > 0
            d = ((W[:, o] - x_in[i, o]) ** 2).sum(axis=1)
            bmu = W[int(np.argmin(d))]
            out[i, ~o] = bmu[~o]
        return out.astype(np.float32)

    return _fn


def _fuzzy_cmeans_method():
    try:
        import skfuzzy  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("scikit-fuzzy unavailable (%s): skipping fuzzy c-means", exc)
        return None

    def _fn(x_in, m_in, ctx):
        import skfuzzy as fuzz
        X = np.where(m_in > 0, x_in, 0.0).astype(np.float64)
        c = min(6, max(2, len(X) // 100))
        try:
            cntr, u, *_ = fuzz.cluster.cmeans(X.T, c, 2.0, error=5e-3, maxiter=50, seed=0)
        except Exception:
            return _impute_mean(x_in, m_in, ctx)
        out = x_in.copy()
        miss = ~(m_in > 0)
        soft = (u.T @ cntr)  # (N, V) membership-weighted centroid
        out[miss] = soft[miss]
        return out.astype(np.float32)

    return _fn


# ---------------------------------------------------------------------------
# Deep imputers: pypots adapters + a minimal torch (denoising) autoencoder
# ---------------------------------------------------------------------------

def _segment(arr, seg_len):
    """Non-overlapping segments + a right-aligned tail. Returns (segs, starts)."""
    n = len(arr)
    length = min(seg_len, n)
    starts = list(range(0, n - length + 1, length)) or [0]
    if starts[-1] + length < n:
        starts.append(n - length)
    segs = np.stack([arr[s:s + length] for s in starts])
    return segs, starts, length


def _stitch(segs, starts, length, n, V):
    out = np.empty((n, V), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    for j, s in enumerate(starts):
        rows = ~filled[s:s + length]
        out[s:s + length][rows] = segs[j][rows]
        filled[s:s + length] = True
    return out


def _train_segments(train_stations, cfg, seg_len):
    feats_n = len(feature_columns(cfg))
    train_end = pd.Timestamp(cfg["splits"]["train_end"]).to_datetime64()
    out = []
    stride = seg_len
    for st in train_stations:
        sel = st.times <= train_end
        vals = np.where(st.mask[sel] > 0, st.values[sel], np.nan).astype(np.float32)
        n = len(vals)
        for s in range(0, n - seg_len + 1, stride):
            out.append(vals[s:s + seg_len])
    if not out:
        return np.zeros((1, seg_len, feats_n), np.float32)
    return np.stack(out)


def _make_pypots_method(name, cfg, train_stations, device, seg_len=168, epochs=10):
    """Train a pypots imputer on train segments; return a per-slice closure."""
    try:
        import pypots  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("pypots/torch unavailable (%s): skipping %s", exc, name)
        return None

    n_feat = len(feature_columns(cfg))
    train_X = _train_segments(train_stations, cfg, seg_len)
    common = dict(n_steps=seg_len, n_features=n_feat, epochs=epochs,
                  batch_size=64, device=device)
    try:
        from pypots.imputation import (BRITS, CSDI, GPVAE, MRNN, SAITS, USGAN,
                                       GRUD, Transformer, TimesNet)
        builders = {
            "saits": lambda: SAITS(n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16,
                                   d_ffn=128, dropout=0.1, **common),
            "transformer": lambda: Transformer(n_layers=2, d_model=64, n_heads=4,
                                               d_k=16, d_v=16, d_ffn=128, dropout=0.1,
                                               **common),
            "brits": lambda: BRITS(rnn_hidden_size=64, **common),
            "mrnn": lambda: MRNN(rnn_hidden_size=64, **common),
            "grud": lambda: GRUD(rnn_hidden_size=64, **common),
            "timesnet": lambda: TimesNet(n_layers=2, top_k=3, d_model=32, d_ffn=32,
                                         n_kernels=4, dropout=0.1, **common),
            "gpvae": lambda: GPVAE(latent_size=16, **common),
            "usgan": lambda: USGAN(rnn_hidden_size=64, **common),
            "csdi": lambda: CSDI(n_layers=2, n_heads=4, n_channels=32,
                                 d_time_embedding=32, d_feature_embedding=8,
                                 d_diffusion_embedding=32, **common),
        }
        if name not in builders:
            return None
        model = builders[name]()
        model.fit({"X": train_X})
    except Exception as exc:
        logger.warning("pypots %s failed to build/fit (%s): skipping", name, exc)
        return None

    def _fn(x_in, m_in, ctx):
        import torch  # noqa: F401
        vals = np.where(m_in > 0, x_in, np.nan).astype(np.float32)
        segs, starts, length = _segment(vals, seg_len)
        try:
            res = model.impute({"X": segs})
        except Exception:
            res = model.predict({"X": segs})["imputation"]
        res = np.asarray(res, dtype=np.float32)
        if res.ndim == 4:        # CSDI returns (samples, n, steps, feat)
            res = res.mean(axis=1)
        rec = _stitch(res, starts, length, len(vals), vals.shape[1])
        return np.where(m_in > 0, x_in, rec).astype(np.float32)

    return _fn


def _make_torch_ae(cfg, train_stations, device, denoising, seed=0, epochs=15):
    """A minimal row-wise (denoising) autoencoder trained on mean-filled train rows."""
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        logger.warning("torch unavailable (%s): skipping autoencoder", exc)
        return None

    V = len(feature_columns(cfg))
    train_end = pd.Timestamp(cfg["splits"]["train_end"]).to_datetime64()
    rows = []
    for st in train_stations:
        sel = (st.times <= train_end) & (st.mask.sum(1) == V)  # complete rows only
        rows.append(st.values[sel])
    X = np.vstack(rows) if rows and sum(len(r) for r in rows) else np.zeros((1, V), np.float32)
    if len(X) < 32:
        return None
    torch.manual_seed(seed)
    dev = torch.device(device)
    net = nn.Sequential(nn.Linear(V, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, V)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for lo in range(0, len(Xt), 256):
            b = Xt[perm[lo:lo + 256]]
            inp = b
            if denoising:
                inp = b * (torch.rand_like(b) > 0.3)
            opt.zero_grad()
            loss = ((net(inp) - b) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()

    def _fn(x_in, m_in, ctx):
        with torch.no_grad():
            xb = torch.tensor(np.where(m_in > 0, x_in, 0.0), dtype=torch.float32, device=dev)
            rec = net(xb).cpu().numpy()
        return np.where(m_in > 0, x_in, rec).astype(np.float32)

    return _fn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class MethodSpec:
    """A method that is *not yet built*: ``build()`` fits/constructs it on demand.

    Lazy construction is what makes resume cheap — a method whose result is already
    on disk is never built (no KNN refit, no deep-model training). ``build()`` returns
    the ``ImputerFn`` or ``None`` when an optional dependency is missing.
    """

    name: str
    family: str
    speed: str
    build: Callable[[], "ImputerFn | None"]


def iter_method_specs(cfg, train_stations, scalers, *, device="cpu",
                      include_slow=True, include_deep=True, seed=42,
                      only=None) -> list[MethodSpec]:
    """Ordered list of every method as a lazy spec (nothing is fitted yet).

    ``forward_fill`` is emitted first so the resumable runner always has the
    imputability baseline before any other method. ``only`` (set of names) restricts
    the registry; ``include_slow`` / ``include_deep`` drop those speed tiers.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    icap = 20000
    sk = lambda *a, **k: _make_sklearn_imputer(*a, cfg=cfg, train_stations=train_stations,  # noqa: E731
                                               scalers=scalers, seed=seed, **k)
    # (name, family, speed, thunk). forward_fill first (imputability baseline).
    reg: list[tuple] = [
        ("forward_fill", "mean-based", "fast", lambda: _impute_ffill),
        ("mean", "mean-based", "fast", lambda: _impute_mean),
        ("last_and_next_mean", "mean-based", "fast", lambda: _impute_last_next),
        ("hour_mean", "mean-based", "fast", lambda: _impute_hour_mean),
        ("daily_mean", "mean-based", "fast", lambda: _impute_day_mean),
        ("linear_interp", "interpolation", "fast", lambda: _impute_linear),
        ("cubic_spline", "interpolation", "fast", lambda: _impute_cubic),
        ("nearest_interp", "interpolation", "fast", lambda: _impute_nearest),
        ("spatial_idw", "spatial", "fast", lambda: _impute_spatial),
        ("knn", "proximity", "fast",
         lambda: sk("knn", subsample=icap, k=5)),
        ("nearest_neighbor", "proximity", "fast",
         lambda: sk("knn", subsample=icap, k=1)),
        ("weighted_knn", "proximity", "fast",
         lambda: sk("knn", subsample=icap, k=5, weights="distance")),
        ("mice", "regression/MICE", "fast", lambda: sk("mice", subsample=icap)),
        ("stochastic_regression", "regression/MICE", "fast",
         lambda: sk("mice", subsample=icap, posterior=True)),
        ("em_gaussian", "EM/MLE", "fast",
         lambda: (lambda x, m, c: _gaussian_em_impute(x, m, c, seed=seed))),
        ("emb_bootstrap", "EM/MLE", "fast",
         lambda: (lambda x, m, c: _gaussian_em_impute(x, m, c, bootstrap=True, seed=seed))),
        ("ppca", "matrix/PCA", "fast", lambda: _ppca_impute),
        ("softimpute", "matrix/PCA", "fast", lambda: _fancyimpute_method("softimpute")),
        ("iterative_svd", "matrix/PCA", "fast", lambda: _fancyimpute_method("iterativesvd")),
        ("kmeans", "clustering", "fast", lambda: _kmeans_impute),
        ("decision_tree_cart", "ML-tree", "slow",
         lambda: sk("custom", subsample=8000, max_iter=5,
                    estimator=DecisionTreeRegressor(max_depth=12, random_state=seed))),
        ("missforest_rf", "ML-tree", "slow",
         lambda: sk("custom", subsample=8000, max_iter=4,
                    estimator=RandomForestRegressor(n_estimators=60, n_jobs=-1,
                                                    random_state=seed))),
        ("svr_regression", "ML-SVM", "slow",
         lambda: sk("custom", subsample=4000, max_iter=3, estimator=SVR(C=1.0))),
        ("matrix_factorization", "matrix/PCA", "slow",
         lambda: _fancyimpute_method("matrixfact")),
        ("arima", "state-space", "slow",
         lambda: (lambda x, m, c: _statespace_impute(x, m, c, "arima"))),
        ("kalman_smoother", "state-space", "slow",
         lambda: (lambda x, m, c: _statespace_impute(x, m, c, "kalman"))),
        ("ssa", "state-space", "slow", lambda: _ssa_impute),
        ("som", "clustering", "slow", lambda: _som_method()),
        ("fuzzy_cmeans", "fuzzy", "slow", lambda: _fuzzy_cmeans_method()),
        ("autoencoder", "deep-AE", "deep",
         lambda: _make_torch_ae(cfg, train_stations, device, denoising=False)),
        ("denoising_ae", "deep-AE", "deep",
         lambda: _make_torch_ae(cfg, train_stations, device, denoising=True)),
    ]
    for nm, fam in [("gpvae", "deep-AE"), ("brits", "deep-RNN"), ("mrnn", "deep-RNN"),
                    ("grud", "deep-RNN"), ("saits", "deep-attention"),
                    ("transformer", "deep-attention"), ("timesnet", "deep-attention"),
                    ("csdi", "deep-attention"), ("usgan", "deep-GAN")]:
        reg.append((nm, fam, "deep",
                    (lambda nm=nm: _make_pypots_method(nm, cfg, train_stations, device))))

    specs = []
    for name, family, speed, thunk in reg:
        if only is not None and name not in only:
            continue
        if speed == "slow" and not include_slow:
            continue
        if speed == "deep" and not include_deep:
            continue
        specs.append(MethodSpec(name, family, speed, thunk))
    return specs


def build_methods(cfg, train_stations, scalers, *, device="cpu",
                  include_slow=True, include_deep=True, seed=42,
                  only=None) -> tuple[list[Method], list[dict]]:
    """Eagerly build every method (back-compat wrapper over :func:`iter_method_specs`).

    Returns ``(methods, skipped)``; a spec whose ``build()`` yields ``None`` (missing
    optional dependency) lands in ``skipped``. Prefer :func:`run_resumable` for long
    runs — it builds lazily and checkpoints.
    """
    specs = iter_method_specs(cfg, train_stations, scalers, device=device,
                              include_slow=include_slow, include_deep=include_deep,
                              seed=seed, only=only)
    methods: list[Method] = []
    skipped: list[dict] = []
    for sp in specs:
        try:
            fn = sp.build()
        except Exception as exc:  # pragma: no cover
            logger.warning("%s failed to build (%s)", sp.name, exc)
            fn = None
        if fn is None:
            skipped.append({"method": sp.name, "family": sp.family,
                            "reason": "optional dependency unavailable or build failed"})
        else:
            methods.append(Method(sp.name, sp.family, sp.speed, fn))
    return methods, skipped


# ---------------------------------------------------------------------------
# Cell-hiding patterns
# ---------------------------------------------------------------------------

def _holdout_mcar(mask, rate, rng):
    obs = mask > 0
    return (rng.random(mask.shape) < rate) & obs


def _holdout_outage(mask, rate, rng, block=(6, 48)):
    obs = mask > 0
    art = np.zeros_like(obs)
    n_test = len(mask)
    budget = int(round(obs.sum() * rate))
    dropped = 0
    attempts = 0
    while dropped < budget and attempts < 20000:
        attempts += 1
        start = int(rng.integers(0, n_test))
        length = int(rng.integers(block[0], block[1] + 1))
        end = min(start + length, n_test)
        blk = obs[start:end] & ~art[start:end]
        n = int(blk.sum())
        if n == 0:
            continue
        art[start:end][blk] = True
        dropped += n
    return art


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def _test_slices(cfg, scalers, max_stations=None):
    df = pd.read_parquet(
        os.path.join(cfg["paths"]["processed_dir"], "all_stations.parquet")
    )
    stations = build_station_arrays(df, cfg, scalers)
    test_lo = np.datetime64(split_ranges(cfg)["test"][0])
    slices = []
    for st in stations:
        sel = st.times >= test_lo
        if sel.sum() == 0:
            continue
        slices.append({"station_id": st.station_id, "times": st.times[sel],
                       "vals": st.values[sel], "mask": st.mask[sel]})
    if max_stations:
        slices = slices[:max_stations]
    return df, stations, slices


def run_benchmark(cfg, scalers, methods, *, patterns=("mcar", "outage"),
                  holdout_rate=0.2, seed=42, max_stations=None,
                  max_slice_len=None):
    """Reconstruction benchmark over the test period. Returns (leaderboard, long).

    ``leaderboard``: one row per (method, pattern) with PM2.5 µg/m³ RMSE/MAE, overall
    standardized RMSE/MAE/R², ``imputability`` (vs forward_fill), runtime. ``long``:
    one row per (method, pattern, variable).
    """
    feats = feature_columns(cfg)
    V = len(feats)
    pm25 = feats.index(cfg["dataset"]["primary_target"])
    std = np.array([scalers[c][1] for c in feats])
    dsname = str(cfg.get("dataset_name", "dhaka")).capitalize()

    df, stations, slices = _test_slices(cfg, scalers, max_stations)
    if max_slice_len:
        for s in slices:
            for k in ("times", "vals", "mask"):
                s[k] = s[k][:max_slice_len]
    spatial = _build_spatial_panel(slices)

    # accumulators keyed by (method, pattern) -> per-variable stats
    def fresh():
        return dict(sse=np.zeros(V), sae=np.zeros(V), n=np.zeros(V),
                    st=np.zeros(V), st2=np.zeros(V), t=0.0)
    acc: dict[tuple[str, str], dict] = {}

    pat_id = {"mcar": 1, "outage": 2}
    for pattern in patterns:
        for s in slices:
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, s["station_id"], pat_id.get(pattern, 0)]))
            mask = s["mask"]
            if pattern == "mcar":
                art = _holdout_mcar(mask, holdout_rate, rng)
            else:
                art = _holdout_outage(mask, holdout_rate, rng)
            if not art.any():
                continue
            m_in = ((mask > 0) & ~art).astype(np.float32)
            x_in = (s["vals"] * m_in).astype(np.float32)
            ctx = SliceCtx(times=s["times"], var_names=feats,
                           station_id=s["station_id"], spatial=spatial)
            tgt = s["vals"]
            for mth in methods:
                key = (mth.name, pattern)
                a = acc.setdefault(key, fresh())
                t0 = time.perf_counter()
                try:
                    rec = mth.fn(x_in, m_in, ctx)
                except Exception as exc:
                    logger.warning("%s failed on slice %s/%s: %s",
                                   mth.name, s["station_id"], pattern, exc)
                    a["t"] += time.perf_counter() - t0
                    continue
                a["t"] += time.perf_counter() - t0
                for v in range(V):
                    cell = art[:, v]
                    if not cell.any():
                        continue
                    e = rec[cell, v] - tgt[cell, v]
                    a["sse"][v] += float((e ** 2).sum())
                    a["sae"][v] += float(np.abs(e).sum())
                    a["n"][v] += int(cell.sum())
                    a["st"][v] += float(tgt[cell, v].sum())
                    a["st2"][v] += float((tgt[cell, v] ** 2).sum())

    fam = {m.name: m.family for m in methods}
    long_rows, board_rows = [], []
    ffill_overall = {}
    for (name, pattern), a in acc.items():
        n_tot = a["n"].sum()
        if n_tot == 0:
            continue
        overall_rmse = float(np.sqrt(a["sse"].sum() / n_tot))
        ffill_overall.setdefault(pattern, {})[name] = overall_rmse

    for (name, pattern), a in sorted(acc.items()):
        n_tot = a["n"].sum()
        if n_tot == 0:
            continue
        overall_rmse = float(np.sqrt(a["sse"].sum() / n_tot))
        overall_mae = float(a["sae"].sum() / n_tot)
        tss = (a["st2"] - np.where(a["n"] > 0, a["st"] ** 2 / np.maximum(a["n"], 1), 0)).sum()
        r2 = float(1 - a["sse"].sum() / tss) if tss > 0 else float("nan")
        base = ffill_overall.get(pattern, {}).get("forward_fill")
        imput = float(1 - overall_rmse / base) if base else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            pm_rmse = float(np.sqrt(a["sse"][pm25] / a["n"][pm25]) * std[pm25]) if a["n"][pm25] else float("nan")
            pm_mae = float(a["sae"][pm25] / a["n"][pm25] * std[pm25]) if a["n"][pm25] else float("nan")
        board_rows.append({
            "dataset": dsname, "method": name, "family": fam.get(name, "?"),
            "pattern": pattern, "pm25_rmse_ugm3": round(pm_rmse, 3),
            "pm25_mae_ugm3": round(pm_mae, 3), "overall_std_rmse": round(overall_rmse, 4),
            "overall_std_mae": round(overall_mae, 4), "overall_r2": round(r2, 4),
            "imputability": round(imput, 4), "runtime_s": round(a["t"], 1),
            "n_cells": int(n_tot),
        })
        for v in range(V):
            if a["n"][v] == 0:
                continue
            long_rows.append({
                "dataset": dsname, "method": name, "family": fam.get(name, "?"),
                "pattern": pattern, "variable": feats[v],
                "rmse_ugm3": round(float(np.sqrt(a["sse"][v] / a["n"][v]) * std[v]), 3),
                "mae_ugm3": round(float(a["sae"][v] / a["n"][v] * std[v]), 3),
                "rmse_std": round(float(np.sqrt(a["sse"][v] / a["n"][v])), 4),
                "n_cells": int(a["n"][v]),
            })

    leaderboard = pd.DataFrame(board_rows).sort_values(
        ["pattern", "overall_std_rmse"]).reset_index(drop=True)
    long = pd.DataFrame(long_rows)
    return leaderboard, long


# ---------------------------------------------------------------------------
# Resumable driver: build lazily, checkpoint per method, skip finished work
# ---------------------------------------------------------------------------

def _prepare_inputs(slices, spatial, feats, patterns, holdout_rate, seed):
    """Method-independent per (pattern, slice) inputs: identical hidden cells for all.

    Returns a list of dicts ``{pattern, art, x_in, m_in, ctx, tgt}``. The holdout is a
    pure function of (seed, station, pattern) so every method scores the same cells and
    a resumed run reproduces them exactly.
    """
    pat_id = {"mcar": 1, "outage": 2}
    prepared = []
    for pattern in patterns:
        for s in slices:
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, int(s["station_id"]), pat_id.get(pattern, 0)]))
            mask = s["mask"]
            art = (_holdout_mcar(mask, holdout_rate, rng) if pattern == "mcar"
                   else _holdout_outage(mask, holdout_rate, rng))
            if not art.any():
                continue
            m_in = ((mask > 0) & ~art).astype(np.float32)
            prepared.append({
                "pattern": pattern, "art": art,
                "x_in": (s["vals"] * m_in).astype(np.float32), "m_in": m_in,
                "tgt": s["vals"],
                "ctx": SliceCtx(times=s["times"], var_names=feats,
                                station_id=s["station_id"], spatial=spatial),
            })
    return prepared


def _score_one(fn, prepared, patterns, V):
    """Accumulate a single method's error over all prepared (pattern, slice) inputs."""
    def fresh():
        return dict(sse=np.zeros(V), sae=np.zeros(V), n=np.zeros(V),
                    st=np.zeros(V), st2=np.zeros(V), t=0.0)
    acc = {p: fresh() for p in patterns}
    for it in prepared:
        a = acc[it["pattern"]]
        t0 = time.perf_counter()
        try:
            rec = fn(it["x_in"], it["m_in"], it["ctx"])
        except Exception as exc:
            logger.warning("method failed on a slice: %s", exc)
            a["t"] += time.perf_counter() - t0
            continue
        a["t"] += time.perf_counter() - t0
        art, tgt = it["art"], it["tgt"]
        for v in range(V):
            cell = art[:, v]
            if not cell.any():
                continue
            e = rec[cell, v] - tgt[cell, v]
            a["sse"][v] += float((e ** 2).sum())
            a["sae"][v] += float(np.abs(e).sum())
            a["n"][v] += int(cell.sum())
            a["st"][v] += float(tgt[cell, v].sum())
            a["st2"][v] += float((tgt[cell, v] ** 2).sum())
    return acc


def _rows_from_acc(name, family, dsname, feats, pm25, std, acc, baseline):
    """Turn one method's accumulators into board + per-variable rows."""
    V = len(feats)
    board, long = [], []
    for pattern, a in acc.items():
        n_tot = a["n"].sum()
        if n_tot == 0:
            continue
        overall_rmse = float(np.sqrt(a["sse"].sum() / n_tot))
        overall_mae = float(a["sae"].sum() / n_tot)
        tss = (a["st2"] - np.where(a["n"] > 0, a["st"] ** 2 / np.maximum(a["n"], 1), 0)).sum()
        r2 = float(1 - a["sse"].sum() / tss) if tss > 0 else float("nan")
        base = baseline.get(pattern)
        imput = float(1 - overall_rmse / base) if base else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            pm_rmse = float(np.sqrt(a["sse"][pm25] / a["n"][pm25]) * std[pm25]) if a["n"][pm25] else float("nan")
            pm_mae = float(a["sae"][pm25] / a["n"][pm25] * std[pm25]) if a["n"][pm25] else float("nan")
        board.append({
            "dataset": dsname, "method": name, "family": family, "pattern": pattern,
            "pm25_rmse_ugm3": round(pm_rmse, 3), "pm25_mae_ugm3": round(pm_mae, 3),
            "overall_std_rmse": round(overall_rmse, 4), "overall_std_mae": round(overall_mae, 4),
            "overall_r2": round(r2, 4), "imputability": round(imput, 4),
            "runtime_s": round(a["t"], 1), "n_cells": int(n_tot),
        })
        for v in range(V):
            if a["n"][v] == 0:
                continue
            long.append({
                "dataset": dsname, "method": name, "family": family, "pattern": pattern,
                "variable": feats[v],
                "rmse_ugm3": round(float(np.sqrt(a["sse"][v] / a["n"][v]) * std[v]), 3),
                "mae_ugm3": round(float(a["sae"][v] / a["n"][v] * std[v]), 3),
                "rmse_std": round(float(np.sqrt(a["sse"][v] / a["n"][v])), 4),
                "n_cells": int(a["n"][v]),
            })
    return board, long


def run_resumable(cfg, scalers, results_base, *, train_stations=None,
                  patterns=("mcar", "outage"), holdout_rate=0.2, seed=42,
                  include_slow=True, include_deep=True, only=None, device="cpu",
                  max_stations=None, max_slice_len=None, log=print):
    """Benchmark with per-method checkpointing and resume.

    Each method is built (fitted/trained) only when it is about to run, scored across
    all slices/patterns, and its rows are written to
    ``<results_base>/<dataset>/per_method/<name>.csv`` immediately. Re-invoking with the
    same ``results_base`` skips any method whose file already exists — so a Colab
    disconnect costs at most the one method in flight. Returns the dataset leaderboard.

    Point ``results_base`` at Google Drive to survive disconnects entirely.
    """
    feats = feature_columns(cfg)
    V = len(feats)
    pm25 = feats.index(cfg["dataset"]["primary_target"])
    std = np.array([scalers[c][1] for c in feats])
    dsname = str(cfg.get("dataset_name", "dhaka")).capitalize()

    if train_stations is None:
        df_tr = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"],
                                             "all_stations.parquet"))
        train_stations = build_station_arrays(df_tr, cfg, scalers)

    _, _, slices = _test_slices(cfg, scalers, max_stations)
    if max_slice_len:
        for s in slices:
            for k in ("times", "vals", "mask"):
                s[k] = s[k][:max_slice_len]
    spatial = _build_spatial_panel(slices)
    prepared = _prepare_inputs(slices, spatial, feats, patterns, holdout_rate, seed)

    ds_dir = os.path.join(results_base, dsname.lower())
    pm_dir = os.path.join(ds_dir, "per_method")
    os.makedirs(pm_dir, exist_ok=True)

    specs = iter_method_specs(cfg, train_stations, scalers, device=device,
                              include_slow=include_slow, include_deep=include_deep,
                              seed=seed, only=only)
    log(f"[{dsname}] {len(specs)} methods queued -> {pm_dir}")

    baseline: dict[str, float] = {}
    # if forward_fill already done in a previous session, load its baseline
    ff_csv = os.path.join(pm_dir, "forward_fill.csv")
    if os.path.exists(ff_csv):
        ff = pd.read_csv(ff_csv)
        baseline = dict(zip(ff["pattern"], ff["overall_std_rmse"]))

    for sp in specs:
        out_csv = os.path.join(pm_dir, f"{sp.name}.csv")
        skip_mark = os.path.join(pm_dir, f"{sp.name}.skipped")
        if os.path.exists(out_csv) or os.path.exists(skip_mark):
            log(f"[{dsname}] skip {sp.name} (already saved)")
            if sp.name == "forward_fill" and os.path.exists(out_csv) and not baseline:
                ff = pd.read_csv(out_csv)
                baseline = dict(zip(ff["pattern"], ff["overall_std_rmse"]))
            continue
        t0 = time.perf_counter()
        try:
            fn = sp.build()
        except Exception as exc:
            fn = None
            log(f"[{dsname}] {sp.name} build error: {exc}")
        if fn is None:
            with open(skip_mark, "w") as fh:
                fh.write("optional dependency unavailable or build failed\n")
            log(f"[{dsname}] {sp.name} SKIPPED (missing dep / build failed)")
            continue
        acc = _score_one(fn, prepared, patterns, V)
        if sp.name == "forward_fill":
            for p, a in acc.items():
                nt = a["n"].sum()
                if nt:
                    baseline[p] = round(float(np.sqrt(a["sse"].sum() / nt)), 4)
        board, long = _rows_from_acc(sp.name, sp.family, dsname, feats, pm25, std,
                                     acc, baseline)
        pd.DataFrame(board).to_csv(out_csv, index=False)
        pd.DataFrame(long).to_csv(os.path.join(pm_dir, f"{sp.name}_pervar.csv"), index=False)
        del fn  # free the (possibly GPU) model before the next method
        log(f"[{dsname}] saved {sp.name}  ({(time.perf_counter()-t0)/60:.1f} min)")

    return aggregate_dataset(ds_dir)


def aggregate_dataset(ds_dir):
    """Concatenate per-method CSVs in ``<ds_dir>/per_method`` into the dataset tables."""
    pm_dir = os.path.join(ds_dir, "per_method")
    boards, longs = [], []
    for fn in sorted(os.listdir(pm_dir)) if os.path.isdir(pm_dir) else []:
        path = os.path.join(pm_dir, fn)
        if fn.endswith("_pervar.csv"):
            longs.append(pd.read_csv(path))
        elif fn.endswith(".csv"):
            boards.append(pd.read_csv(path))
    if not boards:
        return pd.DataFrame()
    board = pd.concat(boards, ignore_index=True).sort_values(
        ["pattern", "overall_std_rmse"]).reset_index(drop=True)
    board.to_csv(os.path.join(ds_dir, "leaderboard.csv"), index=False)
    if longs:
        pd.concat(longs, ignore_index=True).to_csv(
            os.path.join(ds_dir, "per_variable.csv"), index=False)
    return board


def aggregate_all(results_base):
    """Combine every dataset's leaderboard into one table under ``results_base``."""
    boards = []
    for name in sorted(os.listdir(results_base)) if os.path.isdir(results_base) else []:
        ds_dir = os.path.join(results_base, name)
        if not os.path.isdir(ds_dir):
            continue
        b = aggregate_dataset(ds_dir)
        if len(b):
            boards.append(b)
    if not boards:
        return pd.DataFrame()
    allb = pd.concat(boards, ignore_index=True)
    allb.to_csv(os.path.join(results_base, "leaderboard_all.csv"), index=False)
    return allb


# ---------------------------------------------------------------------------
# Coverage map: every technique the user named -> disposition
# ---------------------------------------------------------------------------

def coverage_map() -> pd.DataFrame:
    """Map each survey-listed technique to run / subsumed / skipped (with reason)."""
    R, S, K = "run", "subsumed", "skipped"
    rows: list[tuple] = [
        # (technique, status, mapped_to, note)
        ("Row mean", S, "mean", "cross-sectional mean ~ mean baseline"),
        ("Mean top-bottom", S, "last_and_next_mean", "avg of neighbours"),
        ("Hour mean", R, "hour_mean", ""),
        ("6-hour mean", S, "hour_mean", "coarser hour-grouping variant"),
        ("12-hour mean", S, "hour_mean", "coarser hour-grouping variant"),
        ("Daily mean", R, "daily_mean", ""),
        ("Last-and-next mean", R, "last_and_next_mean", ""),
        ("Previous-year mean", S, "daily_mean", "calendar-group mean; <3y limits a full lag"),
        ("Conditional mean imputation", R, "mice", "regression conditional mean"),
        ("Stochastic regression", R, "stochastic_regression", ""),
        ("ARMA", S, "arima", "ARMA = ARIMA(d=0)"),
        ("ARIMA", R, "arima", ""),
        ("Linear interpolation", R, "linear_interp", ""),
        ("Cubic interpolation", R, "cubic_spline", ""),
        ("Inverse distance weighting", R, "spatial_idw", "equal-weight stand-in: no station coords"),
        ("Kriging", K, "", "needs station lat/lon coordinates (not in dataset)"),
        ("Optimal interpolation", S, "spatial_idw", "cross-station weighting stand-in"),
        ("Nearest neighbor", R, "nearest_neighbor", ""),
        ("Hot deck", S, "nearest_neighbor", "donor = nearest observed row"),
        ("Cold deck", K, "", "needs an external donor dataset"),
        ("Multiple imputation", R, "mice", "MICE is the canonical MI engine"),
        ("Maximum Likelihood Imputation (MLI)", R, "em_gaussian", ""),
        ("Expectation Maximization (EM)", R, "em_gaussian", ""),
        ("Full Information ML (FIML)", S, "em_gaussian", "Gaussian EM gives the FIML estimates"),
        ("Probabilistic Matrix Factorization (PMF)", R, "matrix_factorization", ""),
        ("Singular Value Decomposition (SVD)", R, "iterative_svd", ""),
        ("Tensor decomposition", K, "", "no natural 3rd mode here; deferred"),
        ("Principal Component Analysis (PCA)", R, "iterative_svd", "iterative PCA completion"),
        ("Probabilistic PCA (PPCA)", R, "ppca", ""),
        ("Bayesian PCA (BPCA)", S, "ppca", "PPCA is the point-estimate special case"),
        ("K-Nearest Neighbor (KNN)", R, "knn", ""),
        ("Weighted KNN", R, "weighted_knn", ""),
        ("Sequential KNN", S, "knn", "ordered-variable KNN; same engine"),
        ("Gray KNN", K, "", "no public reference implementation"),
        ("Modified/purity-based KNN", K, "", "no public reference implementation"),
        ("Box-Jenkins time-series", S, "arima", "Box-Jenkins = ARIMA identification"),
        ("Kalman filter / smoothing / state-space", R, "kalman_smoother", ""),
        ("Singular Spectrum Analysis (SSA)", R, "ssa", ""),
        ("Kernel-based imputation", K, "", "bespoke; no standard impl"),
        ("Mixture-kernel imputation", K, "", "bespoke; no standard impl"),
        ("Ratio-Based Imputation (RBI)", K, "", "no public reference implementation"),
        ("Iterative RBI (IRBI)", K, "", "no public reference implementation"),
        ("Bayesian imputation", S, "stochastic_regression", "posterior-sampling MICE"),
        ("MCMC / Data Augmentation", S, "stochastic_regression", "chained posterior draws"),
        ("MICE", R, "mice", ""),
        ("EM with Bootstrapping (EMB / Amelia II)", R, "emb_bootstrap", ""),
        ("Multilayer Perceptron (MLP)", R, "autoencoder", "MLP autoencoder"),
        ("Radial Basis Function (RBF)", K, "", "no standard RBF-net imputer"),
        ("Auto-associative neural network", S, "autoencoder", "auto-associative = autoencoder"),
        ("Probabilistic Neural Network (PNN)", K, "", "classifier, not an imputer"),
        ("Bayesian network imputation", K, "", "needs a hand-built DAG"),
        ("Support Vector Machine (SVM)", R, "svr_regression", ""),
        ("Least-Squares SVM", S, "svr_regression", "LS-SVM ~ SVR variant"),
        ("Decision Tree (ID3/C4.5/CART)", R, "decision_tree_cart", "CART regressor"),
        ("EMI/DMI/SiMI tree extensions", K, "", "bespoke research artifacts"),
        ("Random Forest / MissForest", R, "missforest_rf", ""),
        ("RF proximity/on-the-fly/multivariate", S, "missforest_rf", "same RF engine"),
        ("Single-view clustering", R, "kmeans", ""),
        ("Multi-view / subspace / MKL clustering", K, "", "bespoke; no standard impl"),
        ("k-means clustering", R, "kmeans", ""),
        ("Self-Organizing Map (SOM)", R, "som", ""),
        ("Fuzzy rule-based / fuzzy rough", K, "", "bespoke; no standard impl"),
        ("Fuzzy clustering / c-means / k-means", R, "fuzzy_cmeans", ""),
        ("Fuzzy neighborhood density clustering", K, "", "bespoke; no standard impl"),
        ("Grey-system fuzzy c-means", K, "", "bespoke; no standard impl"),
        ("Iterative fuzzy k-means / IFC", S, "fuzzy_cmeans", "iterative FCM variant"),
        ("DFIC / D-ANFIS", K, "", "bespoke research artifacts"),
        ("Deep autoencoder", R, "autoencoder", ""),
        ("Backpropagation autoencoder", S, "autoencoder", "standard BP-trained AE"),
        ("Variational autoencoder", R, "gpvae", "GP-VAE (temporal VAE)"),
        ("Denoising autoencoder", R, "denoising_ae", ""),
        ("Stacked / multimodal denoising AE", S, "denoising_ae", "deeper DAE variant"),
        ("RNN", S, "brits", "recurrent imputation"),
        ("GRU", S, "grud", ""),
        ("LSTM", S, "brits", "BRITS uses LSTM cells"),
        ("ConvLSTM", K, "", "no standard imputation impl"),
        ("Transfer/iterative LSTM imputation", S, "brits", "same recurrent engine"),
        ("GRU-D", R, "grud", ""),
        ("M-RNN", R, "mrnn", ""),
        ("CNN / ST-spectral CNN", S, "timesnet", "TimesNet uses 2D conv inception blocks"),
        ("GAN", S, "usgan", ""),
        ("GAIN", S, "usgan", "US-GAN adversarial imputer"),
        ("Multi-channel CNN + DCGAN", K, "", "bespoke; no standard impl"),
        ("SAITS / Transformer imputation", R, "saits", ""),
        ("CSDI (diffusion)", R, "csdi", ""),
        ("TimesNet", R, "timesnet", ""),
        ("Hybrid: MICE + KNN", S, "mice", "components benchmarked separately"),
        ("Hybrid: FCM + SVR + GA", K, "", "bespoke pipeline"),
        ("Hybrid: SOM + FOA + LS-SVM", K, "", "bespoke pipeline"),
        ("Hybrid: multiple kernel clustering", K, "", "bespoke; no standard impl"),
        ("Hybrid: bagging / block-bootstrap", S, "emb_bootstrap", "bootstrap ensembling"),
        ("Hybrid: Stineman / weighted moving avg", S, "last_and_next_mean", "interp/MA hybrids"),
        ("Hybrid: Kalman-filter hybrids", S, "kalman_smoother", ""),
        ("Hybrid: KNN + penalized dissimilarity", S, "weighted_knn", ""),
        ("Hybrid: SOM + KNN ensemble", S, "som", "components benchmarked separately"),
        ("Hybrid: LSTM + transfer / bidirectional", S, "brits", "BRITS is bidirectional"),
    ]
    out = pd.DataFrame(rows, columns=["technique", "status", "mapped_to", "note"])
    return out
