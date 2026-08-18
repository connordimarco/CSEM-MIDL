"""Combine multiple solar wind datasets using agreement-first source selection.

Ports the MIDL pipeline's multi-satellite combination algorithm
(l1_combine.py, l1_quality.py, l1_filters.py) into a generalized form
that works with arbitrary named xarray Datasets.
"""

from __future__ import annotations

from itertools import combinations as _combinations

import numpy as np
import pandas as pd
import xarray as xr

_NUMERIC_COLS = ("Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T")

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "Bx": 8.0,
    "By": 8.0,
    "Bz": 8.0,
    "|B|": 8.0,
    "Ux": 80.0,
    "Uy": 40.0,
    "Uz": 40.0,
    "|Vt|": 40.0,
    "rho": 2.0,
}

_SWITCH_MIN = 3

_T_SPIKY_LOG_STD = 0.5
_T_SPIKY_WINDOW = 11

_TRANSITION_COLS: dict[str, str] = {
    "Ux": "pct",
    "Uy": "abs",
    "Uz": "abs",
    "rho": "pct",
    "T": "pct",
}

# ---------------------------------------------------------------------------
# Quality scoring (ported from l1_quality.py)
# ---------------------------------------------------------------------------

_PLATEAU_PARAMS: dict[str, dict] = {
    "Ux": {"window": 15, "std_thresh": 1.0, "max_unique": 3},
    "Uy": {"window": 11, "std_thresh": 0.08, "max_unique": 3},
    "Uz": {"window": 11, "std_thresh": 0.08, "max_unique": 3},
    "rho": {"window": 15, "std_thresh": 0.05, "max_unique": 3},
}

_OUTLIER_PARAMS: dict[str, dict] = {
    "Ux": {"mode": "abs", "threshold": 50.0, "window": 31},
    "Uy": {"mode": "abs", "threshold": 30.0, "window": 31},
    "Uz": {"mode": "abs", "threshold": 30.0, "window": 31},
    "rho": {"mode": "ratio", "threshold": 3.0, "window": 61},
}

_QUALITY_VARS = ["Ux", "Uy", "Uz", "rho"]


def _detect_flat_plateau(series: pd.Series, window: int,
                         std_thresh: float, max_unique: int) -> pd.Series:
    s = series.copy()
    if s.empty or s.isna().all():
        return pd.Series(False, index=s.index)
    rolling_std = s.rolling(window=window, center=True, min_periods=5).std()
    rolling_unique = s.rolling(window=window, center=True, min_periods=5).apply(
        lambda x: pd.Series(x).dropna().nunique(), raw=False,
    )
    mask = (rolling_std <= std_thresh) & (rolling_unique <= max_unique)
    return mask.fillna(False)


def _check_flat_plateau(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for var, p in _PLATEAU_PARAMS.items():
        if var not in df.columns or df[var].isna().all():
            masks[var] = pd.Series(False, index=df.index)
            continue
        masks[var] = _detect_flat_plateau(
            df[var], p["window"], p["std_thresh"], p["max_unique"])
    return masks


def _check_outlier(source_dfs: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.Series]]:
    names = sorted(source_dfs.keys())
    idx = next(iter(source_dfs.values())).index
    result: dict[str, dict[str, pd.Series]] = {name: {} for name in names}

    for var in _QUALITY_VARS:
        p = _OUTLIER_PARAMS[var]
        w = p["window"]

        s: dict[str, pd.Series] = {}
        for name in names:
            df = source_dfs[name]
            s[name] = df[var].reindex(idx) if var in df.columns else pd.Series(np.nan, index=idx)

        has_data = [name for name in names if not s[name].isna().all()]

        if len(has_data) < 3:
            for name in names:
                result[name][var] = pd.Series(False, index=idx)
            continue

        pair_ok: dict[tuple[str, str], pd.Series] = {}
        for a, b in _combinations(has_data, 2):
            key = (a, b)
            if p["mode"] == "abs":
                dev = (s[a] - s[b]).abs()
                roll = dev.rolling(window=w, center=True, min_periods=5).median()
                pair_ok[key] = (roll <= p["threshold"]).fillna(True)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = s[a] / s[b]
                log_r = np.log(ratio.clip(lower=1e-10)).abs()
                log_thresh = np.log(p["threshold"])
                roll = log_r.rolling(window=w, center=True, min_periods=5).median()
                pair_ok[key] = (roll <= log_thresh).fillna(True)

        for name in names:
            if name not in has_data:
                result[name][var] = pd.Series(False, index=idx)
                continue
            others = [n for n in has_data if n != name]
            flagged = pd.Series(False, index=idx)
            for a, b in _combinations(others, 2):
                others_agree = pair_ok[(a, b)]
                key_a = (min(name, a), max(name, a))
                key_b = (min(name, b), max(name, b))
                disagrees_a = ~pair_ok[key_a]
                disagrees_b = ~pair_ok[key_b]
                flagged = flagged | (others_agree & disagrees_a & disagrees_b)
            result[name][var] = flagged.fillna(False)

    return result


def _score_quality(source_dfs: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.Series]]:
    idx = next(iter(source_dfs.values())).index
    outlier_masks = _check_outlier(source_dfs)

    all_bad: dict[str, dict[str, pd.Series]] = {}
    for name, df_target in source_dfs.items():
        plateau_m = _check_flat_plateau(df_target)
        bad: dict[str, pd.Series] = {}
        for var in _QUALITY_VARS:
            composite = pd.Series(False, index=idx)
            if var in outlier_masks.get(name, {}):
                composite = composite | outlier_masks[name][var]
            if var in plateau_m:
                composite = composite | plateau_m[var]
            bad[var] = composite
        all_bad[name] = bad

    return all_bad


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------

def _median_filter_3(a: np.ndarray, min_periods: int = 2) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    if len(a) < 3:
        return a.copy()
    return pd.Series(a).rolling(
        window=3, center=True, min_periods=min_periods).median().to_numpy()


# ---------------------------------------------------------------------------
# Core selection algorithm
# ---------------------------------------------------------------------------

def _agree(v1: float, v2: float, threshold: float) -> bool:
    return abs(v1 - v2) <= threshold


def _fallback_source(
    values: dict[str, float],
    available_keys: list[str],
    prev_value: float,
    deprioritize_keys: list[str] | None = None,
) -> str:
    if deprioritize_keys:
        alternatives = [k for k in available_keys if k not in deprioritize_keys]
        if alternatives:
            if len(alternatives) == 1:
                return alternatives[0]
            available_keys = alternatives

    if np.isfinite(prev_value):
        return min(available_keys, key=lambda k: abs(values[k] - prev_value))
    return available_keys[0]


def _select_column(
    col: str,
    source_series: dict[str, pd.Series],
    threshold: float,
    bad_masks: dict[str, dict[str, pd.Series]] | None = None,
    deprioritize_keys: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = next(iter(source_series.values())).index
    n = len(index)
    out_vals = np.full(n, np.nan, dtype=float)
    out_nsat = np.zeros(n, dtype=int)
    out_source: list[frozenset[str] | None] = [None] * n

    prev_value = np.nan
    locked_source: str | None = None
    switch_count = 0

    for i in range(n):
        values: dict[str, float] = {}
        for key, series in source_series.items():
            v = series.iloc[i]
            if pd.notna(v):
                values[key] = v

        if bad_masks is not None:
            for key in list(values.keys()):
                src_masks = bad_masks.get(key)
                if src_masks is not None and col in src_masks:
                    if bool(src_masks[col].iloc[i]):
                        values.pop(key, None)

        available = sorted(values.keys())
        n_sat = len(available)
        out_nsat[i] = n_sat

        if n_sat == 0:
            continue

        if n_sat == 1:
            out_vals[i] = values[available[0]]
            out_source[i] = frozenset(available)
            prev_value = out_vals[i]
            continue

        pairs: list[tuple[str, str]] = []
        pairs_set: set[tuple[str, str]] = set()
        for p_idx, c1 in enumerate(available):
            for c2 in available[p_idx + 1:]:
                if _agree(values[c1], values[c2], threshold):
                    pairs.append((c1, c2))
                    pairs_set.add((c1, c2))

        max_pairs = n_sat * (n_sat - 1) // 2
        if len(pairs) == max_pairs:
            out_vals[i] = np.median([values[c] for c in available])
            out_source[i] = frozenset(available)
            prev_value = out_vals[i]
            continue

        best_clique: tuple[str, ...] | None = None
        if n_sat >= 4 and len(pairs) >= 3:
            for size in range(n_sat - 1, 2, -1):
                for subset in _combinations(available, size):
                    if all((a, b) in pairs_set
                           for a, b in _combinations(subset, 2)):
                        best_clique = subset
                        break
                if best_clique:
                    break

        if best_clique:
            out_vals[i] = np.median([values[c] for c in best_clique])
            out_source[i] = frozenset(best_clique)
            prev_value = out_vals[i]
            continue

        if pairs:
            best_pair = min(pairs, key=lambda p: abs(values[p[0]] - values[p[1]]))
            out_vals[i] = 0.5 * (values[best_pair[0]] + values[best_pair[1]])
            out_source[i] = frozenset(best_pair)
            prev_value = out_vals[i]
            continue

        candidate = _fallback_source(values, available, prev_value,
                                     deprioritize_keys=deprioritize_keys)

        if locked_source is None or locked_source not in available:
            locked_source = candidate
            switch_count = 0
        elif candidate == locked_source:
            switch_count = 0
        else:
            switch_count += 1
            if switch_count >= _SWITCH_MIN:
                locked_source = candidate
                switch_count = 0

        out_vals[i] = values[locked_source]
        out_source[i] = frozenset([locked_source])
        prev_value = out_vals[i]

    return (pd.Series(out_vals, index=index),
            pd.Series(out_nsat, index=index),
            pd.Series(out_source, index=index))


def _apply_source_to_components(
    source_series: pd.Series,
    component_source_series: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    out = np.full(len(index), np.nan, dtype=float)
    for i in range(len(index)):
        src = source_series.iloc[i]
        if src is None:
            continue
        keys = sorted(src)
        vals = [component_source_series[k].iloc[i]
                for k in keys
                if pd.notna(component_source_series[k].iloc[i])]
        if not vals:
            continue
        if len(vals) == 1:
            out[i] = vals[0]
        elif len(vals) == 2:
            out[i] = 0.5 * (vals[0] + vals[1])
        else:
            out[i] = np.median(vals)
    return pd.Series(out, index=index)


# ---------------------------------------------------------------------------
# Variable combination orchestrator
# ---------------------------------------------------------------------------

def _combine_variables(
    source_dfs: dict[str, pd.DataFrame],
    master_grid: pd.DatetimeIndex,
    thresholds: dict[str, float],
    deprioritize: dict[str, list[str]] | None,
    bad_masks: dict[str, dict[str, pd.Series]] | None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    cols = ["Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho"]
    df_combined = pd.DataFrame(index=master_grid, columns=cols, dtype=float)
    source_map: dict[str, pd.Series] = {}

    names = sorted(source_dfs.keys())

    def _series_for(col: str) -> dict[str, pd.Series]:
        return {
            name: source_dfs[name][col] if col in source_dfs[name].columns
            else pd.Series(np.nan, index=master_grid)
            for name in names
        }

    # --- Block A: Magnetic field (Bx, By, Bz) coupled via |B| ---
    b_series = {comp: _series_for(comp) for comp in ("Bx", "By", "Bz")}
    mag_b_series: dict[str, pd.Series] = {}
    for name in names:
        mag_b_series[name] = np.sqrt(
            b_series["Bx"][name] ** 2 +
            b_series["By"][name] ** 2 +
            b_series["Bz"][name] ** 2)

    _, _, b_source = _select_column(
        "|B|", mag_b_series, thresholds.get("|B|", 8.0), bad_masks=None)

    for comp in ("Bx", "By", "Bz"):
        df_combined[comp] = _apply_source_to_components(
            b_source, b_series[comp], master_grid)
        source_map[comp] = b_source

    # --- Block B: Transverse velocity (Uy, Uz) coupled via |Vt| ---
    vt_series = {comp: _series_for(comp) for comp in ("Uy", "Uz")}
    mag_vt_series: dict[str, pd.Series] = {}
    for name in names:
        mag_vt_series[name] = np.sqrt(
            vt_series["Uy"][name] ** 2 +
            vt_series["Uz"][name] ** 2)

    vt_bad_masks: dict[str, dict[str, pd.Series]] | None = None
    if bad_masks is not None:
        vt_bad_masks = {}
        for name in names:
            name_masks = bad_masks.get(name)
            if name_masks is not None:
                uy_bad = name_masks.get("Uy", pd.Series(False, index=master_grid))
                uz_bad = name_masks.get("Uz", pd.Series(False, index=master_grid))
                vt_bad_masks[name] = {"|Vt|": uy_bad | uz_bad}

    _, _, vt_source = _select_column(
        "|Vt|", mag_vt_series, thresholds.get("|Vt|", 40.0),
        bad_masks=vt_bad_masks)

    for comp in ("Uy", "Uz"):
        df_combined[comp] = _apply_source_to_components(
            vt_source, vt_series[comp], master_grid)
        source_map[comp] = vt_source

    # --- Block C: Independent variables (Ux, rho) ---
    for col in ("Ux", "rho"):
        ss = _series_for(col)
        depri = deprioritize.get(col) if deprioritize else None
        values, _, source = _select_column(
            col, ss, thresholds.get(col, np.inf),
            bad_masks=bad_masks, deprioritize_keys=depri)
        df_combined[col] = values
        source_map[col] = source

    return df_combined, source_map


# ---------------------------------------------------------------------------
# Temperature combination
# ---------------------------------------------------------------------------

def _combine_temperature(
    source_dfs: dict[str, pd.DataFrame],
    master_grid: pd.DatetimeIndex,
    deprioritize_keys: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    names = sorted(source_dfs.keys())

    sat_T: dict[str, pd.Series] = {}
    for name in names:
        df = source_dfs[name]
        if "T" in df.columns and not df["T"].isna().all():
            s = df["T"].reindex(master_grid)
            s = s.interpolate(method="time", limit=2, limit_area="inside")
            s = pd.Series(_median_filter_3(s.values), index=master_grid)
            log_std = (
                np.log(s.clip(lower=1))
                .rolling(_T_SPIKY_WINDOW, center=True, min_periods=5)
                .std()
            )
            s = s.where(log_std <= _T_SPIKY_LOG_STD, other=np.nan)
            sat_T[name] = s
        else:
            sat_T[name] = pd.Series(np.nan, index=master_grid)

    df_t = pd.DataFrame(sat_T)
    log_df = np.log(df_t.clip(lower=1))
    log_median = log_df.median(axis=1, skipna=True)
    out = np.where(df_t.notna().any(axis=1), np.exp(log_median), np.nan)

    mask_2sat_depri = pd.Series(False, index=master_grid)
    if deprioritize_keys:
        n_available = df_t.notna().sum(axis=1)
        depri_present = pd.Series(False, index=master_grid)
        for dk in deprioritize_keys:
            if dk in df_t.columns:
                depri_present = depri_present | df_t[dk].notna()
        mask_2sat_depri = (n_available == 2) & depri_present
        non_depri_cols = [n for n in names if n not in deprioritize_keys]
        if non_depri_cols:
            non_depri = df_t[non_depri_cols].bfill(axis=1).iloc[:, 0]
            out = np.where(mask_2sat_depri, non_depri.values, out)

    t_source: list[frozenset[str] | None] = [None] * len(master_grid)
    for i in range(len(master_grid)):
        contribs = frozenset(
            name for name in names if pd.notna(sat_T[name].iloc[i]))
        if deprioritize_keys and mask_2sat_depri.iloc[i]:
            contribs = contribs - set(deprioritize_keys)
        t_source[i] = contribs if contribs else None
    t_source_series = pd.Series(t_source, index=master_grid)

    combined = pd.Series(out, index=master_grid)
    combined = pd.Series(_median_filter_3(combined.values), index=master_grid)
    return combined, t_source_series


# ---------------------------------------------------------------------------
# Transition smoothing
# ---------------------------------------------------------------------------

def _jump_magnitude(m1: float, m2: float, col_type: str) -> float:
    if col_type == "pct":
        a, b = abs(m1), abs(m2)
        if min(a, b) == 0:
            return 0.0
        return 100.0 * (max(a, b) / min(a, b) - 1.0)
    return abs(m2 - m1)


def _apply_boxcar(smoothed: np.ndarray, original: np.ndarray,
                  i: int, w: int) -> None:
    n = len(original)
    half = w // 2
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    ext_lo = max(0, lo - half)
    ext_hi = min(n, hi + half)
    rolled = (pd.Series(original[ext_lo:ext_hi])
              .rolling(w, center=True, min_periods=1)
              .mean().values)
    smoothed[lo:hi] = rolled[(lo - ext_lo):(lo - ext_lo) + (hi - lo)]


def _smooth_transitions(
    df: pd.DataFrame,
    source_map: dict[str, pd.Series],
    cmax: float = 20.0,
    wmax: int = 60,
    rate: float = 5.0,
) -> pd.DataFrame:
    out = df.copy()

    source_changed: dict[str, pd.Series] = {}
    for col, src_series in source_map.items():
        changed = pd.Series(False, index=src_series.index)
        for i in range(1, len(src_series)):
            prev_src = src_series.iloc[i - 1]
            curr_src = src_series.iloc[i]
            if prev_src != curr_src and prev_src is not None and curr_src is not None:
                changed.iloc[i] = True
        source_changed[col] = changed

    for col, col_type in _TRANSITION_COLS.items():
        if col not in out.columns:
            continue
        sc = source_changed.get(col)
        original = out[col].values.astype(float)
        smoothed = original.copy()

        for i in range(1, len(original)):
            if sc is not None and not bool(sc.iloc[i]):
                continue
            m1, m2 = original[i - 1], original[i]
            if not (np.isfinite(m1) and np.isfinite(m2)):
                continue
            c = _jump_magnitude(m1, m2, col_type)
            if c <= cmax:
                continue
            w = max(2, int(np.round(min(wmax, c / rate))))
            _apply_boxcar(smoothed, original, i, w)

        out[col] = smoothed

    return out


# ---------------------------------------------------------------------------
# Source provenance encoding
# ---------------------------------------------------------------------------

def _encode_provenance(source_map: dict[str, pd.Series],
                       master_grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    provenance: dict[str, pd.Series] = {}

    b_src = source_map.get("Bx")
    if b_src is not None:
        provenance["B_source"] = b_src.apply(
            lambda s: ",".join(sorted(s)) if s else "")

    ux_src = source_map.get("Ux")
    if ux_src is not None:
        provenance["Ux_source"] = ux_src.apply(
            lambda s: ",".join(sorted(s)) if s else "")

    uyz_src = source_map.get("Uy")
    if uyz_src is not None:
        provenance["Uyz_source"] = uyz_src.apply(
            lambda s: ",".join(sorted(s)) if s else "")

    rho_src = source_map.get("rho")
    if rho_src is not None:
        provenance["rho_source"] = rho_src.apply(
            lambda s: ",".join(sorted(s)) if s else "")

    t_src = source_map.get("T")
    if t_src is not None:
        provenance["T_source"] = t_src.apply(
            lambda s: ",".join(sorted(s)) if s else "")

    return provenance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VAR_ATTRS: dict[str, dict[str, str]] = {
    "Bx": {"units": "nT", "long_name": "Magnetic field X component", "coordinate_system": "GSM"},
    "By": {"units": "nT", "long_name": "Magnetic field Y component", "coordinate_system": "GSM"},
    "Bz": {"units": "nT", "long_name": "Magnetic field Z component", "coordinate_system": "GSM"},
    "Ux": {"units": "km/s", "long_name": "Bulk velocity X component", "coordinate_system": "GSM"},
    "Uy": {"units": "km/s", "long_name": "Bulk velocity Y component", "coordinate_system": "GSM"},
    "Uz": {"units": "km/s", "long_name": "Bulk velocity Z component", "coordinate_system": "GSM"},
    "rho": {"units": "cm^-3", "long_name": "Proton number density"},
    "T": {"units": "K", "long_name": "Proton temperature"},
    "B_source": {"long_name": "Magnetic field source(s)"},
    "Ux_source": {"long_name": "Ux source(s)"},
    "Uyz_source": {"long_name": "Uy/Uz source(s)"},
    "rho_source": {"long_name": "Density source(s)"},
    "T_source": {"long_name": "Temperature source(s)"},
    "B_interp": {"long_name": "Magnetic field interpolation-provenance flag",
                 "flag_values": "0, 1, 2",
                 "flag_meanings": "direct mixed all_interpolated"},
    "Ux_interp": {"long_name": "Ux interpolation-provenance flag",
                  "flag_values": "0, 1, 2",
                  "flag_meanings": "direct mixed all_interpolated"},
    "Uyz_interp": {"long_name": "Uy/Uz interpolation-provenance flag",
                   "flag_values": "0, 1, 2",
                   "flag_meanings": "direct mixed all_interpolated"},
    "rho_interp": {"long_name": "Density interpolation-provenance flag",
                   "flag_values": "0, 1, 2",
                   "flag_meanings": "direct mixed all_interpolated"},
    "T_interp": {"long_name": "Temperature interpolation-provenance flag",
                 "flag_values": "0, 1, 2",
                 "flag_meanings": "direct mixed all_interpolated"},
}


def _apply_var_attrs(result_ds: xr.Dataset, coords_system: str) -> None:
    """Stamp standard per-variable attrs, honoring the shared frame.

    Copies each attrs dict per call so the module-level ``_VAR_ATTRS``
    template is never mutated.
    """
    for var, attrs in _VAR_ATTRS.items():
        if var in result_ds:
            var_attrs = dict(attrs)
            if "coordinate_system" in var_attrs:
                var_attrs["coordinate_system"] = coords_system
            result_ds[var].attrs.update(var_attrs)


def merge(
    datasets: dict[str, xr.Dataset],
    thresholds: dict[str, float] | None = None,
    deprioritize: dict[str, list[str]] | None = None,
    smooth: bool = True,
    smooth_cmax: float = 20.0,
    smooth_wmax: int = 60,
    smooth_rate: float = 5.0,
    quality: bool = False,
) -> xr.Dataset:
    """Combine multiple solar wind datasets using MIDL's source selection algorithm.

    Uses the same agreement-first logic as the MIDL pipeline: when sources
    agree within the thresholds their values are averaged or medianed; when
    they disagree, a continuity-based fallback with hysteresis prevents
    rapid oscillation between sources.

    Parameters
    ----------
    datasets : dict[str, xarray.Dataset]
        Named datasets to merge.  Keys are arbitrary source names (e.g.
        ``"ace"``, ``"imp8"``).  Each Dataset must have a ``time``
        coordinate and the standard solar wind variables (Bx, By, Bz,
        Ux, Uy, Uz, rho, T).  Missing variables are treated as all-NaN.
        All inputs must share one coordinate system
        (``attrs['coords_system']``, default ``"GSM"``); mixing frames
        raises ``ValueError``, and the shared frame is stamped onto the
        merged output.
    thresholds : dict[str, float] or None
        Per-variable agreement thresholds.  Defaults match the MIDL
        pipeline: B ±8 nT, Ux ±80 km/s, Uy/Uz ±40 km/s, rho ±2 cm⁻³.
    deprioritize : dict[str, list[str]] or None
        Sources to deprioritize per variable in the fallback path.
        Example: ``{"rho": ["dscovr"], "T": ["dscovr"]}``.
    smooth : bool
        Apply transition smoothing at source changes (default True).
    smooth_cmax, smooth_wmax, smooth_rate : float
        Transition smoothing parameters.
    quality : bool
        Run quality scoring (plateau + outlier detection) before source
        selection.  Requires ≥3 sources for outlier detection to be
        useful.  Default False.

    Returns
    -------
    xarray.Dataset
        Merged dataset with standard variables plus provenance columns
        (B_source, Ux_source, Uyz_source, rho_source, T_source).
    """
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("datasets must be a non-empty dict mapping names to xarray Datasets")
    for name, ds in datasets.items():
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"datasets[{name!r}] must be an xarray.Dataset, got {type(ds).__name__}")
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError(f"datasets[{name!r}] must have a 'time' coordinate")
        times = pd.DatetimeIndex(ds.time.values)
        if not (times == times.floor("min")).all():
            raise ValueError(
                f"datasets[{name!r}] has timestamps not aligned to whole minutes. "
                "Resample to 1-minute cadence first, e.g. "
                "ds.resample(time='1min').mean()"
            )

    coords_seen = {
        ds.attrs.get("coords_system", "GSM") for ds in datasets.values()
    }
    if len(coords_seen) > 1:
        raise ValueError(
            "Cannot merge datasets in different coordinate systems: "
            f"{sorted(coords_seen)}. Rotate all inputs to a common frame first."
        )
    coords_system = coords_seen.pop()

    th = dict(_DEFAULT_THRESHOLDS)
    if thresholds is not None:
        th.update(thresholds)

    all_times: list[pd.DatetimeIndex] = []
    for ds in datasets.values():
        t = pd.DatetimeIndex(ds.time.values)
        all_times.append(t)

    t_min = min(t.min() for t in all_times)
    t_max = max(t.max() for t in all_times)
    master_grid = pd.date_range(t_min.floor("min"), t_max.ceil("min"), freq="min")

    source_dfs: dict[str, pd.DataFrame] = {}
    for name, ds in datasets.items():
        present = list(set(_NUMERIC_COLS) & set(ds.data_vars))
        df = ds[present].to_dataframe() if present else pd.DataFrame(index=pd.DatetimeIndex(ds.time.values))
        df = df.reindex(master_grid)
        for col in _NUMERIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        source_dfs[name] = df

    # --- Single-source fast path ---
    if len(datasets) == 1:
        name = next(iter(datasets))
        df_out = source_dfs[name]
        source_map: dict[str, pd.Series] = {}
        for col in ("Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"):
            has_val = df_out[col].notna() if col in df_out.columns else pd.Series(False, index=master_grid)
            source_map[col] = has_val.apply(
                lambda v, _n=name: frozenset([_n]) if v else None)
        prov = _encode_provenance(source_map, master_grid)
        for pcol, pseries in prov.items():
            df_out[pcol] = pseries
        result_ds = xr.Dataset.from_dataframe(df_out.rename_axis("time"))
        result_ds.attrs["source"] = "MIDL merge"
        result_ds.attrs["coords_system"] = coords_system
        _apply_var_attrs(result_ds, coords_system)
        return result_ds

    # --- Quality scoring ---
    bad_masks: dict[str, dict[str, pd.Series]] | None = None
    if quality:
        bad_masks = _score_quality(source_dfs)

    # --- Combine B, Ux, Uy, Uz, rho ---
    df_combined, source_map = _combine_variables(
        source_dfs, master_grid, th, deprioritize, bad_masks)

    # --- Combine temperature ---
    t_depri = deprioritize.get("T") if deprioritize else None
    t_combined, t_source = _combine_temperature(
        source_dfs, master_grid, deprioritize_keys=t_depri)
    df_combined["T"] = t_combined
    source_map["T"] = t_source

    # --- Transition smoothing ---
    if smooth:
        df_combined = _smooth_transitions(
            df_combined, source_map,
            cmax=smooth_cmax, wmax=smooth_wmax, rate=smooth_rate)

    # --- Build output Dataset ---
    prov = _encode_provenance(source_map, master_grid)
    for pcol, pseries in prov.items():
        df_combined[pcol] = pseries

    result_ds = xr.Dataset.from_dataframe(df_combined.rename_axis("time"))
    result_ds.attrs["source"] = "MIDL merge"
    result_ds.attrs["coords_system"] = coords_system
    _apply_var_attrs(result_ds, coords_system)

    return result_ds
