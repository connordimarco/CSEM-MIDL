"""Client-side propagation of MIDL L1 datasets to an inner boundary."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

RE_KM = 6371.0

_NUMERIC_COLS = ("Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T")

METHODS = {"ballistic"}

# Interpolation-provenance flag groups -> representative data column. The
# merged 0/1/2 flag level cannot ride through the numeric regridding as a
# single number (index interpolation blends adjacent parcels), so each level
# is decomposed into two monotone 0/1 carriers, exactly as the pipeline does
# in l1_midl.py (and MIDL-Web propagate.js):
#   <group>__interp_any = level >= 1, <group>__interp_all = level >= 2.
# After regridding the smeared carriers binarize back:
#   any: blend > 0  (a flagged parent contributed - conservative),
#   all: blend == 1 (every contributing parent was all-interpolated).
_FLAG_REPR_COL = {"B": "Bx", "Ux": "Ux", "Uyz": "Uy", "rho": "rho", "T": "T"}
_FLAG_GROUPS = tuple(_FLAG_REPR_COL)
_INTERP_SUFFIX = "_interp"
_CARRY_ANY = "__interp_any"
_CARRY_ALL = "__interp_all"


def _ballistic_propagate_df(df: pd.DataFrame, target_re: float) -> pd.DataFrame:
    """Ballistically propagate L1 observations to ``target_re`` (Earth radii).

    Matches the pipeline's [l1_propagation.py](https://github.com/...)
    algorithm:

    1. Gap-fill Ux (time-interpolated) for travel-time computation only.
    2. Travel time per minute = (X - target) / |Ux|, using each parcel's
       own source position.
    3. Enforce causality: a parcel is kept only if its arrival time is
       earlier than every later parcel's arrival time (O(n) running min).
    4. Resample onto the original 1-minute grid via index interpolation
       (limit=2 to bridge sub-minute jitter).
    5. Restore the original Ux NaN mask so timing fills don't leak into
       output values.

    Any ``{B,Ux,Uyz,rho,T}_interp`` provenance flags present on ``df`` are
    carried through (via the any/all carrier decomposition) and reassembled
    onto the output grid. The flags take values 0 (direct), 1 (mixed), and
    2 (all-interpolated). Post-propagation gap fills inherit the worse of
    their two bracketing levels, so they too read 0/1/2; there is no distinct
    fill level. The server-side 14/32 Re products use the same 0/1/2 scheme.
    """
    target_x_km = float(target_re) * RE_KM

    x_km = df["X"].astype(np.float64) * RE_KM

    ux_for_timing = df["Ux"].interpolate(method="time")
    ux_orig_nan = df["Ux"].isna()

    travel_s = np.round((x_km.to_numpy() - target_x_km) / np.abs(ux_for_timing.to_numpy()))
    travel = pd.to_timedelta(travel_s, unit="s")
    arrivals = df.index + travel

    finite_mask = np.asarray(arrivals.notna())
    arrivals_finite = arrivals[finite_mask].to_numpy()
    if len(arrivals_finite) == 0:
        valid_finite = np.zeros(0, dtype=bool)
    else:
        running_min = np.minimum.accumulate(arrivals_finite[::-1])[::-1]
        valid_finite = np.empty(len(arrivals_finite), dtype=bool)
        valid_finite[:-1] = arrivals_finite[:-1] <= running_min[1:]
        valid_finite[-1] = True

    valid = np.zeros(len(arrivals), dtype=bool)
    valid[finite_mask] = valid_finite

    numeric = [c for c in _NUMERIC_COLS if c in df.columns]

    # Decompose any interpolation-provenance flags into monotone 0/1 carriers
    # so they survive the numeric regridding; reassembled into <group>_interp
    # columns once the smeared carriers land on the output grid.
    flag_groups = [g for g in _FLAG_GROUPS if f"{g}{_INTERP_SUFFIX}" in df.columns]
    carry_cols: list[str] = []
    src = df
    if flag_groups:
        src = df.copy()
        for g in flag_groups:
            lev = src[f"{g}{_INTERP_SUFFIX}"].fillna(0)
            src[f"{g}{_CARRY_ANY}"] = (lev >= 1).astype(float)
            src[f"{g}{_CARRY_ALL}"] = (lev >= 2).astype(float)
            carry_cols += [f"{g}{_CARRY_ANY}", f"{g}{_CARRY_ALL}"]

    propagated = src.loc[valid, numeric + carry_cols].copy()
    propagated.index = arrivals[valid]

    propagated = propagated[propagated.index.notna()]
    propagated = propagated[~propagated.index.duplicated(keep="first")]
    propagated = propagated.sort_index()

    grid = pd.date_range(df.index.min(), df.index.max(), freq="min")
    merged = propagated.index.union(grid)
    result = (
        propagated.reindex(merged)
        .interpolate(method="index", limit=2)
        .reindex(grid)
    )

    ux_nan_on_grid = ux_orig_nan.reindex(grid, method="nearest", tolerance=pd.Timedelta("30s"))
    result.loc[ux_nan_on_grid.fillna(False).astype(bool), "Ux"] = np.nan

    # Binarize the smeared carriers back into 0/1/2 flag levels, then drop the
    # internal carrier columns from the numeric output.
    for g in flag_groups:
        any_f = result[f"{g}{_CARRY_ANY}"].fillna(0.0) > 0.0
        all_f = (result[f"{g}{_CARRY_ALL}"].fillna(0.0) >= 1.0 - 1e-9) & any_f
        flag = pd.Series(0, index=result.index, dtype="int64")
        flag = flag.mask(any_f, 1)
        flag = flag.mask(all_f, 2)
        # A flag is only meaningful where there is a value: blank it (NaN)
        # where the group's representative column is missing, matching the
        # pipeline (this also promotes the column to float, as in the CSVs).
        repr_col = _FLAG_REPR_COL[g]
        if repr_col in result.columns:
            flag = flag.astype("float64").mask(result[repr_col].isna(), np.nan)
        result[f"{g}{_INTERP_SUFFIX}"] = flag
    if carry_cols:
        result = result.drop(columns=carry_cols)

    return result


def propagate(ds: xr.Dataset, method: str, target_re: float) -> xr.Dataset:
    """Propagate an L1 MIDL dataset to an inner boundary.

    Parameters
    ----------
    ds : xarray.Dataset
        A MIDL L1 dataset as returned by ``midl.load(..., 'l1')``. Must
        contain the per-timestamp source position variable ``X`` (Re) and
        the standard numeric columns (Bx, By, Bz, Ux, Uy, Uz, rho, T).
    method : {'ballistic'}
        Propagation method. Client-side MHD is not supported; use
        ``midl.load(..., 'mhd', target_re=...)`` to fetch server-side
        MHD-propagated data instead.
    target_re : float
        Target boundary distance along the Sun-Earth line, in Earth radii.

    Returns
    -------
    xarray.Dataset
        Propagated dataset on the same 1-minute grid as the input, carrying
        an ``attrs['midl_propagation']`` tag recording the method and target.

    Notes
    -----
    If ``ds`` is already tagged as propagated (``ds.attrs['midl_propagation']``
    is set), a ``UserWarning`` is emitted and propagation proceeds anyway.
    Propagating already-propagated data is almost never what you want;
    download the L1 dataset instead.

    Datasets with Earth's orbital motion included propagate fine: the
    correction is tangential-only, so the Ux values that set the
    per-sample travel times are identical to the convention frame's.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Valid methods: {sorted(METHODS)}")

    coords_system = ds.attrs.get("coords_system", "GSM")
    if coords_system != "GSM":
        raise ValueError(
            f"propagate() requires GSM-frame data, got coords_system="
            f"{coords_system!r}. Ballistic propagation is GSM-only physics: "
            "load with coords='GSM', propagate, then rotate the result."
        )

    if "X" not in ds.data_vars:
        raise ValueError(
            "propagate() requires the L1 dataset (missing 'X' source position). "
            "Call midl.load(..., 'l1') and propagate that, rather than a "
            "pre-propagated 14re/32re download."
        )

    existing = ds.attrs.get("midl_propagation")
    if existing:
        warnings.warn(
            f"Propagating already-propagated data (original "
            f"method={existing.get('method')!r}, "
            f"target_re={existing.get('target_re')!r}). "
            "Download the L1 dataset if you want to propagate yourself.",
            UserWarning,
            stacklevel=2,
        )

    df = ds.to_dataframe()
    result_df = _ballistic_propagate_df(df, target_re)

    result_ds = xr.Dataset.from_dataframe(result_df.rename_axis("time"))
    result_ds.attrs.update(ds.attrs)
    result_ds.attrs["target"] = f"{int(target_re)}Re" if float(target_re).is_integer() else f"{target_re}Re"
    result_ds.attrs["midl_propagation"] = {"method": method, "target_re": float(target_re)}

    for var in result_ds.data_vars:
        if var in ds.data_vars:
            result_ds[var].attrs.update(ds[var].attrs)

    return result_ds


@xr.register_dataset_accessor("midl")
class _MidlAccessor:
    """xarray accessor exposing MIDL helpers on a Dataset.

    Usage::

        ds = midl.load("2024-05-01", "2024-05-02", "l1")
        prop = ds.midl.propagate("ballistic", 14)
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def propagate(self, method: str, target_re: float) -> xr.Dataset:
        return propagate(self._ds, method, target_re)
