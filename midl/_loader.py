"""Load MIDL data from the web host into xarray Datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from midl._cache import ensure_cached, resolve_target
from midl._time import Timelike, months_in_range, parse_timestamp

_VAR_ATTRS: dict[str, dict[str, str]] = {
    "Bx": {"units": "nT", "long_name": "Magnetic field X component", "coordinate_system": "GSM"},
    "By": {"units": "nT", "long_name": "Magnetic field Y component", "coordinate_system": "GSM"},
    "Bz": {"units": "nT", "long_name": "Magnetic field Z component", "coordinate_system": "GSM"},
    "Ux": {"units": "km/s", "long_name": "Bulk velocity X component", "coordinate_system": "GSM"},
    "Uy": {"units": "km/s", "long_name": "Bulk velocity Y component", "coordinate_system": "GSM"},
    "Uz": {"units": "km/s", "long_name": "Bulk velocity Z component", "coordinate_system": "GSM"},
    "rho": {"units": "cm^-3", "long_name": "Proton number density"},
    "T": {"units": "K", "long_name": "Proton temperature"},
    "X": {"units": "Re", "long_name": "Reference satellite X_GSM position", "coordinate_system": "GSM"},
    "B_source": {"long_name": "Magnetic field source satellite(s)"},
    "Ux_source": {"long_name": "Ux source satellite(s)"},
    "Uyz_source": {"long_name": "Uy/Uz source satellite(s)"},
    "rho_source": {"long_name": "Density source satellite(s)"},
    "T_source": {"long_name": "Temperature source satellite(s)"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a single cached MIDL CSV into a DataFrame."""
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def _to_dataset(df: pd.DataFrame, target: str) -> xr.Dataset:
    """Convert a time-indexed DataFrame to an xarray Dataset with metadata."""
    ds = df.to_xarray().rename({"timestamp": "time"})
    ds.attrs["source"] = "MIDL"
    ds.attrs["url"] = "https://csem.engin.umich.edu/MIDL/"
    ds.attrs["target"] = target
    for var, attrs in _VAR_ATTRS.items():
        if var in ds:
            ds[var].attrs.update(attrs)
    return ds


def load(start: Timelike, end: Timelike, target: str) -> xr.Dataset:
    """Load MIDL solar wind data for a time range.

    Downloads monthly CSV files from the MIDL web host on first access
    and caches them locally. Subsequent calls for the same months are
    served from cache.

    Parameters
    ----------
    start, end : str, datetime, numpy.datetime64, or pandas.Timestamp
        Time range bounds (inclusive). Accepts ISO 8601 strings down to
        the minute, e.g. ``"2015-03-17"`` or ``"2015-03-17 14:30"``.
    target : str
        Propagation target: ``"14re"``, ``"32re"``, or ``"l1"``
        (case-insensitive).

    Returns
    -------
    xarray.Dataset
        Dataset with a ``time`` coordinate and data variables
        (Bx, By, Bz, Ux, Uy, Uz, rho, T, plus extras for L1).
    """
    start_ts = parse_timestamp(start)
    end_ts = parse_timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts}) must be <= end ({end_ts})")

    canonical = resolve_target(target)
    months = months_in_range(start_ts, end_ts)

    frames: list[pd.DataFrame] = []
    for ym in months:
        path = ensure_cached(ym, canonical)
        frames.append(_read_csv(path))

    df = pd.concat(frames).sort_index()
    df = df.loc[start_ts:end_ts]

    return _to_dataset(df, canonical)
