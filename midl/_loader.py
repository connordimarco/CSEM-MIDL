"""Load MIDL data from the web host into xarray Datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from midl._cache import canonical_mhd, ensure_cached
from midl._coords import apply_coords, validate_coords
from midl._propagate import propagate
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

_FIXED_BALLISTIC_RE: frozenset[int] = frozenset({14, 32})

# Padding added to the leading edge of the L1 window when client-side
# ballistic propagation is requested, so the propagated output is filled
# across the user's requested range instead of starting NaN.
_BALLISTIC_LEAD_PADDING = pd.Timedelta(hours=2)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a single cached MIDL CSV into a DataFrame."""
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def _load_monthly(start_ts: pd.Timestamp, end_ts: pd.Timestamp, canonical: str) -> pd.DataFrame:
    """Read all monthly CSVs spanning ``[start_ts, end_ts]`` and slice."""
    frames = [_read_csv(ensure_cached(ym, canonical)) for ym in months_in_range(start_ts, end_ts)]
    df = pd.concat(frames).sort_index()
    return df.loc[start_ts:end_ts]


def _to_dataset(df: pd.DataFrame, target: str) -> xr.Dataset:
    """Convert a time-indexed DataFrame to an xarray Dataset with metadata."""
    ds = df.to_xarray().rename({"timestamp": "time"})
    ds.attrs["source"] = "MIDL"
    ds.attrs["url"] = "https://csem.engin.umich.edu/MIDL/"
    if target == "L1":
        ds.attrs["target"] = target
        ds.attrs["midl_propagation"] = None
    elif target.startswith("mhd_"):
        target_re = float(int(target.removeprefix("mhd_").removesuffix("Re")))
        ds.attrs["target"] = f"{int(target_re)}Re"
        ds.attrs["midl_propagation"] = {"method": "mhd", "target_re": target_re}
    else:
        target_re = float(target.removesuffix("Re"))
        ds.attrs["target"] = target
        ds.attrs["midl_propagation"] = {"method": "ballistic", "target_re": target_re}
    for var, attrs in _VAR_ATTRS.items():
        if var in ds:
            ds[var].attrs.update(attrs)
    return ds


def load(
    start: Timelike,
    end: Timelike,
    target_re: float | int | str,
    method: str = "ballistic",
    coords: str = "GSM",
    orbital_motion: bool = True,
) -> xr.Dataset:
    """Load MIDL solar wind data for a time range.

    Downloads monthly CSV files from the MIDL web host on first access
    and caches them locally. Subsequent calls for the same months are
    served from cache.

    Parameters
    ----------
    start, end : str, datetime, numpy.datetime64, or pandas.Timestamp
        Time range bounds (inclusive). Accepts ISO 8601 strings down to
        the minute, e.g. ``"2015-03-17"`` or ``"2015-03-17 14:30"``.
    target_re : int, float, or ``"l1"``
        Location at which to return solar wind values:

        - A number in Earth radii along the Sun-Earth line.
        - The string ``"l1"`` (case-insensitive) to return unpropagated
          L1 observations.
    method : {"ballistic", "mhd"}, default ``"ballistic"``
        Propagation method.

        - ``"ballistic"`` — pre-propagated server files are used for
          ``target_re`` 14 or 32; any other numeric ``target_re`` loads
          L1 and runs client-side ballistic propagation. ``"l1"`` is
          only valid with this method and returns the raw L1 dataset.
        - ``"mhd"`` — server-side 1D MHD propagation. ``target_re`` must
          be an integer in ``[-70, 70]``.
    coords : {"GSM", "GSE", "SM"}, default ``"GSM"``
        Output coordinate system (case-insensitive). MIDL data is native
        GSM; ``"GSE"`` and ``"SM"`` rotate the vector variables
        (Bx, By, Bz, Ux, Uy, Uz) per minute using the server's monthly
        angle files, with psi = ``gsm_gse_angle_deg`` and
        mu = ``dipole_tilt_deg``::

            By_gse = cos(psi)*By_gsm - sin(psi)*Bz_gsm   # Bx unchanged
            Bz_gse = sin(psi)*By_gsm + cos(psi)*Bz_gsm

            Bx_sm  = cos(mu)*Bx_gsm - sin(mu)*Bz_gsm     # By unchanged
            Bz_sm  = sin(mu)*Bx_gsm + cos(mu)*Bz_gsm

        Components are rotated in each minute's instantaneous frame
        orientation (standard convention, same as OMNI). Scalars (rho,
        T) and provenance columns are untouched, and the L1 ``X``
        variable always stays the reference satellite X_GSM position.
        Timestamps missing from the angle table get NaN vector values.
    orbital_motion : bool, default ``True``
        The underlying data follow the community convention (CDAWeb,
        OMNI) in which Earth's orbital motion around the Sun has been
        removed from the velocities, so Vy_GSE averages ~0 rather than
        the ~+29.78 km/s an Earth-co-moving frame would show. The
        default restores it, returning velocities in Earth's rest frame
        — the physically correct upstream input for Earth-centered
        magnetosphere models such as SWMF: Earth's tangential orbital
        speed (seasonally exact via a two-body solution,
        29.29-30.29 km/s) is added along +Y_GSE, expressed in the
        requested ``coords`` frame (for GSM/SM via the same per-minute
        angle tables as ``coords``, so default GSM loads now also fetch
        the cached angle files). Ux is left as-is: the convention is
        only known to remove the tangential component, so the ±0.50 km/s
        radial part is not re-added.
        The choice is stamped in ``ds.attrs['orbital_motion']`` and in
        the ``to_dat`` header. B, rho, and T are frame-origin-
        independent and unaffected. Pass ``False`` for convention-frame
        (CDAWeb/OMNI-comparable) velocities.

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

    if not isinstance(method, str):
        raise ValueError(f"method must be a string, got {type(method).__name__}")
    method_key = method.lower()
    if method_key not in ("ballistic", "mhd"):
        raise ValueError(
            f"Unknown method {method!r}. Valid methods: 'ballistic', 'mhd'"
        )

    coords_key = validate_coords(coords)

    if not isinstance(orbital_motion, bool):
        raise ValueError(
            f"orbital_motion must be a bool, got {type(orbital_motion).__name__}"
        )

    if isinstance(target_re, str):
        if target_re.lower() != "l1":
            raise ValueError(
                f"target_re must be a number or 'l1', got {target_re!r}"
            )
        if method_key != "ballistic":
            raise ValueError(
                "target_re='l1' is only valid with method='ballistic'"
            )
        df = _load_monthly(start_ts, end_ts, "L1")
        return apply_coords(
            _to_dataset(df, "L1"), coords_key, start_ts, end_ts, orbital_motion)

    if not isinstance(target_re, (int, float)) or isinstance(target_re, bool):
        raise ValueError(
            f"target_re must be a number or 'l1', got {type(target_re).__name__}"
        )

    if method_key == "mhd":
        canonical = canonical_mhd(target_re)
        df = _load_monthly(start_ts, end_ts, canonical)
        return apply_coords(
            _to_dataset(df, canonical), coords_key, start_ts, end_ts, orbital_motion)

    # method == "ballistic"
    re_float = float(target_re)
    if re_float.is_integer() and int(re_float) in _FIXED_BALLISTIC_RE:
        canonical = f"{int(re_float)}Re"
        df = _load_monthly(start_ts, end_ts, canonical)
        return apply_coords(
            _to_dataset(df, canonical), coords_key, start_ts, end_ts, orbital_motion)

    # Custom ballistic target — load L1 (with leading padding) and
    # propagate client-side. Propagation runs on native-GSM data; the
    # frame rotation is applied after the final time slice (and the
    # angle months come from that final slice, not the padded range).
    l1_start = start_ts - _BALLISTIC_LEAD_PADDING
    l1_df = _load_monthly(l1_start, end_ts, "L1")
    l1_ds = _to_dataset(l1_df, "L1")
    propagated = propagate(l1_ds, "ballistic", re_float)
    sliced = propagated.sel(time=slice(start_ts, end_ts))
    return apply_coords(sliced, coords_key, start_ts, end_ts, orbital_motion)
