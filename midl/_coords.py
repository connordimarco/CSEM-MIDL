"""Coordinate-frame rotation of MIDL vector variables.

MIDL data is produced in GSM. GSE and SM output frames are derived
client-side by rotating the vector variables (Bx, By, Bz, Ux, Uy, Uz)
per minute using the server's published monthly angle tables
``data/YYYY/MM/YYYYMM_angles.csv`` (columns: ``timestamp``,
``gsm_gse_angle_deg``, ``dipole_tilt_deg``; one row per minute, naive
UTC timestamps on the same 1-minute grid as every other MIDL product).

Sign conventions (pinned; ground truth is the pipeline's l1_angles.py):

    v_GSM = Rx(psi) @ v_GSE         Rx(a) = [[1,   0,     0   ],
                                             [0,  cos a,  sin a],
                                             [0, -sin a,  cos a]]

so the client applies the inverse, v_GSE = Rx(-psi) @ v_GSM:

    Bx_gse = Bx_gsm
    By_gse =  cos(psi)*By_gsm - sin(psi)*Bz_gsm
    Bz_gse =  sin(psi)*By_gsm + cos(psi)*Bz_gsm

    v_SM = Ry(mu) @ v_GSM           Ry(a) = [[cos a, 0, -sin a],
                                             [  0,   1,    0  ],
                                             [sin a, 0,  cos a]]

    Bx_sm =  cos(mu)*Bx_gsm - sin(mu)*Bz_gsm
    By_sm =  By_gsm
    Bz_sm =  sin(mu)*Bx_gsm + cos(mu)*Bz_gsm

where psi is ``gsm_gse_angle_deg`` and mu is ``dipole_tilt_deg``. SM is
derived directly from GSM (not chained through GSE). The ``X`` variable
(reference satellite X_GSM position) is propagation metadata, never
rotated. Components are rotated in each minute's instantaneous frame
orientation (standard convention, same as OMNI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import xarray as xr

from midl._cache import ensure_cached
from midl._orbital import ATTR_INCLUDED, ATTR_REMOVED, add_orbital_motion
from midl._time import months_in_range

VALID_COORDS = ("GSM", "GSE", "SM")

_ANGLES_TARGET = "angles"
_PSI_COL = "gsm_gse_angle_deg"
_TILT_COL = "dipole_tilt_deg"

# Vector variable triples rotated between frames. X is deliberately absent
# (propagation metadata, stays GSM); rho, T, sources, and flags are scalars.
VECTOR_TRIPLES = (("Bx", "By", "Bz"), ("Ux", "Uy", "Uz"))


def validate_coords(coords: str) -> str:
    """Validate a ``coords`` argument and normalize it to upper case."""
    if isinstance(coords, str) and coords.upper() in VALID_COORDS:
        return coords.upper()
    raise ValueError(
        f"Unknown coords {coords!r}. Valid coords: 'GSM', 'GSE', 'SM'"
    )


def load_angles(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Load the per-minute rotation-angle table covering ``[start_ts, end_ts]``.

    Downloads (and caches) the monthly ``YYYYMM_angles.csv`` files via the
    same cache layer as the data CSVs. Raises a clear ``ValueError`` when a
    month has no published angle file.
    """
    frames = []
    for ym in months_in_range(start_ts, end_ts):
        try:
            path = ensure_cached(ym, _ANGLES_TARGET)
        except requests.HTTPError as exc:
            raise ValueError(
                f"No angle file available for {ym} "
                f"({ym.replace('-', '')}_angles.csv). GSE/SM output and the "
                "orbital_motion correction require the server's monthly "
                "angle tables, which may not be published for this month. "
                "Use coords='GSM' (the native frame) without orbital_motion "
                "instead."
            ) from exc
        frames.append(
            pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        )
    return pd.concat(frames).sort_index()


def rotate_dataset(ds: xr.Dataset, coords: str, angles: pd.DataFrame) -> xr.Dataset:
    """Rotate a GSM dataset's vector variables into ``coords``.

    Angle rows are exact-matched to data timestamps (both live on 1-minute
    grids). Any data timestamp missing from the angle table gets all six
    vector values set to NaN (self-reporting); scalar variables are never
    touched. ``coords='GSM'`` returns ``ds`` unchanged.
    """
    coords = validate_coords(coords)
    if coords == "GSM":
        return ds

    times = pd.DatetimeIndex(ds["time"].values)
    aligned = angles.reindex(times)
    col = _PSI_COL if coords == "GSE" else _TILT_COL
    a = np.radians(aligned[col].to_numpy(dtype=np.float64))
    missing = ~np.isfinite(a)
    cos_a, sin_a = np.cos(a), np.sin(a)

    out = ds.copy()
    for vx, vy, vz in VECTOR_TRIPLES:
        if not all(v in ds.data_vars for v in (vx, vy, vz)):
            continue
        x = ds[vx].to_numpy().astype(np.float64)
        y = ds[vy].to_numpy().astype(np.float64)
        z = ds[vz].to_numpy().astype(np.float64)
        if coords == "GSE":
            # v_GSE = Rx(-psi) @ v_GSM
            new_x = x
            new_y = cos_a * y - sin_a * z
            new_z = sin_a * y + cos_a * z
        else:  # SM: v_SM = Ry(mu) @ v_GSM, directly from GSM
            new_x = cos_a * x - sin_a * z
            new_y = y
            new_z = sin_a * x + cos_a * z
        for var, vals in ((vx, new_x), (vy, new_y), (vz, new_z)):
            vals = np.where(missing, np.nan, vals)
            out[var] = ds[var].copy(data=vals)
    return out


def apply_coords(
    ds: xr.Dataset,
    coords: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    orbital_motion: bool = False,
) -> xr.Dataset:
    """Rotate a native-GSM ``load()`` result into ``coords`` and stamp metadata.

    ``start_ts``/``end_ts`` bound the final (post-slice) time range and
    determine which monthly angle files are fetched. Sets
    ``ds.attrs['coords_system']`` on every path (GSM included) and the
    per-variable ``coordinate_system`` attr on the six vector variables.
    ``X`` keeps ``coordinate_system: GSM`` — it is propagation metadata
    (reference satellite X_GSM), never rotated.

    ``orbital_motion=True`` additionally adds Earth's orbital velocity to
    (Ux, Uy, Uz) after the rotation (see midl._orbital), stamping
    ``ds.attrs['orbital_motion']``; the correction in GSM/SM output needs
    the same monthly angle tables as the rotations.
    """
    coords = validate_coords(coords)
    # GSE-frame data rotation and any GSM/SM orbital correction consume the
    # angle tables; only plain native-GSM output skips the fetch.
    angles = None
    if coords != "GSM" or orbital_motion:
        angles = load_angles(start_ts, end_ts)
    if coords != "GSM":
        ds = rotate_dataset(ds, coords, angles)
    if orbital_motion:
        ds = add_orbital_motion(ds, coords, angles)
    ds.attrs["coords_system"] = coords
    ds.attrs["orbital_motion"] = ATTR_INCLUDED if orbital_motion else ATTR_REMOVED
    for triple in VECTOR_TRIPLES:
        for var in triple:
            if var in ds.data_vars:
                ds[var].attrs["coordinate_system"] = coords
    return ds
