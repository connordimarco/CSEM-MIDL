"""Earth orbital-motion correction for MIDL velocity variables.

Like CDAWeb, OMNI, and the underlying instrument teams' L2 products, MIDL
velocities follow the community convention in which Earth's orbital motion
around the Sun has been removed: the frame axes are GSE/GSM but the frame
origin does not co-move with Earth, so Vy_GSE averages ~0 instead of the
~+29.78 km/s aberration a truly Earth-co-moving frame would show
(established by Yingjuan Ma, UCLA, 2026). ``load(..., orbital_motion=True)``
undoes that removal, returning velocities in Earth's rest frame:

    u_earth = u_convention - v_Earth

Earth's heliocentric velocity expressed on GSE axes is

    v_Earth = (-v_r, -v_t, 0)        X_GSE = Earth->Sun = -r_hat
                                     Y_GSE = -theta_hat (duskward)

with v_r the radial speed (positive receding from the Sun, +-0.50 km/s
over the year) and v_t the tangential speed (29.29 km/s at aphelion to
30.29 km/s at perihelion, time-mean 29.78 km/s), so the correction *adds*
``(+v_r, +v_t, 0)`` to (Ux, Uy, Uz) in GSE. For GSM/SM output the same
GSE vector is rotated through the published per-minute angle tables
(see midl._coords for the pinned sign conventions):

    GSM: dU = ( v_r,  cos(psi)*v_t, -sin(psi)*v_t )         Rx(psi)
    SM : dU = Ry(mu) @ dU_GSM

Both speeds come from a closed-form two-body (Kepler) solution of Earth's
orbit — mean anomaly from the standard solar-ephemeris linear fit, Kepler's
equation by Newton iteration, then

    v_r = n*a * e*sin(E) / (1 - e*cos(E))
    v_t = n*a * sqrt(1 - e^2) / (1 - e*cos(E))

Accuracy vs. a full ephemeris is ~0.02 km/s (dominated by the neglected
Earth-Moon barycenter wobble, ~0.013 km/s, and planetary perturbations),
far below solar wind instrument accuracy. Only (Ux, Uy, Uz) are corrected;
B, rho, T, and provenance columns are frame-origin-independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# Two-body constants for Earth's orbit (J2000 osculating values; the slow
# secular drifts are far below the ~0.02 km/s accuracy target).
_AU_KM = 1.495978707e8
_SIDEREAL_YEAR_S = 365.25636 * 86400.0
_NA_KMS = 2.0 * np.pi * _AU_KM / _SIDEREAL_YEAR_S  # n*a = 29.785 km/s
_ECC = 0.016709

# Sun's mean anomaly, degrees: M = _M0 + _MDOT * (days since J2000.0).
# Standard low-precision solar ephemeris fit (Astronomical Almanac / NOAA).
_M0_DEG = 357.52911
_MDOT_DEG_PER_DAY = 0.98560028
_J2000 = pd.Timestamp("2000-01-01T12:00:00")

# Correction column names + attr values shared with _loader/_coords.
VELOCITY_VARS = ("Ux", "Uy", "Uz")
ATTR_INCLUDED = "included"
ATTR_REMOVED = "removed (community convention)"


def earth_orbital_speeds(times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Earth's heliocentric (radial, tangential) speed in km/s at ``times``.

    Radial ``v_r`` is positive when Earth recedes from the Sun (positive
    from perihelion in early January to aphelion in early July, peaking
    ~0.50 km/s near April); tangential ``v_t`` spans 29.29-30.29 km/s.
    """
    d = (times - _J2000) / pd.Timedelta(days=1)
    m = np.radians(_M0_DEG + _MDOT_DEG_PER_DAY * np.asarray(d, dtype=np.float64))
    # Kepler's equation E - e*sin(E) = M by Newton iteration; e is tiny so
    # three iterations from E=M converge to machine precision.
    ecc_anom = m.copy()
    for _ in range(3):
        ecc_anom -= (ecc_anom - _ECC * np.sin(ecc_anom) - m) / (
            1.0 - _ECC * np.cos(ecc_anom)
        )
    denom = 1.0 - _ECC * np.cos(ecc_anom)
    v_r = _NA_KMS * _ECC * np.sin(ecc_anom) / denom
    v_t = _NA_KMS * np.sqrt(1.0 - _ECC**2) / denom
    return v_r, v_t


def add_orbital_motion(
    ds: xr.Dataset, coords: str, angles: pd.DataFrame | None
) -> xr.Dataset:
    """Add Earth's orbital motion to (Ux, Uy, Uz), expressed in ``coords``.

    ``ds`` must already be rotated into ``coords``. ``angles`` is the
    per-minute angle table (required for GSM and SM output, unused for
    GSE). Timestamps missing from the angle table get NaN velocity
    components (self-reporting, matching rotate_dataset); B and scalars
    are never touched.
    """
    times = pd.DatetimeIndex(ds["time"].values)
    v_r, v_t = earth_orbital_speeds(times)

    if coords == "GSE":
        du_x, du_y, du_z = v_r, v_t, np.zeros_like(v_t)
        missing = np.zeros(len(times), dtype=bool)
    else:
        if angles is None:
            raise ValueError(
                f"add_orbital_motion() needs the angle table for coords={coords!r}"
            )
        aligned = angles.reindex(times)
        psi = np.radians(aligned["gsm_gse_angle_deg"].to_numpy(dtype=np.float64))
        # v_GSM = Rx(psi) @ (v_r, v_t, 0)
        du_x = v_r
        du_y = np.cos(psi) * v_t
        du_z = -np.sin(psi) * v_t
        missing = ~np.isfinite(psi)
        if coords == "SM":
            mu = np.radians(aligned["dipole_tilt_deg"].to_numpy(dtype=np.float64))
            du_x, du_z = (
                np.cos(mu) * du_x - np.sin(mu) * du_z,
                np.sin(mu) * du_x + np.cos(mu) * du_z,
            )
            missing |= ~np.isfinite(mu)

    out = ds.copy()
    for var, du in zip(VELOCITY_VARS, (du_x, du_y, du_z)):
        if var not in ds.data_vars:
            continue
        vals = ds[var].to_numpy().astype(np.float64) + du
        vals = np.where(missing, np.nan, vals)
        out[var] = ds[var].copy(data=vals)
        out[var].attrs["orbital_motion"] = ATTR_INCLUDED
    return out
