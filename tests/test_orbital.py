"""Tests for midl._orbital (Earth orbital-motion correction)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from midl._coords import apply_coords
from midl._orbital import (
    ATTR_INCLUDED,
    ATTR_REMOVED,
    add_orbital_motion,
    earth_orbital_speeds,
)
from midl._propagate import propagate

YEAR_DAILY = pd.date_range("2026-01-01", "2026-12-31", freq="D")


def _ds(times, **overrides):
    """Minimal dataset with the six vector variables plus scalars."""
    n = len(times)
    defaults = {
        "Bx": np.full(n, 1.0), "By": np.full(n, 2.0), "Bz": np.full(n, 3.0),
        "Ux": np.full(n, -400.0), "Uy": np.full(n, 5.0), "Uz": np.full(n, -6.0),
        "rho": np.full(n, 4.0), "T": np.full(n, 1e5),
    }
    defaults.update(overrides)
    data = {k: ("time", np.asarray(v, dtype=float)) for k, v in defaults.items()}
    return xr.Dataset(data, coords={"time": pd.DatetimeIndex(times)})


def _angles(times, psi, mu):
    n = len(times)
    return pd.DataFrame(
        {
            "gsm_gse_angle_deg": np.broadcast_to(np.asarray(psi, dtype=float), n).copy(),
            "dipole_tilt_deg": np.broadcast_to(np.asarray(mu, dtype=float), n).copy(),
        },
        index=pd.DatetimeIndex(times, name="timestamp"),
    )


class TestEarthOrbitalSpeeds:
    def test_tangential_range_and_mean(self):
        _, v_t = earth_orbital_speeds(YEAR_DAILY)
        # Perihelion/aphelion bounds: 30.29 / 29.29 km/s.
        assert 29.28 < v_t.min() < 29.31
        assert 30.26 < v_t.max() < 30.30
        # Time-mean of v_t is n*a*sqrt(1-e^2) = 29.781 km/s.
        assert abs(v_t.mean() - 29.78) < 0.02

    def test_radial_range_and_mean(self):
        v_r, _ = earth_orbital_speeds(YEAR_DAILY)
        # Eccentricity term peaks at n*a*e ~ 0.50 km/s; time-mean is 0.
        assert 0.48 < v_r.max() < 0.51
        assert -0.51 < v_r.min() < -0.48
        assert abs(v_r.mean()) < 0.01

    def test_perihelion_timing(self):
        # Fastest tangential speed (perihelion) falls in early January;
        # slowest (aphelion) in early July.
        _, v_t = earth_orbital_speeds(YEAR_DAILY)
        assert YEAR_DAILY[v_t.argmax()].month == 1
        assert YEAR_DAILY[v_t.argmin()].month == 7

    def test_radial_sign_receding_in_spring(self):
        # Between perihelion (Jan) and aphelion (Jul) Earth recedes from
        # the Sun: v_r > 0, peaking near April.
        v_r, _ = earth_orbital_speeds(pd.DatetimeIndex(["2026-04-04"]))
        assert v_r[0] > 0.45

    def test_j2000_epoch_finite(self):
        v_r, v_t = earth_orbital_speeds(pd.DatetimeIndex(["2000-01-01 12:00"]))
        assert np.isfinite(v_r[0]) and np.isfinite(v_t[0])


TIMES2 = pd.date_range("2024-03-01 00:00", periods=2, freq="min")


class TestAddOrbitalMotion:
    def test_gse_adds_vt_to_uy_and_vr_to_ux(self):
        ds = _ds(TIMES2)
        v_r, v_t = earth_orbital_speeds(TIMES2)
        out = add_orbital_motion(ds, "GSE", None)
        np.testing.assert_allclose(out["Ux"].values, ds["Ux"].values + v_r, atol=1e-12)
        np.testing.assert_allclose(out["Uy"].values, ds["Uy"].values + v_t, atol=1e-12)
        np.testing.assert_allclose(out["Uz"].values, ds["Uz"].values, atol=1e-12)

    def test_gsm_90deg_moves_correction_to_uz(self):
        # psi = 90: dUy = cos(psi)*v_t = 0, dUz = -sin(psi)*v_t = -v_t.
        ds = _ds(TIMES2)
        v_r, v_t = earth_orbital_speeds(TIMES2)
        out = add_orbital_motion(ds, "GSM", _angles(TIMES2, psi=90.0, mu=0.0))
        np.testing.assert_allclose(out["Ux"].values, ds["Ux"].values + v_r, atol=1e-9)
        np.testing.assert_allclose(out["Uy"].values, ds["Uy"].values, atol=1e-9)
        np.testing.assert_allclose(out["Uz"].values, ds["Uz"].values - v_t, atol=1e-9)

    def test_gsm_matches_rotated_gse_correction(self):
        # Rx(psi) @ (v_r, v_t, 0) for a generic angle.
        psi_deg = 23.0
        ds = _ds(TIMES2)
        v_r, v_t = earth_orbital_speeds(TIMES2)
        out = add_orbital_motion(ds, "GSM", _angles(TIMES2, psi=psi_deg, mu=0.0))
        psi = np.radians(psi_deg)
        np.testing.assert_allclose(
            out["Uy"].values, ds["Uy"].values + np.cos(psi) * v_t, atol=1e-9)
        np.testing.assert_allclose(
            out["Uz"].values, ds["Uz"].values - np.sin(psi) * v_t, atol=1e-9)

    def test_sm_chains_tilt(self):
        # mu = 90, psi = 0: dU_GSM = (v_r, v_t, 0); Ry(90) -> (0, v_t, v_r).
        ds = _ds(TIMES2)
        v_r, v_t = earth_orbital_speeds(TIMES2)
        out = add_orbital_motion(ds, "SM", _angles(TIMES2, psi=0.0, mu=90.0))
        np.testing.assert_allclose(out["Ux"].values, ds["Ux"].values, atol=1e-9)
        np.testing.assert_allclose(out["Uy"].values, ds["Uy"].values + v_t, atol=1e-9)
        np.testing.assert_allclose(out["Uz"].values, ds["Uz"].values + v_r, atol=1e-9)

    def test_b_and_scalars_untouched(self):
        ds = _ds(TIMES2)
        out = add_orbital_motion(ds, "GSE", None)
        for var in ("Bx", "By", "Bz", "rho", "T"):
            np.testing.assert_array_equal(out[var].values, ds[var].values)

    def test_missing_angle_row_nans_velocities_only(self):
        times = pd.date_range("2024-03-01 00:00", periods=3, freq="min")
        ds = _ds(times)
        out = add_orbital_motion(ds, "GSM", _angles([times[0], times[2]], 10.0, 20.0))
        for var in ("Ux", "Uy", "Uz"):
            assert np.isnan(out[var].values[1])
            assert np.isfinite(out[var].values[0])
            assert np.isfinite(out[var].values[2])
        np.testing.assert_array_equal(out["Bx"].values, ds["Bx"].values)

    def test_gsm_without_angles_raises(self):
        with pytest.raises(ValueError, match="angle table"):
            add_orbital_motion(_ds(TIMES2), "GSM", None)


class TestApplyCoordsWiring:
    def test_default_stamps_removed_and_skips_angles(self):
        with patch("midl._coords.load_angles") as mock_la:
            out = apply_coords(
                _ds(TIMES2), "GSM", TIMES2[0], TIMES2[-1])
        mock_la.assert_not_called()
        assert out.attrs["orbital_motion"] == ATTR_REMOVED

    def test_gsm_orbital_fetches_angles_and_stamps_included(self):
        with patch(
            "midl._coords.load_angles",
            return_value=_angles(TIMES2, psi=0.0, mu=0.0),
        ) as mock_la:
            out = apply_coords(
                _ds(TIMES2), "GSM", TIMES2[0], TIMES2[-1], orbital_motion=True)
        mock_la.assert_called_once()
        assert out.attrs["orbital_motion"] == ATTR_INCLUDED
        assert out["Uy"].attrs["orbital_motion"] == ATTR_INCLUDED
        # psi = 0: full tangential speed lands in Uy.
        _, v_t = earth_orbital_speeds(TIMES2)
        np.testing.assert_allclose(out["Uy"].values, 5.0 + v_t, atol=1e-9)

    def test_gse_orbital_equals_rotate_then_correct(self):
        angles = _angles(TIMES2, psi=17.0, mu=0.0)
        with patch("midl._coords.load_angles", return_value=angles):
            out = apply_coords(
                _ds(TIMES2), "GSE", TIMES2[0], TIMES2[-1], orbital_motion=True)
        from midl._coords import rotate_dataset
        expected = add_orbital_motion(
            rotate_dataset(_ds(TIMES2), "GSE", angles), "GSE", None)
        for var in ("Ux", "Uy", "Uz"):
            np.testing.assert_allclose(
                out[var].values, expected[var].values, atol=1e-12)


class TestPropagateGuard:
    def test_orbital_motion_dataset_refused(self):
        ds = _ds(TIMES2)
        ds.attrs["coords_system"] = "GSM"
        ds.attrs["orbital_motion"] = "included"
        with pytest.raises(ValueError, match="orbital motion"):
            propagate(ds, "ballistic", 20)
