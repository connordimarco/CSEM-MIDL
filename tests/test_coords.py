"""Tests for midl._coords (GSM -> GSE/SM frame rotation)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from midl._coords import (
    VALID_COORDS,
    load_angles,
    rotate_dataset,
    validate_coords,
)

DATA_DIR = Path(__file__).parent / "data"


def _ds(times, Bx, By, Bz, Ux=None, Uy=None, Uz=None, rho=None, T=None):
    """Build a minimal GSM dataset on the given times."""
    n = len(times)
    data = {
        "Bx": ("time", np.asarray(Bx, dtype=float)),
        "By": ("time", np.asarray(By, dtype=float)),
        "Bz": ("time", np.asarray(Bz, dtype=float)),
        "Ux": ("time", np.asarray(Ux if Ux is not None else np.zeros(n), dtype=float)),
        "Uy": ("time", np.asarray(Uy if Uy is not None else np.zeros(n), dtype=float)),
        "Uz": ("time", np.asarray(Uz if Uz is not None else np.zeros(n), dtype=float)),
        "rho": ("time", np.asarray(rho if rho is not None else np.full(n, 5.0), dtype=float)),
        "T": ("time", np.asarray(T if T is not None else np.full(n, 1e5), dtype=float)),
    }
    return xr.Dataset(data, coords={"time": pd.DatetimeIndex(times)})


def _angles(times, psi, mu):
    """Angle table indexed by timestamp, broadcasting scalar angles."""
    n = len(times)
    return pd.DataFrame(
        {
            "gsm_gse_angle_deg": np.broadcast_to(np.asarray(psi, dtype=float), n).copy(),
            "dipole_tilt_deg": np.broadcast_to(np.asarray(mu, dtype=float), n).copy(),
        },
        index=pd.DatetimeIndex(times, name="timestamp"),
    )


TIMES2 = pd.date_range("2024-03-01 00:00", periods=2, freq="min")


class TestValidateCoords:
    def test_normalizes_case(self):
        assert validate_coords("gse") == "GSE"
        assert validate_coords("Sm") == "SM"
        assert validate_coords("GSM") == "GSM"

    def test_valid_coords_tuple(self):
        assert VALID_COORDS == ("GSM", "GSE", "SM")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="'GSM', 'GSE', 'SM'"):
            validate_coords("GEI")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Valid coords"):
            validate_coords(42)


class TestGsmIdentity:
    def test_gsm_returns_dataset_unchanged(self):
        ds = _ds(TIMES2, Bx=[1.0, 2.0], By=[3.0, 4.0], Bz=[5.0, 6.0])
        angles = _angles(TIMES2, psi=90.0, mu=90.0)
        out = rotate_dataset(ds, "GSM", angles)
        xr.testing.assert_identical(out, ds)


class TestKnownAngles:
    """90-degree rotations pin the sign conventions exactly."""

    def test_gse_90deg(self):
        # psi = 90: By_gse = cos*By - sin*Bz = -Bz ; Bz_gse = sin*By + cos*Bz = By
        ds = _ds(TIMES2, Bx=[1.0, 1.0], By=[2.0, 2.0], Bz=[3.0, 3.0],
                 Ux=[-400.0, -400.0], Uy=[10.0, 10.0], Uz=[20.0, 20.0])
        out = rotate_dataset(ds, "GSE", _angles(TIMES2, psi=90.0, mu=0.0))
        np.testing.assert_allclose(out["Bx"].values, [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(out["By"].values, [-3.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(out["Bz"].values, [2.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(out["Ux"].values, [-400.0, -400.0], atol=1e-12)
        np.testing.assert_allclose(out["Uy"].values, [-20.0, -20.0], atol=1e-12)
        np.testing.assert_allclose(out["Uz"].values, [10.0, 10.0], atol=1e-12)

    def test_sm_90deg(self):
        # mu = 90: Bx_sm = cos*Bx - sin*Bz = -Bz ; Bz_sm = sin*Bx + cos*Bz = Bx
        ds = _ds(TIMES2, Bx=[1.0, 1.0], By=[2.0, 2.0], Bz=[3.0, 3.0])
        out = rotate_dataset(ds, "SM", _angles(TIMES2, psi=0.0, mu=90.0))
        np.testing.assert_allclose(out["Bx"].values, [-3.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(out["By"].values, [2.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(out["Bz"].values, [1.0, 1.0], atol=1e-12)

    def test_scalars_untouched(self):
        ds = _ds(TIMES2, Bx=[1.0, 1.0], By=[2.0, 2.0], Bz=[3.0, 3.0],
                 rho=[4.5, 4.6], T=[8e4, 9e4])
        out = rotate_dataset(ds, "GSE", _angles(TIMES2, psi=90.0, mu=90.0))
        np.testing.assert_array_equal(out["rho"].values, ds["rho"].values)
        np.testing.assert_array_equal(out["T"].values, ds["T"].values)


class TestRoundTrip:
    def test_gse_round_trip_restores_gsm(self):
        # Rx(-psi) then Rx(psi) (== rotating the GSE result with -psi) is identity.
        rng = np.random.default_rng(42)
        n = 5
        times = pd.date_range("2024-03-01", periods=n, freq="min")
        ds = _ds(times, Bx=rng.normal(size=n), By=rng.normal(size=n),
                 Bz=rng.normal(size=n), Ux=rng.normal(size=n),
                 Uy=rng.normal(size=n), Uz=rng.normal(size=n))
        psi = np.linspace(-30.0, 25.0, n)
        gse = rotate_dataset(ds, "GSE", _angles(times, psi=psi, mu=0.0))
        back = rotate_dataset(gse, "GSE", _angles(times, psi=-psi, mu=0.0))
        for var in ("Bx", "By", "Bz", "Ux", "Uy", "Uz"):
            np.testing.assert_allclose(back[var].values, ds[var].values, atol=1e-12)


class TestSmDirectFromGsm:
    def test_sm_ignores_psi_entirely(self):
        # With psi nonzero, an implementation chained through GSE would
        # change By; the direct Ry(mu) path leaves By identical to GSM.
        ds = _ds(TIMES2, Bx=[1.5, -0.5], By=[2.5, 3.5], Bz=[-1.0, 4.0])
        angles = _angles(TIMES2, psi=37.0, mu=25.0)
        out = rotate_dataset(ds, "SM", angles)
        np.testing.assert_array_equal(out["By"].values, ds["By"].values)
        mu = np.radians(25.0)
        exp_x = np.cos(mu) * ds["Bx"].values - np.sin(mu) * ds["Bz"].values
        exp_z = np.sin(mu) * ds["Bx"].values + np.cos(mu) * ds["Bz"].values
        np.testing.assert_allclose(out["Bx"].values, exp_x, atol=1e-12)
        np.testing.assert_allclose(out["Bz"].values, exp_z, atol=1e-12)


class TestNaN:
    def test_nan_data_passes_through(self):
        ds = _ds(TIMES2, Bx=[np.nan, 1.0], By=[2.0, np.nan], Bz=[3.0, 4.0])
        out = rotate_dataset(ds, "GSE", _angles(TIMES2, psi=10.0, mu=0.0))
        # Bx untouched by GSE: NaN stays NaN, finite stays finite.
        assert np.isnan(out["Bx"].values[0])
        assert out["Bx"].values[1] == 1.0
        # Row 1 mixes NaN By into both rotated components.
        assert np.isnan(out["By"].values[1])
        assert np.isnan(out["Bz"].values[1])
        # Row 0 rotated components are finite.
        assert np.isfinite(out["By"].values[0])
        assert np.isfinite(out["Bz"].values[0])

    def test_missing_angle_row_nans_all_six_vectors(self):
        times = pd.date_range("2024-03-01 00:00", periods=3, freq="min")
        ds = _ds(times, Bx=[1.0] * 3, By=[2.0] * 3, Bz=[3.0] * 3,
                 Ux=[-400.0] * 3, Uy=[5.0] * 3, Uz=[6.0] * 3,
                 rho=[7.0] * 3, T=[8e4] * 3)
        # Angle table is missing the middle minute.
        angles = _angles([times[0], times[2]], psi=10.0, mu=20.0)
        for coords in ("GSE", "SM"):
            out = rotate_dataset(ds, coords, angles)
            for var in ("Bx", "By", "Bz", "Ux", "Uy", "Uz"):
                assert np.isnan(out[var].values[1]), (coords, var)
                assert np.isfinite(out[var].values[0]), (coords, var)
                assert np.isfinite(out[var].values[2]), (coords, var)
            # Scalars untouched, even on the missing-angle row.
            np.testing.assert_array_equal(out["rho"].values, ds["rho"].values)
            np.testing.assert_array_equal(out["T"].values, ds["T"].values)


class TestPerMinuteAngles:
    def test_row_wise_application(self):
        # Two rows, same vector, very different psi: each output row must
        # reflect its own minute's angle.
        ds = _ds(TIMES2, Bx=[1.0, 1.0], By=[2.0, 2.0], Bz=[3.0, 3.0])
        out = rotate_dataset(ds, "GSE", _angles(TIMES2, psi=[0.0, 90.0], mu=0.0))
        np.testing.assert_allclose(out["By"].values, [2.0, -3.0], atol=1e-12)
        np.testing.assert_allclose(out["Bz"].values, [3.0, 2.0], atol=1e-12)


class TestRealEpochBakedValues:
    """Expected values computed offline with the pipeline's l1_angles rx()/ry()
    (spacepy-derived angle table rows for 2026-07-01T00:00 and 00:01)."""

    TIMES = pd.date_range("2026-07-01 00:00", periods=2, freq="min")

    def _real_ds(self):
        return _ds(
            self.TIMES,
            Bx=[1.23, 1.25], By=[0.45, 0.44], Bz=[-2.10, -2.08],
            Ux=[-400.5, -401.2], Uy=[12.30, 12.10], Uz=[-5.10, -5.00],
        )

    def _real_angles(self):
        return pd.read_csv(
            DATA_DIR / "202607_angles.csv",
            parse_dates=["timestamp"], index_col="timestamp",
        )

    def test_gse(self):
        out = rotate_dataset(self._real_ds(), "GSE", self._real_angles())
        np.testing.assert_allclose(out["Bx"].values, [1.23, 1.25], rtol=1e-10)
        np.testing.assert_allclose(
            out["By"].values, [0.250114825597, 0.242585826684], rtol=1e-10)
        np.testing.assert_allclose(
            out["Bz"].values, [-2.133059439870, -2.112143962113], rtol=1e-10)
        np.testing.assert_allclose(out["Ux"].values, [-400.5, -401.2], rtol=1e-10)
        np.testing.assert_allclose(
            out["Uy"].values, [11.764697942364, 11.576580101903], rtol=1e-10)
        np.testing.assert_allclose(
            out["Uz"].values, [-6.236335648835, -6.114964688715], rtol=1e-10)

    def test_sm(self):
        out = rotate_dataset(self._real_ds(), "SM", self._real_angles())
        np.testing.assert_allclose(
            out["Bx"].values, [1.881807377014, 1.892641442108], rtol=1e-10)
        np.testing.assert_allclose(out["By"].values, [0.45, 0.44], rtol=1e-10)
        np.testing.assert_allclose(
            out["Bz"].values, [-1.543276059496, -1.518818083780], rtol=1e-10)
        np.testing.assert_allclose(
            out["Ux"].values, [-373.888751216564, -374.673783029867], rtol=1e-10)
        np.testing.assert_allclose(out["Uy"].values, [12.30, 12.10], rtol=1e-10)
        np.testing.assert_allclose(
            out["Uz"].values, [-143.643523048267, -143.547888560188], rtol=1e-10)


class TestLoadAngles:
    def test_reads_fixture_via_cache(self):
        from unittest.mock import patch

        with patch(
            "midl._coords.ensure_cached",
            return_value=DATA_DIR / "202403_angles.csv",
        ) as mock_ec:
            df = load_angles(
                pd.Timestamp("2024-03-01 00:00"), pd.Timestamp("2024-03-01 00:09"))
        mock_ec.assert_called_once_with("2024-03", "angles")
        assert list(df.columns) == ["gsm_gse_angle_deg", "dipole_tilt_deg"]
        assert len(df) == 10
        assert df.loc["2024-03-01 00:00:00", "gsm_gse_angle_deg"] == -12.3456
