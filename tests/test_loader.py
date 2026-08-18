"""Tests for midl._loader."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from midl._loader import _read_csv, _to_dataset, load

DATA_DIR = Path(__file__).parent / "data"


class TestReadCsv:
    def test_reads_32re(self):
        df = _read_csv(DATA_DIR / "202403_32Re.csv")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "Bx" in df.columns
        assert "T" in df.columns
        assert len(df) == 10

    def test_reads_l1(self):
        df = _read_csv(DATA_DIR / "202403_L1.csv")
        assert "X" in df.columns
        assert "B_source" in df.columns
        assert len(df) == 10

    def test_nan_handling(self):
        df = _read_csv(DATA_DIR / "202403_32Re.csv")
        # Row at 00:05 is all NaN
        row = df.loc["2024-03-01 00:05:00"]
        assert pd.isna(row["Bx"])
        assert pd.isna(row["T"])


class TestToDataset:
    def test_creates_dataset(self):
        df = _read_csv(DATA_DIR / "202403_32Re.csv")
        ds = _to_dataset(df, "32Re")
        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.coords
        assert "Bx" in ds.data_vars
        assert ds.attrs["target"] == "32Re"

    def test_variable_attrs(self):
        df = _read_csv(DATA_DIR / "202403_32Re.csv")
        ds = _to_dataset(df, "32Re")
        assert ds["Bx"].attrs["units"] == "nT"
        assert ds["rho"].attrs["units"] == "cm^-3"
        assert ds["Ux"].attrs["coordinate_system"] == "GSM"


class TestLoadFixedBallistic:
    def test_load_slices_time(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:02", "2024-03-01 00:06", 32)
            assert len(ds.time) == 5  # 00:02, 00:03, 00:04, 00:05, 00:06
            assert ds.time.values[0] == pd.Timestamp("2024-03-01 00:02")
            assert ds.time.values[-1] == pd.Timestamp("2024-03-01 00:06")

    def test_default_method_is_ballistic(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 32)
        assert ds.attrs["midl_propagation"] == {"method": "ballistic", "target_re": 32.0}

    def test_load_14re(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 14)
        assert ds.attrs["target"] == "14Re"

    def test_load_float_14(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 14.0)
        assert ds.attrs["target"] == "14Re"

    def test_load_rejects_bad_range(self):
        with pytest.raises(ValueError, match="must be <="):
            load("2024-03-02", "2024-03-01", 32)


class TestLoadL1:
    def test_load_l1(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", "l1")
        assert "X" in ds.data_vars
        assert "B_source" in ds.data_vars
        assert ds.attrs["target"] == "L1"
        assert ds.attrs["midl_propagation"] is None

    def test_load_l1_case_insensitive(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", "L1")
        assert ds.attrs["target"] == "L1"

    def test_l1_rejects_mhd_method(self):
        with pytest.raises(ValueError, match="'l1' is only valid with method='ballistic'"):
            load("2024-03-01", "2024-03-01 00:03", "l1", method="mhd")


class TestLoadMhd:
    def test_load_mhd(self):
        with patch(
            "midl._loader.ensure_cached",
            return_value=DATA_DIR / "202403_mhd_032Re.csv",
        ):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 32, method="mhd")
        assert set(ds.data_vars) >= {"Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"}
        assert ds.attrs["target"] == "32Re"
        assert ds.attrs["midl_propagation"] == {"method": "mhd", "target_re": 32.0}
        assert ds["Bx"].attrs["units"] == "nT"

    def test_mhd_rejects_non_integer_re(self):
        with pytest.raises(ValueError, match="must be an integer"):
            load("2024-03-01", "2024-03-01 00:03", 32.5, method="mhd")

    def test_mhd_rejects_out_of_range(self):
        with pytest.raises(ValueError, match=r"\[-70, 70\]"):
            load("2024-03-01", "2024-03-01 00:03", 200, method="mhd")


class TestLoadCustomBallistic:
    def test_custom_re_invokes_client_propagation(self):
        l1_df = _read_csv(DATA_DIR / "202403_L1.csv")
        times = pd.date_range("2024-03-01 00:00", "2024-03-01 00:09", freq="min")
        sentinel = xr.Dataset(
            {"Bx": (("time",), np.arange(len(times), dtype=float))},
            coords={"time": times},
        )
        with (
            patch("midl._loader._load_monthly", return_value=l1_df) as mock_lm,
            patch("midl._loader.propagate", return_value=sentinel) as mock_prop,
        ):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 20)

        mock_lm.assert_called_once()
        # _load_monthly receives the L1 canonical.
        assert mock_lm.call_args[0][2] == "L1"
        mock_prop.assert_called_once()
        call_args = mock_prop.call_args[0]
        assert call_args[1] == "ballistic"
        assert call_args[2] == 20.0
        # Result is sliced to the requested range.
        assert ds.time.values[0] == pd.Timestamp("2024-03-01 00:00")
        assert ds.time.values[-1] == pd.Timestamp("2024-03-01 00:03")

    def test_custom_re_pads_leading_edge(self):
        l1_df = _read_csv(DATA_DIR / "202403_L1.csv")
        sentinel = xr.Dataset(
            {"Bx": (("time",), [0.0])},
            coords={"time": [pd.Timestamp("2024-03-01 00:00")]},
        )
        with (
            patch("midl._loader._load_monthly", return_value=l1_df) as mock_lm,
            patch("midl._loader.propagate", return_value=sentinel),
        ):
            load("2024-03-01 05:00", "2024-03-01 05:09", 20)

        # _load_monthly(start_ts_padded, end_ts, "L1") — start is padded backward.
        l1_start_arg = mock_lm.call_args[0][0]
        assert l1_start_arg == pd.Timestamp("2024-03-01 03:00")


class TestLoadValidation:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            load("2024-03-01", "2024-03-01 00:03", 32, method="magic")

    def test_non_string_method(self):
        with pytest.raises(ValueError, match="method must be a string"):
            load("2024-03-01", "2024-03-01 00:03", 32, method=42)  # type: ignore[arg-type]

    def test_bad_target_re_string(self):
        with pytest.raises(ValueError, match="must be a number or 'l1'"):
            load("2024-03-01", "2024-03-01 00:03", "bogus")

    def test_bad_target_re_type(self):
        with pytest.raises(ValueError, match="must be a number or 'l1'"):
            load("2024-03-01", "2024-03-01 00:03", [32])  # type: ignore[arg-type]

    def test_method_case_insensitive(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 32, method="BALLISTIC")
        assert ds.attrs["midl_propagation"]["method"] == "ballistic"


class TestLoadCoords:
    def test_gsm_output_identical_to_default(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            default = load("2024-03-01 00:00", "2024-03-01 00:09", 32)
            explicit = load("2024-03-01 00:00", "2024-03-01 00:09", 32, coords="GSM")
        xr.testing.assert_identical(default, explicit)
        assert default.attrs["coords_system"] == "GSM"
        assert default["Bx"].attrs["coordinate_system"] == "GSM"

    def test_gse_sets_attrs(self):
        with (
            patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"),
            patch("midl._coords.ensure_cached", return_value=DATA_DIR / "202403_angles.csv"),
        ):
            ds = load("2024-03-01 00:00", "2024-03-01 00:09", 32, coords="GSE")
        assert ds.attrs["coords_system"] == "GSE"
        for var in ("Bx", "By", "Bz", "Ux", "Uy", "Uz"):
            assert ds[var].attrs["coordinate_system"] == "GSE"

    def test_gse_rotates_values(self):
        with (
            patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"),
            patch("midl._coords.ensure_cached", return_value=DATA_DIR / "202403_angles.csv"),
        ):
            gsm = load("2024-03-01 00:00", "2024-03-01 00:09", 32)
            gse = load("2024-03-01 00:00", "2024-03-01 00:09", 32, coords="GSE")
        psi = np.radians(-12.3456)  # angle fixture row for 00:00
        exp_by = np.cos(psi) * gsm["By"].values[0] - np.sin(psi) * gsm["Bz"].values[0]
        exp_bz = np.sin(psi) * gsm["By"].values[0] + np.cos(psi) * gsm["Bz"].values[0]
        assert gse["Bx"].values[0] == gsm["Bx"].values[0]
        np.testing.assert_allclose(gse["By"].values[0], exp_by, rtol=1e-12)
        np.testing.assert_allclose(gse["Bz"].values[0], exp_bz, rtol=1e-12)

    def test_coords_case_insensitive(self):
        with (
            patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"),
            patch("midl._coords.ensure_cached", return_value=DATA_DIR / "202403_angles.csv"),
        ):
            ds = load("2024-03-01 00:00", "2024-03-01 00:09", 32, coords="gse")
        assert ds.attrs["coords_system"] == "GSE"

    def test_l1_x_stays_gsm(self):
        with (
            patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"),
            patch("midl._coords.ensure_cached", return_value=DATA_DIR / "202403_angles.csv"),
        ):
            gsm = load("2024-03-01 00:00", "2024-03-01 00:09", "l1")
            gse = load("2024-03-01 00:00", "2024-03-01 00:09", "l1", coords="GSE")
        np.testing.assert_array_equal(gse["X"].values, gsm["X"].values)
        assert gse["X"].attrs["coordinate_system"] == "GSM"

    def test_invalid_coords_raises(self):
        with pytest.raises(ValueError, match="'GSM', 'GSE', 'SM'"):
            load("2024-03-01", "2024-03-01 00:03", 32, coords="GEI")

    def test_custom_ballistic_propagates_unrotated_gsm(self):
        # ORDERING: propagate() must receive native-GSM (un-rotated) data;
        # the rotation happens after propagation and the final time slice.
        l1_df = _read_csv(DATA_DIR / "202403_L1.csv")
        times = pd.date_range("2024-03-01 00:00", "2024-03-01 00:09", freq="min")
        sentinel = xr.Dataset(
            {"Bx": (("time",), np.arange(len(times), dtype=float))},
            coords={"time": times},
        )
        with (
            patch("midl._loader._load_monthly", return_value=l1_df),
            patch("midl._loader.propagate", return_value=sentinel) as mock_prop,
            patch("midl._coords.ensure_cached", return_value=DATA_DIR / "202403_angles.csv"),
        ):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", 20, coords="GSE")

        mock_prop.assert_called_once()
        prop_input = mock_prop.call_args[0][0]
        # The dataset handed to propagate carries the raw GSM fixture values.
        np.testing.assert_array_equal(prop_input["By"].values, l1_df["By"].values)
        np.testing.assert_array_equal(prop_input["Bz"].values, l1_df["Bz"].values)
        assert prop_input.attrs.get("coords_system", "GSM") == "GSM"
        # The returned dataset is stamped with the requested frame.
        assert ds.attrs["coords_system"] == "GSE"
