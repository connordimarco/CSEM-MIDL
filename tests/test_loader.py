"""Tests for midl._loader."""

from pathlib import Path
from unittest.mock import patch

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


class TestLoad:
    def test_load_slices_time(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_32Re.csv"):
            ds = load("2024-03-01 00:02", "2024-03-01 00:06", "32re")
            assert len(ds.time) == 5  # 00:02, 00:03, 00:04, 00:05, 00:06
            assert ds.time.values[0] == pd.Timestamp("2024-03-01 00:02")
            assert ds.time.values[-1] == pd.Timestamp("2024-03-01 00:06")

    def test_load_l1(self):
        with patch("midl._loader.ensure_cached", return_value=DATA_DIR / "202403_L1.csv"):
            ds = load("2024-03-01 00:00", "2024-03-01 00:03", "l1")
            assert "X" in ds.data_vars
            assert "B_source" in ds.data_vars
            assert ds.attrs["target"] == "L1"

    def test_load_rejects_bad_range(self):
        with pytest.raises(ValueError, match="must be <="):
            load("2024-03-02", "2024-03-01", "32re")
