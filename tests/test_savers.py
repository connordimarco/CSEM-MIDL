"""Tests for midl._savers."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import xarray as xr

from midl._loader import _read_csv, _to_dataset
from midl._savers import to_csv, to_dat

DATA_DIR = Path(__file__).parent / "data"


def _load_sample_32re() -> xr.Dataset:
    df = _read_csv(DATA_DIR / "202403_32Re.csv")
    return _to_dataset(df, "32Re")


def _load_sample_l1() -> xr.Dataset:
    df = _read_csv(DATA_DIR / "202403_L1.csv")
    return _to_dataset(df, "L1")


class TestToCsv:
    def test_roundtrip(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.csv"
        to_csv(ds, out)

        df = pd.read_csv(out, parse_dates=["timestamp"], index_col="timestamp")
        assert len(df) == 10
        assert "Bx" in df.columns
        # Check timestamp format
        with open(out) as f:
            lines = f.readlines()
        assert "2024-03-01T00:00:00" in lines[1]

    def test_csv_precision(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.csv"
        to_csv(ds, out)

        df = pd.read_csv(out)
        # T should be integer (0 decimals)
        t_str = df["T"].dropna().iloc[0]
        assert float(t_str) == int(float(t_str))


class TestToDat:
    def test_header_format(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert lines[0].startswith("MIDL 32Re Data")
        assert "nT, km/s, cm^-3, K" in lines[0]
        assert lines[1].strip().startswith("year")
        assert lines[2].strip() == "#START"

    def test_data_lines(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        # First data line
        data_line = lines[3]
        parts = data_line.split()
        assert parts[0] == "2024"
        assert parts[1] == "3"
        assert parts[2] == "1"
        assert parts[3] == "0"  # hour
        assert parts[4] == "0"  # minute

    def test_nan_as_nan_string(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            content = f.read()
        lines = content.strip().split("\n")
        # Row at index 5 (line 8 = header3 + 5 data rows) has NaN
        nan_line = lines[3 + 5]  # 3 header + row index 5
        assert "nan" in nan_line

    def test_l1_has_extra_columns(self, tmp_path):
        ds = _load_sample_l1()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert "X B_source" in lines[1]
        assert ", Re)" in lines[0]

    def test_no_target_attr_raises(self, tmp_path):
        ds = _load_sample_32re()
        del ds.attrs["target"]
        import pytest
        with pytest.raises(ValueError, match="target"):
            to_dat(ds, tmp_path / "out.dat")
