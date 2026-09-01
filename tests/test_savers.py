"""Tests for midl._savers."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from midl._loader import _read_csv, _to_dataset
from midl._savers import _INTERP_COLS, to_csv, to_dat

DATA_DIR = Path(__file__).parent / "data"


def _load_sample_32re() -> xr.Dataset:
    df = _read_csv(DATA_DIR / "202403_32Re.csv")
    return _to_dataset(df, "32Re")


def _load_sample_l1() -> xr.Dataset:
    df = _read_csv(DATA_DIR / "202403_L1.csv")
    return _to_dataset(df, "L1")


def _add_interp(ds: xr.Dataset) -> xr.Dataset:
    """Attach synthetic {B,Ux,Uyz,rho,T}_interp flags (sample CSVs predate them).

    Values cycle 0,1,2,3 with index 1 set to NaN so the missing-flag (blank in
    CSV / 'nan' in DAT) path is exercised.
    """
    ds = ds.copy()
    n = ds.sizes["time"]
    base = np.array([float(i % 4) for i in range(n)])
    base[1] = np.nan
    for col in _INTERP_COLS:
        ds[col] = ("time", base.copy())
    return ds


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

    def test_csv_writes_interp_as_int(self, tmp_path):
        ds = _add_interp(_load_sample_32re())
        out = tmp_path / "out.csv"
        to_csv(ds, out)

        with open(out) as f:
            lines = f.read().splitlines()
        assert all(c in lines[0] for c in _INTERP_COLS)
        # Integer flags, not floats: "1" not "1.0". Index-1 row is NaN -> blank.
        body = lines[1:]
        assert body[0].rstrip().endswith(",0,0,0,0,0")
        assert body[1].rstrip().endswith(",,,,,")  # NaN -> empty fields
        assert body[2].rstrip().endswith(",2,2,2,2,2")


class TestToDat:
    def test_header_format(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert lines[0].startswith("MIDL 32Re,")
        assert "(nT, km/s, cm^-3, K)" in lines[0]
        assert lines[1].strip() == ""
        assert lines[2].strip() == "#COORDINATES"
        assert lines[3].split()[0] == "GSM"
        assert lines[3].strip().endswith("Earth orbital motion EXCLUDED")
        assert lines[4].strip() == ""
        assert lines[5].split()[:5] == ["year", "month", "day", "hour", "minute"]
        assert lines[5].split()[5:13] == [
            "Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"]
        assert lines[6].strip() == "#START"

    def test_orbital_motion_included_line(self, tmp_path):
        ds = _load_sample_32re()
        ds.attrs["orbital_motion"] = "included"
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert lines[3].split()[0] == "GSM"
        assert lines[3].strip().endswith("Earth orbital motion included")

    def test_data_lines(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        # First data line: 5 time integers, then the floats.
        data_line = lines[7]
        parts = data_line.split()
        assert parts[0] == "2024"
        assert parts[1] == "3"
        assert parts[2] == "1"
        assert parts[3] == "0"  # hour
        assert parts[4] == "0"  # minute
        assert "." in parts[5]  # Bx directly follows the time stamp

    def test_nan_as_nan_string(self, tmp_path):
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            content = f.read()
        lines = content.strip().split("\n")
        # Row at index 5 has NaN (7 header lines precede the data)
        nan_line = lines[7 + 5]
        assert "nan" in nan_line

    def test_l1_has_extra_columns(self, tmp_path):
        ds = _load_sample_l1()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert "X B_source" in lines[5]
        assert ", Re)" in lines[0]

    def test_no_target_attr_raises(self, tmp_path):
        ds = _load_sample_32re()
        del ds.attrs["target"]
        import pytest
        with pytest.raises(ValueError, match="target"):
            to_dat(ds, tmp_path / "out.dat")

    def test_interp_columns_in_header_and_rows(self, tmp_path):
        # Ballistic product: no L1 source cols, but interp must still appear.
        ds = _add_interp(_load_sample_32re())
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.read().splitlines()
        header = lines[5].split()
        assert header[-5:] == _INTERP_COLS
        # Data rows: 5 trailing flag fields (width 5), nan on the NaN row.
        first_row = lines[7].split()
        assert first_row[-5:] == ["0", "0", "0", "0", "0"]
        nan_row = lines[8].split()
        assert nan_row[-5:] == ["nan", "nan", "nan", "nan", "nan"]

    def test_interp_columns_follow_l1_source(self, tmp_path):
        # L1 product: interp flags come after the five *_source columns.
        ds = _add_interp(_load_sample_l1())
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.read().splitlines()
        header = lines[5].split()
        assert header[-10:] == [
            "B_source", "Ux_source", "Uyz_source", "rho_source", "T_source",
            *_INTERP_COLS,
        ]

    def test_coordinates_block_says_gse_when_rotated(self, tmp_path):
        ds = _load_sample_32re()
        ds.attrs["coords_system"] = "GSE"
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.readlines()
        assert lines[2].strip() == "#COORDINATES"
        assert lines[3].split()[0] == "GSE"
        assert "GSM" not in lines[3]

    def test_header_units_gsm_byte_identical(self, tmp_path):
        # With coords_system='GSM' (or absent) the header is exactly the
        # historical string; the golden reference fixtures rely on this.
        ds = _load_sample_32re()
        out_default = tmp_path / "default.dat"
        to_dat(ds, out_default)
        ds.attrs["coords_system"] = "GSM"
        out_gsm = tmp_path / "gsm.dat"
        to_dat(ds, out_gsm)

        assert out_default.read_bytes() == out_gsm.read_bytes()
        with open(out_default) as f:
            lines = f.readlines()
        assert lines[3].split()[0] == "GSM"

    def test_no_interp_when_absent(self, tmp_path):
        # MHD / legacy datasets have no flags: header and rows stay flag-free.
        ds = _load_sample_32re()
        out = tmp_path / "out.dat"
        to_dat(ds, out)

        with open(out) as f:
            lines = f.read().splitlines()
        assert not any(c in lines[5] for c in _INTERP_COLS)
        assert lines[5].split()[-1] == "T"
