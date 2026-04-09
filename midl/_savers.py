"""Export xarray Datasets to CSV and DAT formats."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import xarray as xr

_CSV_PRECISION: dict[str, int] = {
    "Bx": 2, "By": 2, "Bz": 2,
    "Ux": 1, "Uy": 2, "Uz": 2,
    "rho": 3, "T": 0, "X": 1,
}

# DAT field specs: (width, decimals) — matches MIDL-Web/static/app.js:484-491
_DAT_FIELDS: list[tuple[str, int, int]] = [
    ("Bx", 9, 2), ("By", 9, 2), ("Bz", 9, 2),
    ("Ux", 10, 1), ("Uy", 10, 2), ("Uz", 10, 2),
    ("rho", 10, 3), ("T", 11, 0),
]

_DAT_L1_EXTRA_FLOAT = ("X", 10, 1)
_DAT_L1_SOURCE_COLS = ["B_source", "Ux_source", "Uyz_source", "rho_source", "T_source"]
_DAT_SOURCE_WIDTH = 5


def to_csv(ds: xr.Dataset, path: str | Path) -> None:
    """Write a MIDL Dataset to CSV in the standard MIDL format.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset returned by ``midlpy.load()``.
    path : str or Path
        Output file path.
    """
    df = ds.to_dataframe()
    df.index.name = "timestamp"
    for col, decimals in _CSV_PRECISION.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)
    df.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S")


def _fmt_float(value: float, width: int, decimals: int) -> str:
    """Format a float with fixed width, or right-justified 'nan'."""
    if math.isnan(value):
        return "nan".rjust(width)
    return f"{value:.{decimals}f}".rjust(width)


def _fmt_source(value: object, width: int) -> str:
    """Format a source column value, right-justified."""
    s = str(value).strip() if pd.notna(value) else "nan"
    return s.rjust(width)


def to_dat(ds: xr.Dataset, path: str | Path) -> None:
    """Write a MIDL Dataset to SWMF-compatible DAT format.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset returned by ``midlpy.load()``.
    path : str or Path
        Output file path.
    """
    target = ds.attrs.get("target")
    if target is None:
        raise ValueError(
            "Dataset has no 'target' attribute. "
            "Use a Dataset returned by midlpy.load()."
        )

    is_l1 = target == "L1"

    # Build header
    units = "GSM nT, km/s, cm^-3, K"
    if is_l1:
        units += ", Re"
    header1 = f"MIDL {target} Data ({units})\n"

    cols = "year month day hour minute Bx By Bz Ux Uy Uz rho T"
    if is_l1:
        cols += " X B_source Ux_source Uyz_source rho_source T_source"
    header2 = cols + "\n"

    df = ds.to_dataframe()

    with open(path, "w", encoding="utf-8") as f:
        f.write(header1)
        f.write(header2)
        f.write("#START\n")

        for t, row in df.iterrows():
            ts = pd.Timestamp(t)
            line = (
                f"{ts.year:4d}"
                f"{ts.month:3d}"
                f"{ts.day:3d}"
                f"{ts.hour:3d}"
                f"{ts.minute:3d}"
            )

            for col, width, decimals in _DAT_FIELDS:
                line += _fmt_float(float(row[col]), width, decimals)

            if is_l1:
                col, width, decimals = _DAT_L1_EXTRA_FLOAT
                line += _fmt_float(float(row[col]), width, decimals)
                for src_col in _DAT_L1_SOURCE_COLS:
                    line += _fmt_source(row.get(src_col), _DAT_SOURCE_WIDTH)

            f.write(line + "\n")
