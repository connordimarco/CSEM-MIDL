"""Timestamp parsing and month-range utilities."""

from __future__ import annotations

import datetime
from typing import Union

import numpy as np
import pandas as pd

Timelike = Union[str, datetime.datetime, pd.Timestamp, np.datetime64]


def parse_timestamp(value: Timelike) -> pd.Timestamp:
    """Convert any datetime-like value to a pandas Timestamp.

    Accepts ISO 8601 strings, datetime.datetime, numpy.datetime64,
    or pandas.Timestamp.
    """
    if isinstance(value, (str, datetime.datetime, pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value)
    raise TypeError(
        f"Unsupported type {type(value).__name__!r}. "
        "Expected str, datetime, numpy.datetime64, or pandas.Timestamp."
    )


def months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Return all 'YYYY-MM' strings for months touched by [start, end]."""
    months: list[str] = []
    y, m = start.year, start.month
    ey, em = end.year, end.month
    while y < ey or (y == ey and m <= em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months
