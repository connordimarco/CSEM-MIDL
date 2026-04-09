"""Tests for midl._time."""

import datetime

import numpy as np
import pandas as pd
import pytest

from midl._time import months_in_range, parse_timestamp


class TestParseTimestamp:
    def test_iso_string(self):
        result = parse_timestamp("2015-03-17")
        assert result == pd.Timestamp("2015-03-17")

    def test_iso_string_with_time(self):
        result = parse_timestamp("2015-03-17 14:30")
        assert result == pd.Timestamp("2015-03-17 14:30")

    def test_datetime(self):
        dt = datetime.datetime(2015, 3, 17, 14, 30)
        result = parse_timestamp(dt)
        assert result == pd.Timestamp("2015-03-17 14:30")

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2015-03-17")
        result = parse_timestamp(ts)
        assert result == ts

    def test_numpy_datetime64(self):
        dt64 = np.datetime64("2015-03-17")
        result = parse_timestamp(dt64)
        assert result == pd.Timestamp("2015-03-17")

    def test_rejects_int(self):
        with pytest.raises(TypeError, match="int"):
            parse_timestamp(42)

    def test_rejects_garbage_string(self):
        with pytest.raises(ValueError):
            parse_timestamp("not-a-date")


class TestMonthsInRange:
    def test_single_month(self):
        start = pd.Timestamp("2015-03-10")
        end = pd.Timestamp("2015-03-20")
        assert months_in_range(start, end) == ["2015-03"]

    def test_two_months(self):
        start = pd.Timestamp("2015-03-17")
        end = pd.Timestamp("2015-04-02")
        assert months_in_range(start, end) == ["2015-03", "2015-04"]

    def test_cross_year(self):
        start = pd.Timestamp("2015-11-01")
        end = pd.Timestamp("2016-02-15")
        assert months_in_range(start, end) == [
            "2015-11", "2015-12", "2016-01", "2016-02"
        ]

    def test_same_day(self):
        ts = pd.Timestamp("2015-03-17")
        assert months_in_range(ts, ts) == ["2015-03"]

    def test_full_year(self):
        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-12-31")
        result = months_in_range(start, end)
        assert len(result) == 12
        assert result[0] == "2020-01"
        assert result[-1] == "2020-12"
