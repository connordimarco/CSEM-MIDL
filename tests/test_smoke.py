"""Smoke tests that compare midl output against website reference files.

These tests require network access (or cached data) to download from the
MIDL web host. Mark with ``pytest -m smoke`` to run, skipped by default.
"""

from pathlib import Path

import pytest

import midl

REF_DIR = Path(__file__).parent / "data" / "reference"

pytestmark = pytest.mark.smoke


def _compare_files(generated: Path, reference: Path) -> None:
    gen = generated.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    ref = reference.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    assert gen == ref, f"Mismatch: {generated.name} vs {reference.name}"


# --- Test case 1: single day, single month ---

CASE1 = ("2005-01-01", "2005-01-02", "2005-01-01T0000_to_2005-01-02T0000")


@pytest.mark.parametrize("target", ["14re", "32re", "l1"])
class TestSingleDay:
    def test_csv(self, target, tmp_path):
        start, end, label = CASE1
        canonical = {"14re": "14Re", "32re": "32Re", "l1": "L1"}[target]
        ds = midl.load(start, end, target)
        out = tmp_path / f"test_{target}.csv"
        midl.to_csv(ds, out)
        ref = REF_DIR / f"MIDL_{canonical}_{label}.csv"
        _compare_files(out, ref)

    def test_dat(self, target, tmp_path):
        start, end, label = CASE1
        canonical = {"14re": "14Re", "32re": "32Re", "l1": "L1"}[target]
        ds = midl.load(start, end, target)
        out = tmp_path / f"test_{target}.dat"
        midl.to_dat(ds, out)
        ref = REF_DIR / f"MIDL_{canonical}_{label}.dat"
        _compare_files(out, ref)


# --- Test case 2: cross-month range ---

CASE2 = ("2005-01-01 10:00", "2005-02-03 13:00", "2005-01-01T1000_to_2005-02-03T1300")


@pytest.mark.parametrize("target", ["14re", "32re", "l1"])
class TestCrossMonth:
    def test_csv(self, target, tmp_path):
        start, end, label = CASE2
        canonical = {"14re": "14Re", "32re": "32Re", "l1": "L1"}[target]
        ds = midl.load(start, end, target)
        out = tmp_path / f"test_{target}.csv"
        midl.to_csv(ds, out)
        ref = REF_DIR / f"MIDL_{canonical}_{label}.csv"
        _compare_files(out, ref)

    def test_dat(self, target, tmp_path):
        start, end, label = CASE2
        canonical = {"14re": "14Re", "32re": "32Re", "l1": "L1"}[target]
        ds = midl.load(start, end, target)
        out = tmp_path / f"test_{target}.dat"
        midl.to_dat(ds, out)
        ref = REF_DIR / f"MIDL_{canonical}_{label}.dat"
        _compare_files(out, ref)
