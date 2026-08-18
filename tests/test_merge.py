"""Tests for midl.merge() — multi-source combination."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from midl._merge import merge


def _synthetic_dataset(
    n: int = 60,
    bx: float = 5.0,
    ux: float = -400.0,
    rho: float = 5.0,
    T: float = 1e5,
    start: str = "2024-05-01",
    noise: float = 0.0,
    seed: int = 42,
) -> xr.Dataset:
    idx = pd.date_range(start, periods=n, freq="min")
    rng = np.random.default_rng(seed)
    data = {
        "Bx": np.full(n, bx) + noise * rng.standard_normal(n),
        "By": np.zeros(n) + noise * rng.standard_normal(n),
        "Bz": np.zeros(n) + noise * rng.standard_normal(n),
        "Ux": np.full(n, ux) + noise * rng.standard_normal(n),
        "Uy": np.zeros(n) + noise * rng.standard_normal(n),
        "Uz": np.zeros(n) + noise * rng.standard_normal(n),
        "rho": np.full(n, rho) + noise * rng.standard_normal(n),
        "T": np.full(n, T) + noise * rng.standard_normal(n),
    }
    return xr.Dataset(
        {k: ("time", v) for k, v in data.items()},
        coords={"time": idx},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMergeValidation:
    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            merge({})

    def test_non_dataset_raises(self):
        with pytest.raises(TypeError, match="xarray.Dataset"):
            merge({"bad": pd.DataFrame()})

    def test_missing_time_coord_raises(self):
        ds = xr.Dataset({"Bx": ("x", [1, 2, 3])})
        with pytest.raises(ValueError, match="time"):
            merge({"bad": ds})


class TestMergeCoords:
    def test_mixed_coords_raises(self):
        a = _synthetic_dataset()
        b = _synthetic_dataset()
        b.attrs["coords_system"] = "GSE"
        with pytest.raises(ValueError, match="different coordinate systems"):
            merge({"a": a, "b": b})

    def test_shared_coords_propagates_to_output(self):
        a = _synthetic_dataset()
        b = _synthetic_dataset()
        a.attrs["coords_system"] = "GSE"
        b.attrs["coords_system"] = "GSE"
        merged = merge({"a": a, "b": b})
        assert merged.attrs["coords_system"] == "GSE"
        assert merged["Bx"].attrs["coordinate_system"] == "GSE"

    def test_default_is_gsm(self):
        merged = merge({"a": _synthetic_dataset(), "b": _synthetic_dataset()})
        assert merged.attrs["coords_system"] == "GSM"
        assert merged["Bx"].attrs["coordinate_system"] == "GSM"

    def test_single_source_path_propagates_coords(self):
        a = _synthetic_dataset()
        a.attrs["coords_system"] = "SM"
        merged = merge({"a": a})
        assert merged.attrs["coords_system"] == "SM"
        assert merged["Ux"].attrs["coordinate_system"] == "SM"


# ---------------------------------------------------------------------------
# Single source
# ---------------------------------------------------------------------------


class TestMergeSingleSource:
    def test_passthrough(self):
        ds = _synthetic_dataset(n=10, bx=3.0)
        result = merge({"only": ds})
        np.testing.assert_allclose(result["Bx"].values, 3.0, atol=1e-10)

    def test_provenance_set(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"mysat": ds})
        assert all(v == "mysat" for v in result["B_source"].values if v)


# ---------------------------------------------------------------------------
# Two sources agreeing
# ---------------------------------------------------------------------------


class TestMergeTwoAgreeing:
    def test_mean_of_two(self):
        ds1 = _synthetic_dataset(n=10, bx=5.0)
        ds2 = _synthetic_dataset(n=10, bx=6.0)  # within ±8 nT threshold
        result = merge({"a": ds1, "b": ds2})
        # B coupled via |B|: both agree so output should be mean
        # |B| for ds1 = 5.0, |B| for ds2 = 6.0, diff=1 < 8 -> agree -> mean=5.5
        # Then _apply_source_to_components: Bx from both -> mean = 5.5
        np.testing.assert_allclose(result["Bx"].values, 5.5, atol=1e-10)

    def test_both_in_provenance(self):
        ds1 = _synthetic_dataset(n=10, bx=5.0)
        ds2 = _synthetic_dataset(n=10, bx=6.0)
        result = merge({"a": ds1, "b": ds2})
        assert all(v == "a,b" for v in result["B_source"].values if v)


# ---------------------------------------------------------------------------
# Two sources disagreeing
# ---------------------------------------------------------------------------


class TestMergeTwoDisagreeing:
    def test_fallback_uses_first_alphabetically(self):
        ds1 = _synthetic_dataset(n=10, bx=5.0)
        ds2 = _synthetic_dataset(n=10, bx=50.0)  # diff=45 > 8 nT
        result = merge({"alpha": ds1, "beta": ds2})
        # No prev_value at start -> first alphabetically = "alpha"
        np.testing.assert_allclose(result["Bx"].values, 5.0, atol=1e-10)

    def test_deprioritize(self):
        ds1 = _synthetic_dataset(n=10, ux=-400.0)
        ds2 = _synthetic_dataset(n=10, ux=-600.0)  # diff=200 > 80 km/s
        result = merge({"a": ds1, "b": ds2},
                       deprioritize={"Ux": ["a"]})
        # "a" is deprioritized -> "b" chosen
        np.testing.assert_allclose(result["Ux"].values, -600.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Three sources
# ---------------------------------------------------------------------------


class TestMergeThreeSources:
    def test_two_agree_pair_wins(self):
        ds1 = _synthetic_dataset(n=10, rho=5.0)
        ds2 = _synthetic_dataset(n=10, rho=5.5)  # agrees with ds1 (diff=0.5 < 2)
        ds3 = _synthetic_dataset(n=10, rho=20.0)  # outlier
        result = merge({"a": ds1, "b": ds2, "c": ds3})
        # a and b agree, c is outlier. Not all agree (c disagrees with both).
        # Best pair: (a, b) with diff 0.5. Output = mean(5.0, 5.5) = 5.25
        np.testing.assert_allclose(result["rho"].values, 5.25, atol=1e-10)

    def test_all_agree_uses_median(self):
        ds1 = _synthetic_dataset(n=10, rho=5.0)
        ds2 = _synthetic_dataset(n=10, rho=5.5)
        ds3 = _synthetic_dataset(n=10, rho=6.0)  # all within ±2
        result = merge({"a": ds1, "b": ds2, "c": ds3})
        # All pairs agree -> median of [5.0, 5.5, 6.0] = 5.5
        np.testing.assert_allclose(result["rho"].values, 5.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Coupled vectors
# ---------------------------------------------------------------------------


class TestCoupledVectors:
    def test_b_components_share_source(self):
        ds1 = _synthetic_dataset(n=10, bx=5.0)
        ds2 = _synthetic_dataset(n=10, bx=50.0)  # |B| disagrees
        result = merge({"a": ds1, "b": ds2})
        # Same source decision for all B components
        bx_src = result["B_source"].values
        assert all(v == bx_src[0] for v in bx_src if v)

    def test_uyz_share_source(self):
        # Create datasets where Uy values differ but |Vt| still decides
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(10)),
            "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)),
            "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.full(10, 20.0)),
            "Uz": ("time", np.full(10, 10.0)),
            "rho": ("time", np.full(10, 5.0)),
            "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.zeros(10)),
            "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)),
            "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.full(10, 25.0)),
            "Uz": ("time", np.full(10, 12.0)),
            "rho": ("time", np.full(10, 5.0)),
            "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # |Vt|_1 = sqrt(400+100)=22.36, |Vt|_2 = sqrt(625+144)=27.73
        # diff=5.37 < 40 -> agree -> both in source
        assert all(v == "a,b" for v in result["Uyz_source"].values if v)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


class TestTemperature:
    def test_geometric_median_two_sources(self):
        ds1 = _synthetic_dataset(n=10, T=1e5)
        ds2 = _synthetic_dataset(n=10, T=4e5)
        result = merge({"a": ds1, "b": ds2})
        # Geometric median = exp(median(log(1e5), log(4e5))) = exp((log(1e5)+log(4e5))/2)
        expected = np.exp(0.5 * (np.log(1e5) + np.log(4e5)))
        # After median filters, interior values should be close to expected
        np.testing.assert_allclose(result["T"].values[2:-2], expected, rtol=0.01)

    def test_deprioritize_temperature(self):
        ds1 = _synthetic_dataset(n=10, T=1e5)
        ds2 = _synthetic_dataset(n=10, T=5e5)
        result = merge({"good": ds1, "bad": ds2},
                       deprioritize={"T": ["bad"]})
        # With only 2 sources and "bad" deprioritized, should use "good"
        np.testing.assert_allclose(result["T"].values[2:-2], 1e5, rtol=0.01)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


class TestSmoothing:
    def test_smooth_reduces_jump(self):
        idx = pd.date_range("2024-05-01", periods=20, freq="min")
        ux_1 = np.full(20, -400.0)
        ux_2 = np.full(20, np.nan)
        ux_2[10:] = -250.0  # large disagreement starts at minute 10
        ux_1[10:] = np.nan   # source 1 disappears
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(20)), "By": ("time", np.zeros(20)),
            "Bz": ("time", np.zeros(20)), "Ux": ("time", ux_1),
            "Uy": ("time", np.zeros(20)), "Uz": ("time", np.zeros(20)),
            "rho": ("time", np.full(20, 5.0)), "T": ("time", np.full(20, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.zeros(20)), "By": ("time", np.zeros(20)),
            "Bz": ("time", np.zeros(20)), "Ux": ("time", ux_2),
            "Uy": ("time", np.zeros(20)), "Uz": ("time", np.zeros(20)),
            "rho": ("time", np.full(20, 5.0)), "T": ("time", np.full(20, 1e5)),
        }, coords={"time": idx})
        result_smooth = merge({"a": ds1, "b": ds2}, smooth=True)
        result_raw = merge({"a": ds1, "b": ds2}, smooth=False)
        # With smoothing, the transition at minute 10 should be less abrupt
        diff_smooth = abs(float(result_smooth["Ux"].values[10]) -
                         float(result_smooth["Ux"].values[9]))
        diff_raw = abs(float(result_raw["Ux"].values[10]) -
                      float(result_raw["Ux"].values[9]))
        # smoothed jump should be less than or equal to raw
        assert diff_smooth <= diff_raw

    def test_smooth_false_preserves(self):
        ds1 = _synthetic_dataset(n=10, ux=-400.0)
        ds2 = _synthetic_dataset(n=10, ux=-400.0)
        result = merge({"a": ds1, "b": ds2}, smooth=False)
        np.testing.assert_allclose(result["Ux"].values, -400.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


class TestQuality:
    def test_quality_flag_flat_plateau(self):
        idx = pd.date_range("2024-05-01", periods=60, freq="min")
        # Source "bad" has Ux stuck at exactly -400 for 60 minutes (flat plateau)
        ds_bad = xr.Dataset({
            "Bx": ("time", np.zeros(60)), "By": ("time", np.zeros(60)),
            "Bz": ("time", np.zeros(60)),
            "Ux": ("time", np.full(60, -400.0)),
            "Uy": ("time", np.full(60, 0.0)),
            "Uz": ("time", np.full(60, 0.0)),
            "rho": ("time", np.full(60, 5.0)),
            "T": ("time", np.full(60, 1e5)),
        }, coords={"time": idx})
        # Source "good" has Ux with natural variation
        rng = np.random.default_rng(123)
        ds_good = xr.Dataset({
            "Bx": ("time", np.zeros(60)), "By": ("time", np.zeros(60)),
            "Bz": ("time", np.zeros(60)),
            "Ux": ("time", -400.0 + 5.0 * rng.standard_normal(60)),
            "Uy": ("time", 2.0 * rng.standard_normal(60)),
            "Uz": ("time", 2.0 * rng.standard_normal(60)),
            "rho": ("time", 5.0 + 0.5 * rng.standard_normal(60)),
            "T": ("time", np.full(60, 1e5)),
        }, coords={"time": idx})
        # Third source needed for outlier detection
        ds_other = xr.Dataset({
            "Bx": ("time", np.zeros(60)), "By": ("time", np.zeros(60)),
            "Bz": ("time", np.zeros(60)),
            "Ux": ("time", -400.0 + 5.0 * rng.standard_normal(60)),
            "Uy": ("time", 2.0 * rng.standard_normal(60)),
            "Uz": ("time", 2.0 * rng.standard_normal(60)),
            "rho": ("time", 5.0 + 0.5 * rng.standard_normal(60)),
            "T": ("time", np.full(60, 1e5)),
        }, coords={"time": idx})
        result = merge({"bad": ds_bad, "good": ds_good, "other": ds_other},
                       quality=True)
        # The flat-plateau source "bad" should be excluded from Ux in the
        # middle of the time range (plateau detection needs the rolling window)
        mid_sources = result["Ux_source"].values[20:40]
        # At least some should exclude "bad"
        assert any("bad" not in s for s in mid_sources if s)


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


class TestGridAlignment:
    def test_union_of_ranges(self):
        ds1 = _synthetic_dataset(n=30, start="2024-05-01 00:00")
        ds2 = _synthetic_dataset(n=30, start="2024-05-01 00:20")
        result = merge({"a": ds1, "b": ds2})
        # ds1: 00:00 to 00:29, ds2: 00:20 to 00:49
        # Union: 00:00 to 00:49 = 50 minutes
        assert len(result.time) == 50

    def test_non_overlapping_uses_single(self):
        ds1 = _synthetic_dataset(n=10, bx=3.0, start="2024-05-01 00:00")
        ds2 = _synthetic_dataset(n=10, bx=7.0, start="2024-05-01 00:20")
        result = merge({"a": ds1, "b": ds2})
        # Minutes 0-9: only "a" has data -> Bx=3.0
        np.testing.assert_allclose(result["Bx"].values[:10], 3.0, atol=1e-10)
        # Minutes 20-29: only "b" has data -> Bx=7.0
        np.testing.assert_allclose(result["Bx"].values[20:30], 7.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_is_xarray_dataset(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"a": ds})
        assert isinstance(result, xr.Dataset)

    def test_has_time_coord(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"a": ds})
        assert "time" in result.coords

    def test_has_standard_variables(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"a": ds})
        for var in ("Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho", "T"):
            assert var in result.data_vars

    def test_has_provenance_columns(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"a": ds})
        for col in ("B_source", "Ux_source", "Uyz_source", "rho_source", "T_source"):
            assert col in result.data_vars

    def test_has_attrs(self):
        ds = _synthetic_dataset(n=10)
        result = merge({"a": ds})
        assert result.attrs["source"] == "MIDL merge"


# ---------------------------------------------------------------------------
# All-sources-NaN at a timestep
# ---------------------------------------------------------------------------


class TestAllNaN:
    def test_both_nan_produces_nan_and_empty_provenance(self):
        """When all sources have NaN at a timestep, output is NaN with empty provenance."""
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        # Both sources have NaN at minute 5
        bx_1 = np.full(10, 5.0)
        bx_1[5] = np.nan
        bx_2 = np.full(10, 6.0)
        bx_2[5] = np.nan
        ds1 = xr.Dataset({
            "Bx": ("time", bx_1), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", bx_2), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # Bx at minute 5 should be NaN (both sources missing)
        assert np.isnan(result["Bx"].values[5])
        # Provenance at minute 5 should be empty string
        assert result["B_source"].values[5] == ""

    def test_all_variables_nan_at_timestep(self):
        """All variables NaN for both sources at a timestep."""
        idx = pd.date_range("2024-05-01", periods=5, freq="min")
        # Both sources have all-NaN at index 2
        def _make_ds(bx_val):
            arr = np.full(5, bx_val)
            arr[2] = np.nan
            ux = np.full(5, -400.0)
            ux[2] = np.nan
            rho = np.full(5, 5.0)
            rho[2] = np.nan
            T = np.full(5, 1e5)
            T[2] = np.nan
            return xr.Dataset({
                "Bx": ("time", arr), "By": ("time", np.where(np.isnan(arr), np.nan, 0.0)),
                "Bz": ("time", np.where(np.isnan(arr), np.nan, 0.0)),
                "Ux": ("time", ux), "Uy": ("time", np.where(np.isnan(ux), np.nan, 0.0)),
                "Uz": ("time", np.where(np.isnan(ux), np.nan, 0.0)),
                "rho": ("time", rho), "T": ("time", T),
            }, coords={"time": idx})

        result = merge({"a": _make_ds(5.0), "b": _make_ds(6.0)})
        for var in ("Bx", "By", "Bz", "Ux", "Uy", "Uz", "rho"):
            assert np.isnan(result[var].values[2]), f"{var} should be NaN"
        for prov_col in ("B_source", "Ux_source", "Uyz_source", "rho_source"):
            assert result[prov_col].values[2] == "", f"{prov_col} should be empty"


# ---------------------------------------------------------------------------
# Threshold equality boundary
# ---------------------------------------------------------------------------


class TestThresholdBoundary:
    def test_bx_exactly_at_threshold_agrees(self):
        """Two values differing by exactly the threshold (8.0 nT) should agree."""
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        # |B| for ds1 = sqrt(0^2+0^2+0^2) = 0; |B| for ds2 = sqrt(8^2+0^2+0^2) = 8
        # diff = |8.0 - 0.0| = 8.0, threshold = 8.0, uses <=, so they agree
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.full(10, 8.0)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # Both should be in provenance (they agree)
        assert all(v == "a,b" for v in result["B_source"].values if v)
        # Output should be the mean of the two Bx values
        np.testing.assert_allclose(result["Bx"].values, 4.0, atol=1e-10)

    def test_bx_just_above_threshold_disagrees(self):
        """Values differing by slightly more than the threshold should disagree."""
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        # |B| for ds1 = 0, |B| for ds2 = 8.01 -> diff = 8.01 > 8.0 -> disagree
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.full(10, 8.01)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # Only one source should be used (they disagree)
        assert all(v in ("a", "b") for v in result["B_source"].values if v)

    def test_ux_exactly_at_threshold_agrees(self):
        """Ux diff of exactly 80 km/s should be treated as agreeing."""
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -320.0)),
            "Uy": ("time", np.zeros(10)), "Uz": ("time", np.zeros(10)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # Both sources should agree (|-400 - (-320)| = 80 = threshold)
        assert all(v == "a,b" for v in result["Ux_source"].values if v)
        np.testing.assert_allclose(result["Ux"].values, -360.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Ux selects different source from Uy/Uz
# ---------------------------------------------------------------------------


class TestUxUyzDifferentSource:
    def test_ux_and_uyz_provenance_can_differ(self):
        """Ux disagreement can produce a different source than |Vt| coupling for Uy/Uz."""
        idx = pd.date_range("2024-05-01", periods=10, freq="min")
        # Source "a": Ux=-400 (will disagree with "b" on Ux), Uy=10, Uz=5
        # Source "b": Ux=-600 (diff=200 > 80 -> Ux disagrees), Uy=12, Uz=6
        # |Vt| for a = sqrt(100+25) = 11.18, |Vt| for b = sqrt(144+36) = 13.42
        # |Vt| diff = 2.24 < 40 -> Uy/Uz agree -> both in Uyz_source
        # Ux disagrees -> only one source in Ux_source
        ds1 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -400.0)),
            "Uy": ("time", np.full(10, 10.0)), "Uz": ("time", np.full(10, 5.0)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        ds2 = xr.Dataset({
            "Bx": ("time", np.zeros(10)), "By": ("time", np.zeros(10)),
            "Bz": ("time", np.zeros(10)), "Ux": ("time", np.full(10, -600.0)),
            "Uy": ("time", np.full(10, 12.0)), "Uz": ("time", np.full(10, 6.0)),
            "rho": ("time", np.full(10, 5.0)), "T": ("time", np.full(10, 1e5)),
        }, coords={"time": idx})
        result = merge({"a": ds1, "b": ds2})
        # Uyz should use both (|Vt| agrees)
        uyz_sources = result["Uyz_source"].values
        assert all(v == "a,b" for v in uyz_sources if v)
        # Ux should use only one source (Ux disagrees)
        ux_sources = result["Ux_source"].values
        assert all(v in ("a", "b") for v in ux_sources if v)
        # Confirm they differ
        assert ux_sources[0] != uyz_sources[0]


# ---------------------------------------------------------------------------
# Off-minute timestamps raise ValueError
# ---------------------------------------------------------------------------


class TestOffMinuteTimestamps:
    def test_non_minute_aligned_raises(self):
        """Timestamps not aligned to whole minutes should raise ValueError."""
        idx = pd.to_datetime(["2024-05-01 00:00:30", "2024-05-01 00:01:30",
                              "2024-05-01 00:02:30"])
        ds = xr.Dataset({
            "Bx": ("time", [1.0, 2.0, 3.0]), "By": ("time", [0.0, 0.0, 0.0]),
            "Bz": ("time", [0.0, 0.0, 0.0]), "Ux": ("time", [-400.0, -400.0, -400.0]),
            "Uy": ("time", [0.0, 0.0, 0.0]), "Uz": ("time", [0.0, 0.0, 0.0]),
            "rho": ("time", [5.0, 5.0, 5.0]), "T": ("time", [1e5, 1e5, 1e5]),
        }, coords={"time": idx})
        with pytest.raises(ValueError, match="minute|resample"):
            merge({"sat": ds})

    def test_half_second_offset_raises(self):
        """Even a sub-second offset from the minute boundary should raise."""
        idx = pd.to_datetime(["2024-05-01 00:00:00.500"])
        ds = xr.Dataset({
            "Bx": ("time", [1.0]), "By": ("time", [0.0]),
            "Bz": ("time", [0.0]), "Ux": ("time", [-400.0]),
            "Uy": ("time", [0.0]), "Uz": ("time", [0.0]),
            "rho": ("time", [5.0]), "T": ("time", [1e5]),
        }, coords={"time": idx})
        with pytest.raises(ValueError, match="minute|resample"):
            merge({"sat": ds})

    def test_minute_aligned_does_not_raise(self):
        """Minute-aligned timestamps should pass validation without error."""
        ds = _synthetic_dataset(n=5)
        # Should not raise
        result = merge({"sat": ds})
        assert "Bx" in result.data_vars
