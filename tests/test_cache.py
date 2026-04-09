"""Tests for midl._cache."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from midl._cache import csv_url, ensure_cached, resolve_target


class TestResolveTarget:
    def test_lowercase(self):
        assert resolve_target("32re") == "32Re"

    def test_mixed_case(self):
        assert resolve_target("14Re") == "14Re"

    def test_uppercase(self):
        assert resolve_target("L1") == "L1"

    def test_all_lower_l1(self):
        assert resolve_target("l1") == "L1"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown target"):
            resolve_target("invalid")


class TestCsvUrl:
    def test_32re(self):
        url = csv_url("2015-03", "32Re")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2015/03/201503_32Re.csv"

    def test_l1(self):
        url = csv_url("2024-01", "L1")
        assert url == "https://csem.engin.umich.edu/MIDL/data/2024/01/202401_L1.csv"


class TestEnsureCached:
    def test_downloads_on_miss(self, tmp_path):
        csv_content = b"timestamp,Bx\n2024-03-01T00:00:00,1.23\n"

        mock_resp = MagicMock()
        mock_resp.content = csv_content
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("midl._cache.cache_dir", return_value=tmp_path),
            patch("midl._cache.requests.get", return_value=mock_resp) as mock_get,
        ):
            path = ensure_cached("2024-03", "32Re")
            assert path.exists()
            assert path.read_bytes() == csv_content
            mock_get.assert_called_once()

    def test_serves_from_cache(self, tmp_path):
        cached_file = tmp_path / "202403_32Re.csv"
        cached_file.write_text("cached content")

        with (
            patch("midl._cache.cache_dir", return_value=tmp_path),
            patch("midl._cache.requests.get") as mock_get,
        ):
            path = ensure_cached("2024-03", "32Re")
            assert path == cached_file
            mock_get.assert_not_called()
