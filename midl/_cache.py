"""Download and local file-cache layer."""

from __future__ import annotations

from pathlib import Path

import platformdirs
import requests

BASE_URL = "https://csem.engin.umich.edu/MIDL/data"

TARGETS: dict[str, str] = {
    "14re": "14Re",
    "32re": "32Re",
    "l1": "L1",
}


def resolve_target(target: str) -> str:
    """Normalize a case-insensitive target string to its canonical form."""
    canonical = TARGETS.get(target.lower())
    if canonical is None:
        valid = ", ".join(TARGETS.values())
        raise ValueError(f"Unknown target {target!r}. Valid targets: {valid}")
    return canonical


def csv_url(year_month: str, target: str) -> str:
    """Build the full URL for a monthly CSV file.

    Parameters
    ----------
    year_month : str
        ``"YYYY-MM"`` string.
    target : str
        Canonical target name (``"14Re"``, ``"32Re"``, or ``"L1"``).
    """
    year, month = year_month.split("-")
    return f"{BASE_URL}/{year}/{month}/{year}{month}_{target}.csv"


def cache_dir() -> Path:
    """Return (and create) the local cache directory."""
    d = Path(platformdirs.user_cache_dir("midl"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_cached(year_month: str, target: str) -> Path:
    """Return a local path to the CSV, downloading it if not already cached."""
    year, month = year_month.split("-")
    filename = f"{year}{month}_{target}.csv"
    path = cache_dir() / filename

    if path.exists():
        return path

    url = csv_url(year_month, target)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path
