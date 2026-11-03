"""Configuration loading and repository path resolution.

All paths are resolved relative to the repository root so that the pipeline can
be started from any working directory (the jury runs ``python -m tsfdi ...``).
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """Repository root: the directory that contains ``config/`` and ``src/``."""
    env = os.environ.get("TSFDI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "sites.yaml").is_file():
            return parent
    raise RuntimeError("cannot locate repository root (config/sites.yaml not found)")


def path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def data_dir() -> Path:
    return path("data", "bundle")


def results_dir() -> Path:
    d = path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = path("results", "cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# YAML configuration
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    with open(path("config", name), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load(name: str) -> dict[str, Any]:
    """Load ``config/<name>`` (a cached, read-only dict)."""
    return _load_yaml(name)


def reload_all() -> None:
    _load_yaml.cache_clear()


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    lat: float
    lon: float
    elevation_m: float
    climate: str
    variables: tuple[str, ...]

    @property
    def slug(self) -> str:
        return self.name.lower()


def sites() -> list[Site]:
    out = []
    for s in load("sites.yaml")["sites"]:
        out.append(
            Site(
                id=s["id"],
                name=s["name"],
                lat=float(s["lat"]),
                lon=float(s["lon"]),
                elevation_m=float(s["elevation_m"]),
                climate=s["climate"],
                variables=tuple(s["variables"]),
            )
        )
    return out


def site(name_or_id: str) -> Site:
    key = name_or_id.upper()
    for s in sites():
        if s.id.upper() == key or s.name.upper() == key:
            return s
    raise KeyError(f"unknown site {name_or_id!r}")


def variable_spec(var: str) -> dict[str, Any]:
    return load("sites.yaml")["variables"][var]


def period(which: str = "calibration") -> tuple[str, str]:
    p = load("sites.yaml")["period"][which]
    return str(p[0]), str(p[1])


SEED = 20261028  # master seed; every stochastic component derives from it (config/attacks.yaml "seed")
