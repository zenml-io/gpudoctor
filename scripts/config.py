"""Configuration loader for tracked images."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Pattern

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRACKED_IMAGES_PATH = ROOT / "data" / "tracked_images.yaml"
IMAGES_PATH = ROOT / "data" / "images.json"
SCHEMA_PATH = ROOT / "data" / "schema.json"


@dataclass
class DiscoverConfig:
    """Configuration for auto-discovering new tags."""

    pattern: Pattern[str]
    limit: int = 10


@dataclass
class DockerHubSeed:
    """A Docker Hub image family to track."""

    id: str
    namespace: str
    repo: str
    parser: str
    tags: list[str] = field(default_factory=list)
    discover: DiscoverConfig | None = None


@dataclass
class GHCRSeed:
    """A GitHub Container Registry image family to track."""

    id: str
    org: str
    repo: str
    parser: str
    tags: list[str] = field(default_factory=list)


@dataclass
class NGCSeed:
    """An NVIDIA NGC image family to track."""

    id: str
    org: str
    repo: str
    parser: str
    tags: list[str] = field(default_factory=list)
    team: str | None = None  # For nested repos like nvidia/rapidsai/base


@dataclass
class QuaySeed:
    """A Quay.io image family to track."""

    id: str
    org: str
    repo: str
    parser: str
    tags: list[str] = field(default_factory=list)


@dataclass
class TrackedImagesConfig:
    """Full configuration from tracked_images.yaml."""

    dockerhub: list[DockerHubSeed] = field(default_factory=list)
    ghcr: list[GHCRSeed] = field(default_factory=list)
    ngc: list[NGCSeed] = field(default_factory=list)
    quay: list[QuaySeed] = field(default_factory=list)


def _parse_discover(discover_cfg: dict | None) -> DiscoverConfig | None:
    """Parse discover configuration block."""
    if not discover_cfg:
        return None
    pattern = discover_cfg.get("pattern")
    if not pattern:
        return None
    return DiscoverConfig(
        pattern=re.compile(pattern),
        limit=discover_cfg.get("limit", 10),
    )


def load_config(config_path: Path | None = None) -> TrackedImagesConfig:
    """Load and parse the tracked images configuration.

    Args:
        config_path: Optional path to config file. Defaults to data/tracked_images.yaml.

    Returns:
        TrackedImagesConfig with all parsed seeds.
    """
    path = config_path or TRACKED_IMAGES_PATH
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}

    # Parse Docker Hub seeds
    dockerhub_seeds: list[DockerHubSeed] = []
    for entry in cfg.get("dockerhub", []):
        dockerhub_seeds.append(
            DockerHubSeed(
                id=entry["id"],
                namespace=entry["namespace"],
                repo=entry["repo"],
                parser=entry["parser"],
                tags=entry.get("tags", []),
                discover=_parse_discover(entry.get("discover")),
            )
        )

    # Parse GHCR seeds
    ghcr_seeds: list[GHCRSeed] = []
    for entry in cfg.get("ghcr", []):
        ghcr_seeds.append(
            GHCRSeed(
                id=entry["id"],
                org=entry["org"],
                repo=entry["repo"],
                parser=entry["parser"],
                tags=entry.get("tags", []),
            )
        )

    # Parse NGC seeds
    ngc_seeds: list[NGCSeed] = []
    for entry in cfg.get("ngc", []):
        ngc_seeds.append(
            NGCSeed(
                id=entry["id"],
                org=entry["org"],
                repo=entry["repo"],
                parser=entry["parser"],
                tags=entry.get("tags", []),
                team=entry.get("team"),
            )
        )

    # Parse Quay.io seeds
    quay_seeds: list[QuaySeed] = []
    for entry in cfg.get("quay", []):
        quay_seeds.append(
            QuaySeed(
                id=entry["id"],
                org=entry["org"],
                repo=entry["repo"],
                parser=entry["parser"],
                tags=entry.get("tags", []),
            )
        )

    return TrackedImagesConfig(
        dockerhub=dockerhub_seeds,
        ghcr=ghcr_seeds,
        ngc=ngc_seeds,
        quay=quay_seeds,
    )


def load_dockerhub_seeds(config_path: Path | None = None) -> list[DockerHubSeed]:
    """Convenience function to load only Docker Hub seeds."""
    return load_config(config_path).dockerhub


def load_ghcr_seeds(config_path: Path | None = None) -> list[GHCRSeed]:
    """Convenience function to load only GHCR seeds."""
    return load_config(config_path).ghcr


def load_ngc_seeds(config_path: Path | None = None) -> list[NGCSeed]:
    """Convenience function to load only NGC seeds."""
    return load_config(config_path).ngc


def load_quay_seeds(config_path: Path | None = None) -> list[QuaySeed]:
    """Convenience function to load only Quay.io seeds."""
    return load_config(config_path).quay
