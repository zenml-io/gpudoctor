# Registry fetchers for Docker Hub, GHCR, NGC
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TagInfo:
    """Common tag metadata returned by all fetchers."""

    name: str  # Tag name (e.g., "2.5.1-cuda12.4-cudnn9-runtime")
    last_updated: str  # ISO datetime string from registry
    architectures: list[str]  # e.g., ["amd64", "arm64"]
    compressed_size_mb: int  # Best-effort size in MB
