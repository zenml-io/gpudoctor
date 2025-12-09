from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ImageRef:
    """Canonical reference for a container image from the catalog.

    `raw` is taken from the catalog's `name` field, while `registry` is the
    logical registry identifier from `metadata.registry` (dockerhub, ghcr, ngc, ...).

    `host` and `repository` are derived from `raw` plus the registry, and
    `canonical()` returns the fully qualified image reference suitable for
    CLI tools like skopeo and trivy.
    """

    raw: str
    registry: str
    host: str
    repository: str
    tag: str

    def canonical(self) -> str:
        """Return a fully qualified image reference: host/repository:tag."""
        return f"{self.host}/{self.repository}:{self.tag}"


@dataclass
class InspectInfo:
    """Subset of image inspection metadata used for auditing."""

    digest: str
    env: dict[str, str]
    labels: dict[str, str]
    architecture: Optional[str]
    inspected_at: str  # ISO 8601 datetime


@dataclass
class SecuritySummary:
    """Normalized security summary derived from a vulnerability scan."""

    total_cves: int
    critical: int
    high: int
    medium: int
    low: int
    rating: Literal["A", "B", "C", "D", "F"]
    last_scan: str  # YYYY-MM-DD
    scanner: Literal["trivy"]


@dataclass
class AuditResult:
    """Combined result of inspection + optional security scan for a digest."""

    digest: str
    inspect: InspectInfo
    security: Optional[SecuritySummary]
    trivy_report_ref: Optional[str]
    packages_ref: Optional[str]


CatalogImage = dict[str, Any]


def image_to_ref(image: CatalogImage) -> ImageRef:
    """Convert a catalog image dict into an ImageRef.

    The catalog stores:
      - image["name"] as either:
          * "namespace/repo:tag" for Docker Hub
          * "host/namespace/repo:tag" for other registries
      - image["metadata"]["registry"] as a logical registry id.

    This helper normalizes those into a host/repository:tag reference.
    """
    name = image.get("name")
    if not isinstance(name, str) or ":" not in name:
        raise ValueError(f"Image is missing a valid 'name' field: {name!r}")

    repo_part, tag = name.rsplit(":", 1)
    metadata = image.get("metadata") or {}
    registry = metadata.get("registry", "dockerhub")

    # Map logical registry identifiers to default hosts.
    default_hosts: dict[str, str] = {
        "dockerhub": "docker.io",
        "ghcr": "ghcr.io",
        "ngc": "nvcr.io",
        "ecr": "public.ecr.aws",
        "gcr": "gcr.io",
        "quay": "quay.io",
        "mcr": "mcr.microsoft.com",
    }

    host: str
    repository: str

    # If repo_part already includes a hostname (e.g., ghcr.io/owner/repo),
    # keep it and split host vs repository. This is how non-Docker Hub
    # builders currently populate image["name"].
    first_segment, _, rest = repo_part.partition("/")
    if "." in first_segment or ":" in first_segment:
        host = first_segment
        repository = rest or ""
    else:
        # Docker Hub and similar cases: repo_part is "namespace/repo".
        host = default_hosts.get(registry, "docker.io")
        repository = repo_part

    return ImageRef(
        raw=name,
        registry=registry,
        host=host,
        repository=repository,
        tag=tag,
    )