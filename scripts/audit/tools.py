from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .models import ImageRef

logger = logging.getLogger(__name__)


def ensure_audit_dirs(audit_root: Path) -> tuple[Path, Path, Path]:
    """Ensure audit subdirectories exist and return their paths.

    Returns:
        (inspect_dir, trivy_dir, packages_dir)
    """
    audit_root.mkdir(parents=True, exist_ok=True)
    inspect_dir = audit_root / "inspect"
    trivy_dir = audit_root / "trivy"
    packages_dir = audit_root / "packages"

    for d in (inspect_dir, trivy_dir, packages_dir):
        d.mkdir(parents=True, exist_ok=True)

    return inspect_dir, trivy_dir, packages_dir


def run_skopeo_inspect(image: ImageRef) -> dict[str, Any]:
    """Run `skopeo inspect` for an image and return the parsed JSON.

    Raises:
        RuntimeError on failure.
    """
    target = f"docker://{image.canonical()}"
    cmd = ["skopeo", "inspect", target]

    logger.info("Inspecting image with skopeo: %s", image.canonical())
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "skopeo inspect failed for %s: %s",
            image.canonical(),
            exc.stderr.strip() if exc.stderr else exc,
        )
        raise RuntimeError(f"skopeo inspect failed for {image.canonical()}") from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode skopeo inspect output for %s: %s", image.canonical(), exc)
        raise RuntimeError(f"Invalid JSON from skopeo inspect for {image.canonical()}") from exc


def run_trivy_scan(image: ImageRef, digest: str, arch: str) -> dict[str, Any]:
    """Run `trivy image` in JSON mode for a given digest and architecture.

    The image is referenced by digest to ensure scans are digest-stable.
    """
    # Prefer digest-based reference so that scans are tied to immutable content.
    target = f"{image.host}/{image.repository}@{digest}"
    platform = f"linux/{arch}"
    cmd = [
        "trivy",
        "image",
        "--quiet",
        "--format",
        "json",
        "--severity",
        "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL",
        "--platform",
        platform,
        target,
    ]

    logger.info("Running Trivy scan for %s (%s)", target, platform)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Trivy scan failed for %s: %s",
            target,
            exc.stderr.strip() if exc.stderr else exc,
        )
        raise RuntimeError(f"Trivy scan failed for {target}") from exc

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode Trivy JSON output for %s: %s", target, exc)
        raise RuntimeError(f"Invalid JSON from Trivy for {target}") from exc