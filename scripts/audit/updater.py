from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List

from .models import AuditResult, CatalogImage, InspectInfo


def _normalize_version(value: str) -> str:
    """Extract a compact major.minor version from a string, when possible."""
    match = re.search(r"\d+\.\d+(?:\.\d+)?", value)
    if not match:
        return value
    version = match.group(0)
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version


def _derive_observed(info: InspectInfo, image: CatalogImage) -> Dict[str, Any]:
    """Derive observed versions from env/labels and catalog runtime hints."""
    observed: Dict[str, Any] = dict()

    env = info.env or {}
    labels = info.labels or {}

    def _from_env(*names: str) -> str | None:
        for name in names:
            val = env.get(name)
            if isinstance(val, str) and val:
                return val
        return None

    def _from_labels(*names: str) -> str | None:
        for name in names:
            val = labels.get(name)
            if isinstance(val, str) and val:
                return val
        return None

    py_raw = _from_env("PYTHON_VERSION", "PYTHON")
    if py_raw:
        observed["python"] = _normalize_version(py_raw)

    cuda_raw = _from_env("CUDA_VERSION") or _from_labels(
        "com.nvidia.cuda.version",
        "nvidia.com/cuda.version",
    )
    if cuda_raw:
        observed["cuda"] = _normalize_version(cuda_raw)

    cudnn_raw = _from_env("CUDNN_VERSION") or _from_labels(
        "com.nvidia.cudnn.version",
        "nvidia.com/cudnn.version",
    )
    if cudnn_raw:
        observed["cudnn"] = _normalize_version(cudnn_raw)

    runtime = image.get("runtime") or {}
    os_info = runtime.get("os") or {}
    os_name = os_info.get("name")
    os_version = os_info.get("version")
    if os_name:
        observed.setdefault("os_name", os_name)
    if os_version:
        observed.setdefault("os_version", os_version)

    return observed


def apply_audit_to_image(
    image: CatalogImage,
    audit: AuditResult,
    *,
    today: date,
) -> None:
    """Apply a single digest-level AuditResult to a catalog image in-place."""
    existing_audit = image.get("audit") or {}
    if not isinstance(existing_audit, dict):
        existing_audit = {}

    audit_block: Dict[str, Any] = dict(existing_audit)

    # Always update digest and last-seen timestamp.
    audit_block["digest"] = audit.digest
    audit_block["digest_last_seen"] = today.isoformat()
    audit_block["verify_method"] = "skopeo-inspect"
    audit_block["verified_at"] = today.isoformat()

    observed = dict(existing_audit.get("observed") or {})
    observed.update(_derive_observed(audit.inspect, image))
    if observed:
        audit_block["observed"] = observed

    # Preserve any existing notes while still allowing manual edits later.
    if "notes" not in audit_block:
        audit_block["notes"] = None

    # Only update artifact refs when we have new values; otherwise leave
    # existing refs intact so previous scans remain discoverable.
    if audit.trivy_report_ref is not None:
        audit_block["trivy_report_ref"] = audit.trivy_report_ref
    if audit.packages_ref is not None:
        audit_block["packages_ref"] = audit.packages_ref

    image["audit"] = audit_block

    # Apply security summary if available. If audit.security is None,
    # leave any existing security block untouched.
    if audit.security is not None:
        sec = audit.security
        image["security"] = {
            "total_cves": sec.total_cves,
            "critical": sec.critical,
            "high": sec.high,
            "medium": sec.medium,
            "low": sec.low,
            "rating": sec.rating,
            "last_scan": sec.last_scan,
            "scanner": sec.scanner,
        }


def apply_results_to_catalog(
    catalog: Dict[str, Any],
    results_by_digest: Dict[str, AuditResult],
    images_by_digest: Dict[str, List[int]],
    *,
    today: date,
) -> Dict[str, Any]:
    """Apply AuditResults to all matching images in the catalog.

    Args:
        catalog: Full catalog document (mutated in-place).
        results_by_digest: Mapping from digest to AuditResult.
        images_by_digest: Mapping from digest to list of image indexes in
            catalog["images"] that should receive that result.
        today: Current date, used for digest_last_seen and verified_at.
    """
    images: List[CatalogImage] = catalog.get("images") or []
    for digest, result in results_by_digest.items():
        indexes = images_by_digest.get(digest, [])
        for idx in indexes:
            if 0 <= idx < len(images):
                apply_audit_to_image(images[idx], result, today=today)
    return catalog