from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from .models import CatalogImage


@dataclass
class Selection:
    """Represents a catalog image chosen for auditing."""

    index: int
    image_id: str
    reason: str
    image: CatalogImage


def _parse_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _matches_filters(image: CatalogImage, filters: Dict[str, str] | None) -> bool:
    if not filters:
        return True

    metadata = image.get("metadata") or {}
    for key, expected in filters.items():
        if key == "id":
            if str(image.get("id")) != expected:
                return False
        elif key in ("provider", "registry", "status"):
            if str(metadata.get(key)) != expected:
                return False
        else:
            # Unknown filter key – ignore it for now rather than failing the run.
            continue
    return True


def select_images(
    images: List[CatalogImage],
    *,
    arch: str = "amd64",
    max_images: int | None = None,
    filters: Dict[str, str] | None = None,
) -> List[Selection]:
    """Select images for auditing, ordered by priority.

    Priority rules (lower number = earlier in list):
      1. security block is missing
      2. audit block is missing or lacks a digest
      3. security.last_scan is present; older scans come first

    TTL and per-digest skip logic are handled later via AuditCache; this
    selector only orders candidate images.
    """
    today = date.today()
    candidates: List[tuple[int, date, Selection]] = []

    for idx, image in enumerate(images):
        runtime = image.get("runtime") or {}
        architectures = runtime.get("architectures") or []
        if arch not in architectures:
            continue

        if not _matches_filters(image, filters):
            continue

        image_id = str(image.get("id"))
        security = image.get("security")
        audit = image.get("audit")

        if security is None:
            priority = 1
            reason = "missing security"
            last_scan_date: date | None = None
        else:
            sec_last_scan = _parse_date((security or {}).get("last_scan"))
            if audit is None or not isinstance(audit, dict) or not audit.get("digest"):
                priority = 2
                reason = "missing audit digest"
                last_scan_date = sec_last_scan
            else:
                priority = 3
                reason = "stale or existing security"
                last_scan_date = sec_last_scan

        candidates.append(
            (
                priority,
                last_scan_date or today,
                Selection(index=idx, image_id=image_id, reason=reason, image=image),
            )
        )

    # Sort by priority then last_scan (oldest first), then image_id for stability.
    candidates.sort(key=lambda item: (item[0], item[1], item[2].image_id))

    if max_images is not None and max_images >= 0:
        candidates = candidates[: max_images]

    return [sel for _, _, sel in candidates]