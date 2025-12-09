from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from .models import InspectInfo, SecuritySummary

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_ROOT = ROOT / "data" / ".audit"


class AuditCache:
    """Persistent digest-level cache for audit results.

    The cache tracks per-digest inspection metadata, security summaries, and
    failure backoff state. Large artifacts (inspect JSON, Trivy reports,
    package inventories) are stored on disk under data/.audit/, while this
    index file keeps small pointers and timestamps.
    """

    def __init__(self, audit_root: Path | None = None, enabled: bool = True):
        self.audit_root = audit_root or DEFAULT_AUDIT_ROOT
        self.enabled = enabled

        self.index_path = self.audit_root / "index.json"
        self.inspect_dir = self.audit_root / "inspect"
        self.trivy_dir = self.audit_root / "trivy"
        self.packages_dir = self.audit_root / "packages"

        self._data: Dict[str, Any] = {}
        self._dirty = False

        self._ensure_dirs()
        if self.enabled:
            self._load()
        else:
            # Keep an in-memory empty structure so helpers still work.
            self._data = {"version": 1, "entries": {}}

    # directory helpers

    def _ensure_dirs(self) -> None:
        for d in (self.audit_root, self.inspect_dir, self.trivy_dir, self.packages_dir):
            d.mkdir(parents=True, exist_ok=True)

    # index load/save

    def _load(self) -> None:
        if not self.index_path.exists():
            self._data = {"version": 1, "entries": {}}
            logger.debug("No audit index found at %s, starting fresh", self.index_path)
            return

        try:
            with self.index_path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load audit index from %s: %s", self.index_path, exc)
            self._data = {"version": 1, "entries": {}}
            return

        if not isinstance(data, dict):
            self._data = {"version": 1, "entries": {}}
            logger.warning("Audit index at %s was not a JSON object; resetting", self.index_path)
            return

        data.setdefault("version", 1)
        data.setdefault("entries", {})
        self._data = data
        logger.info(
            "Loaded %d audit entries from %s",
            len(self._data.get("entries", {})),
            self.index_path,
        )

    def save(self) -> None:
        """Persist the audit index to disk if modified."""
        if not self.enabled or not self._dirty:
            return

        payload = dict(self._data)
        payload.setdefault("version", 1)
        payload.setdefault("entries", {})

        try:
            with self.index_path.open("w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            logger.info(
                "Saved %d audit entries to %s",
                len(payload.get("entries", {})),
                self.index_path,
            )
            self._dirty = False
        except OSError as exc:
            logger.warning("Failed to save audit index to %s: %s", self.index_path, exc)

    # core helpers

    def _entries(self) -> Dict[str, Any]:
        return self._data.setdefault("entries", {})

    def get_entry(self, digest: str) -> Optional[Dict[str, Any]]:
        """Return the cached entry for a digest, if any."""
        return self._entries().get(digest)

    def _ensure_entry(self, digest: str) -> Dict[str, Any]:
        entries = self._entries()
        if digest not in entries:
            entries[digest] = {
                "digest": digest,
                "failures": 0,
                "skip_until": None,
                "skip_reason": None,
            }
        return entries[digest]

    # artifact path helpers

    @staticmethod
    def _digest_to_filename(digest: str) -> str:
        # Normalise 'sha256:abc...' to 'sha256-abc...'
        return digest.replace(":", "-")

    def inspect_path_for(self, digest: str) -> Path:
        return self.inspect_dir / f"{self._digest_to_filename(digest)}.json"

    def trivy_report_path_for(self, digest: str) -> Path:
        return self.trivy_dir / f"{self._digest_to_filename(digest)}.json"

    def packages_path_for(self, digest: str) -> Path:
        return self.packages_dir / f"{self._digest_to_filename(digest)}.json.gz"

    @staticmethod
    def relative_ref(path: Path) -> str:
        """Return a repository-relative path for storing in the catalog."""
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            # Fall back to a string path if not under the repo root.
            return str(path)

    # audit metadata updates

    def update_inspect(self, digest: str, info: InspectInfo, inspect_ref: str) -> None:
        """Record inspection metadata for a digest."""
        if not self.enabled:
            return

        entry = self._ensure_entry(digest)
        entry["inspect"] = {
            "inspected_at": info.inspected_at,
            "inspect_ref": inspect_ref,
            "architecture": info.architecture,
            "env_keys": sorted(info.env.keys()),
            "label_keys": sorted(info.labels.keys()),
        }
        self._dirty = True

    def update_security(
        self,
        digest: str,
        summary: SecuritySummary,
        trivy_report_ref: Optional[str],
        packages_ref: Optional[str],
    ) -> None:
        """Record security scan metadata for a digest."""
        if not self.enabled:
            return

        entry = self._ensure_entry(digest)
        entry["security"] = {
            "last_security_scan": summary.last_scan,
            "scanner": summary.scanner,
            "total_cves": summary.total_cves,
            "critical": summary.critical,
            "high": summary.high,
            "medium": summary.medium,
            "low": summary.low,
            "rating": summary.rating,
            "trivy_report_ref": trivy_report_ref,
            "packages_ref": packages_ref,
        }
        # Reset failure/backoff state on success
        entry["failures"] = 0
        entry["skip_until"] = None
        entry["skip_reason"] = None
        self._dirty = True

    def register_failure(self, digest: str, reason: str, today: date) -> None:
        """Record a failed scan attempt and update backoff skip markers."""
        if not self.enabled:
            return

        entry = self._ensure_entry(digest)
        failures = int(entry.get("failures", 0)) + 1
        entry["failures"] = failures
        entry["skip_reason"] = reason

        # Simple backoff schedule: 1, 3, 7 days for subsequent failures.
        if failures == 1:
            days = 1
        elif failures == 2:
            days = 3
        else:
            days = 7

        skip_until = today + timedelta(days=days)
        entry["skip_until"] = skip_until.isoformat()
        self._dirty = True

    def should_run_security_scan(
        self,
        digest: str,
        today: date,
        ttl_days: int,
        force: bool,
    ) -> bool:
        """Return True if a new security scan should be run for this digest.

        This considers:
          - --force flag
          - skip_until backoff markers
          - last_security_scan timestamp vs TTL
        """
        if not self.enabled:
            return True
        if force:
            return True

        entry = self.get_entry(digest)
        if not entry:
            return True

        # Backoff on repeated failures
        skip_until_str = entry.get("skip_until")
        if isinstance(skip_until_str, str):
            try:
                skip_until = date.fromisoformat(skip_until_str)
            except ValueError:
                skip_until = None
            if skip_until and skip_until >= today:
                return False

        security = entry.get("security")
        if not isinstance(security, dict):
            return True

        last_scan_str = security.get("last_security_scan")
        if not isinstance(last_scan_str, str):
            return True

        try:
            last_scan = date.fromisoformat(last_scan_str)
        except ValueError:
            return True

        age_days = (today - last_scan).days
        if age_days < 0:
            # Clock skew or bad data – treat as fresh to avoid hammering scanners.
            return False

        return age_days >= ttl_days