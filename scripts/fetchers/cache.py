"""Local cache for registry tag metadata to avoid redundant API calls.

The cache stores successful TagInfo fetches in a JSON file with daily expiry.
Cache entries are keyed by registry:namespace/org:repo:tag.

Usage:
    from fetchers.cache import TagCache

    cache = TagCache()  # loads from default location

    # Check cache before fetching
    tag_info = cache.get("dockerhub", "pytorch", "pytorch", "2.1.0-cuda12.1")
    if tag_info is None:
        tag_info = dockerhub_client.get_tag("pytorch", "pytorch", "2.1.0-cuda12.1")
        if tag_info:
            cache.set("dockerhub", "pytorch", "pytorch", "2.1.0-cuda12.1", tag_info)

    # Save cache when done
    cache.save()
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict

from . import TagInfo

logger = logging.getLogger(__name__)

# Default cache location alongside the data files
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / ".tag_cache.json"


class TagCache:
    """In-memory cache for TagInfo objects with JSON persistence and daily expiry."""

    def __init__(self, cache_path: Path | None = None, enabled: bool = True):
        """Initialize the cache.

        Args:
            cache_path: Path to cache file. Defaults to data/.tag_cache.json.
            enabled: If False, all cache operations are no-ops.
        """
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.enabled = enabled
        self._data: Dict[str, Any] = {}
        self._dirty = False

        if self.enabled:
            self._load()

    def _make_key(self, registry: str, namespace: str, repo: str, tag: str) -> str:
        """Create a unique cache key for a tag."""
        return f"{registry}:{namespace}:{repo}:{tag}"

    def _load(self) -> None:
        """Load cache from disk if it exists and is from today."""
        if not self.cache_path.exists():
            logger.debug("No cache file found at %s", self.cache_path)
            self._data = {"date": date.today().isoformat(), "entries": {}}
            return

        try:
            with self.cache_path.open() as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load cache from %s: %s", self.cache_path, e)
            self._data = {"date": date.today().isoformat(), "entries": {}}
            return

        cached_date = self._data.get("date")
        today = date.today().isoformat()

        if cached_date != today:
            logger.info(
                "Cache expired (cached: %s, today: %s), starting fresh",
                cached_date,
                today,
            )
            self._data = {"date": today, "entries": {}}
        else:
            entry_count = len(self._data.get("entries", {}))
            logger.info("Loaded %d cached tag entries from %s", entry_count, self.cache_path)

    def get(self, registry: str, namespace: str, repo: str, tag: str) -> TagInfo | None:
        """Retrieve a cached TagInfo if available.

        Returns None if not cached, cache disabled, or cache expired.
        """
        if not self.enabled:
            return None

        key = self._make_key(registry, namespace, repo, tag)
        entry = self._data.get("entries", {}).get(key)

        if entry is None:
            return None

        try:
            return TagInfo(
                name=entry["name"],
                last_updated=entry["last_updated"],
                architectures=entry["architectures"],
                compressed_size_mb=entry["compressed_size_mb"],
            )
        except (KeyError, TypeError) as e:
            logger.debug("Invalid cache entry for %s: %s", key, e)
            return None

    def set(
        self,
        registry: str,
        namespace: str,
        repo: str,
        tag: str,
        tag_info: TagInfo,
    ) -> None:
        """Store a TagInfo in the cache."""
        if not self.enabled:
            return

        key = self._make_key(registry, namespace, repo, tag)
        self._data.setdefault("entries", {})[key] = asdict(tag_info)
        self._dirty = True

    def save(self) -> None:
        """Persist the cache to disk if it has been modified."""
        if not self.enabled or not self._dirty:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
                f.write("\n")
            logger.info(
                "Saved %d cached tag entries to %s",
                len(self._data.get("entries", {})),
                self.cache_path,
            )
            self._dirty = False
        except OSError as e:
            logger.warning("Failed to save cache to %s: %s", self.cache_path, e)

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        entries = self._data.get("entries", {})
        by_registry: Dict[str, int] = {}
        for key in entries:
            registry = key.split(":")[0]
            by_registry[registry] = by_registry.get(registry, 0) + 1
        return {
            "total": len(entries),
            "by_registry": by_registry,
        }
