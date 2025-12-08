"""Docker Hub API client for fetching image tag metadata."""

from __future__ import annotations

import logging
from typing import Iterator

import httpx

from . import TagInfo

logger = logging.getLogger(__name__)

# Docker Hub API base URL
DOCKER_HUB_API = "https://hub.docker.com/v2"

# Request timeout in seconds
TIMEOUT = 30.0


def iter_tags(
    namespace: str,
    repo: str,
    page_size: int = 100,
    max_pages: int = 5,
) -> Iterator[TagInfo]:
    """Iterate over all tags for a Docker Hub repository.

    Args:
        namespace: Docker Hub namespace (e.g., "pytorch")
        repo: Repository name (e.g., "pytorch")
        page_size: Number of tags per API request
        max_pages: Maximum number of pages to fetch

    Yields:
        TagInfo for each tag found
    """
    url = f"{DOCKER_HUB_API}/repositories/{namespace}/{repo}/tags"
    params = {"page_size": page_size, "page": 1}

    with httpx.Client(timeout=TIMEOUT) as client:
        for page in range(1, max_pages + 1):
            params["page"] = page
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                logger.warning(
                    "Failed to fetch tags for %s/%s page %d: %s",
                    namespace,
                    repo,
                    page,
                    e,
                )
                break

            results = data.get("results", [])
            if not results:
                break

            for result in results:
                tag_info = _parse_tag_result(result)
                if tag_info:
                    yield tag_info

            # Check if there are more pages
            if not data.get("next"):
                break


def get_tag(namespace: str, repo: str, tag: str) -> TagInfo | None:
    """Fetch metadata for a specific tag.

    Args:
        namespace: Docker Hub namespace
        repo: Repository name
        tag: Tag name

    Returns:
        TagInfo if found, None otherwise
    """
    url = f"{DOCKER_HUB_API}/repositories/{namespace}/{repo}/tags/{tag}"

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            return _parse_tag_result(data)
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch tag %s/%s:%s: %s", namespace, repo, tag, e)
        return None


def _parse_tag_result(result: dict) -> TagInfo | None:
    """Parse a Docker Hub tag API result into TagInfo.

    Args:
        result: Raw API response for a tag

    Returns:
        TagInfo or None if parsing fails
    """
    try:
        name = result["name"]
        last_updated = result.get("last_updated", "")

        # Extract architectures from images array, filtering to valid values
        images = result.get("images", [])
        valid_archs = {"amd64", "arm64"}  # Schema-allowed values
        architectures = sorted(
            {img.get("architecture") for img in images if img.get("architecture")}
            & valid_archs
        )

        # Get compressed size - prefer amd64, fall back to first available
        compressed_size_bytes = 0
        for img in images:
            if img.get("architecture") == "amd64":
                compressed_size_bytes = img.get("size", 0)
                break
        if not compressed_size_bytes and images:
            compressed_size_bytes = images[0].get("size", 0)

        # Convert bytes to MB
        compressed_size_mb = compressed_size_bytes // (1024 * 1024)

        return TagInfo(
            name=name,
            last_updated=last_updated,
            architectures=sorted(architectures),
            compressed_size_mb=compressed_size_mb,
        )
    except (KeyError, TypeError) as e:
        logger.debug("Failed to parse tag result: %s", e)
        return None


def get_available_tags(
    namespace: str,
    repo: str,
    limit: int = 50,
) -> list[str]:
    """Get a list of available tag names for a repository.

    Useful for discovery without full metadata.

    Args:
        namespace: Docker Hub namespace
        repo: Repository name
        limit: Maximum number of tags to return

    Returns:
        List of tag names
    """
    tags = []
    for tag_info in iter_tags(namespace, repo, page_size=min(limit, 100)):
        tags.append(tag_info.name)
        if len(tags) >= limit:
            break
    return tags
