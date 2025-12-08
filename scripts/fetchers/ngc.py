"""NVIDIA NGC API client for fetching image tag metadata.

This client uses two sources of information:

1. The NGC REST API at https://api.ngc.nvidia.com/v2 to obtain high-level
   tag metadata such as the last update time (when available).
2. The OCI registry API exposed at nvcr.io to retrieve the image manifest
   and config, which are used to determine architectures and compressed size.

Public images can be queried without credentials. For private images, an
NGC API key may be supplied via the NGC_API_KEY environment variable or the
api_key argument to get_tag().
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import httpx

from . import TagInfo

logger = logging.getLogger(__name__)

NVCR_REGISTRY = "https://nvcr.io"
NGC_API_BASE = "https://api.ngc.nvidia.com/v2"
TIMEOUT = 30.0

# Accept both OCI and Docker manifest formats, including multi-arch indexes.
ACCEPT_MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)

# Accept common image config formats.
ACCEPT_CONFIG_TYPES = (
    "application/vnd.oci.image.config.v1+json,"
    "application/vnd.docker.container.image.v1+json,"
    "application/json"
)


def _parse_www_authenticate(header_value: str) -> Dict[str, str]:
    """Parse a WWW-Authenticate header into its key/value components."""
    params: Dict[str, str] = {}
    if not header_value:
        return params

    try:
        scheme, rest = header_value.split(" ", 1)
    except ValueError:
        return params

    if scheme.lower() != "bearer":
        return params

    for part in rest.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip()] = value.strip().strip('"')

    return params


def _get_bearer_token(client: httpx.Client, auth_header: str, image_name: str) -> str | None:
    """Obtain a registry bearer token using the WWW-Authenticate challenge."""
    params = _parse_www_authenticate(auth_header)
    realm = params.get("realm")
    if not realm:
        return None

    service = params.get("service")
    scope = params.get("scope") or f"repository:{image_name}:pull"

    query: Dict[str, str] = {}
    if service:
        query["service"] = service
    if scope:
        query["scope"] = scope

    try:
        response = client.get(realm, params=query)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        logger.warning("Failed to obtain bearer token for nvcr.io/%s: %s", image_name, e)
        return None
    except ValueError as e:
        logger.debug("Failed to decode bearer token response for nvcr.io/%s: %s", image_name, e)
        return None

    token = data.get("token") or data.get("access_token")
    if not isinstance(token, str) or not token:
        logger.warning("Bearer token response for nvcr.io/%s did not contain a token", image_name)
        return None

    return token


def _get_manifest(
    client: httpx.Client,
    image_name: str,
    reference: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Fetch the manifest (or manifest index) for an image reference.

    Returns the manifest JSON and any auth headers that should be reused
    for subsequent registry requests.
    """
    url = f"{NVCR_REGISTRY}/v2/{image_name}/manifests/{reference}"
    base_headers = {"Accept": ACCEPT_MANIFEST_TYPES}
    auth_headers: Dict[str, str] = {}

    try:
        response = client.get(url, headers=base_headers)
        if response.status_code == 401:
            token = _get_bearer_token(client, response.headers.get("WWW-Authenticate", ""), image_name)
            if token:
                auth_headers["Authorization"] = f"Bearer {token}"
                response = client.get(url, headers={**base_headers, **auth_headers})
        response.raise_for_status()
        manifest = response.json()
    except httpx.HTTPError:
        raise
    except ValueError as e:
        logger.debug("Failed to decode manifest response for nvcr.io/%s:%s: %s", image_name, reference, e)
        raise

    return manifest, auth_headers


def _get_json(
    client: httpx.Client,
    url: str,
    auth_headers: Dict[str, str],
    accept: str | None = None,
) -> Dict[str, Any]:
    """GET a JSON document from the registry or NGC API, reusing auth headers."""
    headers: Dict[str, str] = dict(auth_headers)
    if accept:
        headers["Accept"] = accept

    response = client.get(url, headers=headers)
    response.raise_for_status()

    try:
        return response.json()
    except ValueError as e:
        logger.debug("Failed to decode JSON response from %s: %s", url, e)
        raise


def _parse_single_manifest(
    client: httpx.Client,
    image_name: str,
    manifest: Dict[str, Any],
    auth_headers: Dict[str, str],
) -> Tuple[List[str], int, str | None]:
    """Parse a single-architecture manifest into architectures, size, and created time."""
    layers = manifest.get("layers") or []
    size_bytes = 0
    for layer in layers:
        try:
            layer_size = int(layer.get("size", 0))
        except (TypeError, ValueError):
            continue
        if layer_size > 0:
            size_bytes += layer_size

    config_desc = manifest.get("config") or {}
    config_digest = config_desc.get("digest")

    architecture: str | None = None
    created: str | None = None

    if isinstance(config_digest, str) and config_digest:
        url = f"{NVCR_REGISTRY}/v2/{image_name}/blobs/{config_digest}"
        try:
            config = _get_json(client, url, auth_headers, accept=ACCEPT_CONFIG_TYPES)
            arch = config.get("architecture")
            if isinstance(arch, str) and arch:
                architecture = arch
            created_val = config.get("created")
            if isinstance(created_val, str) and created_val:
                created = created_val
        except httpx.HTTPError as e:
            logger.debug("Failed to fetch config blob for nvcr.io/%s@%s: %s", image_name, config_digest, e)

    # Filter to valid schema architectures
    valid_archs = {"amd64", "arm64"}
    if architecture and architecture in valid_archs:
        architectures = [architecture]
    else:
        architectures = ["amd64"]  # Default fallback
    return architectures, size_bytes, created


def _parse_index_manifest(
    client: httpx.Client,
    image_name: str,
    index_manifest: Dict[str, Any],
    auth_headers: Dict[str, str],
) -> Tuple[List[str], int, str | None]:
    """Parse a manifest index (multi-arch) into architectures, size, and created time.

    Architectures are taken from the manifest index entries. Size and created time
    are determined from the selected child manifest (preferring amd64 if present).
    """
    manifests = index_manifest.get("manifests") or []
    if not manifests:
        raise ValueError("Manifest index contains no manifests")

    arch_set: set[str] = set()
    selected: Dict[str, Any] | None = None

    for entry in manifests:
        platform = entry.get("platform") or {}
        arch = platform.get("architecture")
        if isinstance(arch, str) and arch:
            arch_set.add(arch)

        if not selected and entry.get("digest"):
            selected = entry

        if arch == "amd64" and entry.get("digest"):
            selected = entry

    if not selected:
        raise ValueError("No manifest entry with digest found in manifest index")

    digest = selected.get("digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("Selected manifest entry is missing a valid digest")

    child_url = f"{NVCR_REGISTRY}/v2/{image_name}/manifests/{digest}"
    child_manifest = _get_json(client, child_url, auth_headers, accept=ACCEPT_MANIFEST_TYPES)

    single_arches, size_bytes, created = _parse_single_manifest(
        client,
        image_name,
        child_manifest,
        auth_headers,
    )

    # Filter to valid schema architectures only
    valid_archs = {"amd64", "arm64"}
    all_arches = (arch_set | set(single_arches)) & valid_archs
    architectures = sorted(all_arches) or ["amd64"]

    return architectures, size_bytes, created


def _fetch_ngc_tag_metadata(
    client: httpx.Client,
    org: str,
    repo: str,
    tag: str,
    api_key: str | None,
    team: str | None = None,
) -> Dict[str, Any] | None:
    """Fetch tag metadata from the NGC REST API, if available.

    The exact response structure can vary; this function returns the
    decoded JSON object for further inspection, or None on failure.
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build URL based on whether this is a team-scoped container
    if team:
        url = f"{NGC_API_BASE}/orgs/{org}/teams/{team}/containers/{repo}/tags/{tag}"
    else:
        url = f"{NGC_API_BASE}/orgs/{org}/containers/{repo}/tags/{tag}"

    try:
        response = client.get(url, headers=headers)
        if response.status_code == 404:
            logger.warning(
                "NGC API reported tag not found for %s/%s:%s",
                org,
                repo,
                tag,
            )
            return None

        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(
            "Failed to fetch NGC API metadata for %s/%s:%s: %s",
            org,
            repo,
            tag,
            e,
        )
        return None

    try:
        data = response.json()
    except ValueError as e:
        logger.debug(
            "Failed to decode NGC API response for %s/%s:%s as JSON: %s",
            org,
            repo,
            tag,
            e,
        )
        return None

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tag")
            if name == tag:
                return item

    logger.debug(
        "NGC API response for %s/%s:%s did not contain a matching tag object",
        org,
        repo,
        tag,
    )
    return None


def _extract_last_updated_from_metadata(metadata: Dict[str, Any] | None) -> str | None:
    """Best-effort extraction of a timestamp field from NGC tag metadata."""
    if not isinstance(metadata, dict):
        return None

    candidate_keys = (
        "last_modified",
        "lastModified",
        "modified",
        "modified_at",
        "modifiedAt",
        "updated_at",
        "updatedAt",
        "last_updated",
        "lastUpdated",
        "created_at",
        "createdAt",
    )

    for key in candidate_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value

    for nested_key in ("metadata", "image", "container"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            nested_value = _extract_last_updated_from_metadata(nested)
            if nested_value:
                return nested_value

    return None


def get_tag(
    org: str,
    repo: str,
    tag: str,
    api_key: str | None = None,
    team: str | None = None,
) -> TagInfo | None:
    """Fetch metadata for a specific NGC image tag.

    Args:
        org: NGC organization name (e.g., "nvidia")
        repo: Container repository name (e.g., "pytorch")
        tag: Tag name (e.g., "24.12-py3")
        api_key: Optional NGC API key. If not provided, the NGC_API_KEY
            environment variable will be used if set.
        team: Optional team name for nested repos (e.g., "rapidsai" for
            nvidia/rapidsai/base).

    Returns:
        TagInfo if found and parsed successfully, otherwise None.
    """
    # For team-scoped images, the registry path is org/team/repo
    if team:
        image_name = f"{org}/{team}/{repo}"
    else:
        image_name = f"{org}/{repo}"
    effective_api_key = api_key or os.getenv("NGC_API_KEY")

    with httpx.Client(timeout=TIMEOUT) as client:
        metadata = _fetch_ngc_tag_metadata(client, org, repo, tag, effective_api_key, team)

        try:
            manifest, auth_headers = _get_manifest(client, image_name, tag)
        except Exception as e:  # httpx.HTTPError or JSON decode issues
            logger.warning(
                "Failed to fetch manifest for nvcr.io/%s:%s: %s",
                image_name,
                tag,
                e,
            )
            return None

        try:
            if "manifests" in manifest:
                architectures, size_bytes, created = _parse_index_manifest(
                    client,
                    image_name,
                    manifest,
                    auth_headers,
                )
            else:
                architectures, size_bytes, created = _parse_single_manifest(
                    client,
                    image_name,
                    manifest,
                    auth_headers,
                )
        except Exception as e:
            logger.warning(
                "Failed to parse manifest data for nvcr.io/%s:%s: %s",
                image_name,
                tag,
                e,
            )
            return None

    if not architectures:
        architectures = ["amd64"]

    compressed_size_mb = size_bytes // (1024 * 1024)

    # Prefer NGC API metadata timestamps, then image config creation time,
    # and finally fall back to the current time so downstream consumers
    # always see a valid ISO datetime string.
    last_updated = (
        _extract_last_updated_from_metadata(metadata)
        or created
        or datetime.now(timezone.utc).isoformat()
    )

    return TagInfo(
        name=tag,
        last_updated=last_updated,
        architectures=architectures,
        compressed_size_mb=int(compressed_size_mb),
    )