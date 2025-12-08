"""GitHub Container Registry (GHCR) API client for fetching image tag metadata.

This client talks directly to the OCI registry API exposed at ghcr.io.
It retrieves the image manifest and config for a given tag, then derives:

- architectures: CPU architectures from the image config or manifest index
- compressed_size_mb: Sum of compressed layer sizes (approximate)
- last_updated: Creation timestamp from the image config

Only public images are expected to work without authentication. Private images
may require standard registry token-based authentication, which is handled
via the WWW-Authenticate challenge if returned by the registry.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import httpx

from . import TagInfo

logger = logging.getLogger(__name__)

GHCR_REGISTRY = "https://ghcr.io"
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
        logger.warning("Failed to obtain bearer token for ghcr.io/%s: %s", image_name, e)
        return None
    except ValueError as e:
        logger.debug("Failed to decode bearer token response for ghcr.io/%s: %s", image_name, e)
        return None

    token = data.get("token") or data.get("access_token")
    if not isinstance(token, str) or not token:
        logger.warning("Bearer token response for ghcr.io/%s did not contain a token", image_name)
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
    url = f"{GHCR_REGISTRY}/v2/{image_name}/manifests/{reference}"
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
        logger.debug("Failed to decode manifest response for ghcr.io/%s:%s: %s", image_name, reference, e)
        raise

    return manifest, auth_headers


def _get_json(
    client: httpx.Client,
    url: str,
    auth_headers: Dict[str, str],
    accept: str | None = None,
) -> Dict[str, Any]:
    """GET a JSON document from the registry, reusing auth headers."""
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
        url = f"{GHCR_REGISTRY}/v2/{image_name}/blobs/{config_digest}"
        try:
            config = _get_json(client, url, auth_headers, accept=ACCEPT_CONFIG_TYPES)
            arch = config.get("architecture")
            if isinstance(arch, str) and arch:
                architecture = arch
            created_val = config.get("created")
            if isinstance(created_val, str) and created_val:
                created = created_val
        except httpx.HTTPError as e:
            logger.debug("Failed to fetch config blob for ghcr.io/%s@%s: %s", image_name, config_digest, e)

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

    child_url = f"{GHCR_REGISTRY}/v2/{image_name}/manifests/{digest}"
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


def get_tag(org: str, repo: str, tag: str) -> TagInfo | None:
    """Fetch metadata for a specific GHCR image tag.

    Args:
        org: GitHub organization or user (e.g., "huggingface")
        repo: Repository/package name (e.g., "text-generation-inference")
        tag: Tag name

    Returns:
        TagInfo if found and parsed successfully, otherwise None.
    """
    image_name = f"{org}/{repo}"

    with httpx.Client(timeout=TIMEOUT) as client:
        try:
            manifest, auth_headers = _get_manifest(client, image_name, tag)
        except Exception as e:  # httpx.HTTPError or JSON decode issues
            logger.warning(
                "Failed to fetch manifest for ghcr.io/%s:%s: %s",
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
                "Failed to parse manifest data for ghcr.io/%s:%s: %s",
                image_name,
                tag,
                e,
            )
            return None

    if not architectures:
        architectures = ["amd64"]

    compressed_size_mb = size_bytes // (1024 * 1024)

    # Prefer the image config creation time; fall back to current time
    # so downstream consumers always see a valid ISO datetime string.
    last_updated = created or datetime.now(timezone.utc).isoformat()

    return TagInfo(
        name=tag,
        last_updated=last_updated,
        architectures=architectures,
        compressed_size_mb=int(compressed_size_mb),
    )