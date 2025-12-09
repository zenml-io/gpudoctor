"""Merge logic for combining generated images with existing catalog entries.

The merge strategy respects ownership rules:
- Auto-managed fields: script can overwrite these
- Curated fields: preserve existing values, never overwrite

This ensures that hand-curated metadata (notes, recommended_for, etc.)
survives automatic catalog updates.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fields that the automation script owns and can overwrite
# Defined as tuples representing nested paths
AUTO_FIELDS: set[tuple[str, ...]] = {
    ("name",),
    ("metadata", "last_updated"),
    ("size", "compressed_mb"),
    ("runtime", "architectures"),
}

# Fields that may be populated by enrichment/audit pipelines and should not
# be cleared when newly generated images provide null values.
PRESERVE_IF_NEW_NULL: set[tuple[str, ...]] = {
    ("security",),
    ("audit",),
    ("size", "uncompressed_mb"),
    ("cuda", "min_driver"),
    ("cuda", "compute_capabilities"),
}


def _get_nested(data: dict, path: tuple[str, ...]) -> Any:
    """Get a value from a nested dict using a path tuple.

    Args:
        data: The dictionary to traverse
        path: Tuple of keys representing the path

    Returns:
        The value at the path, or None if not found
    """
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(data: dict, path: tuple[str, ...], value: Any) -> None:
    """Set a value in a nested dict using a path tuple.

    Creates intermediate dicts as needed.

    Args:
        data: The dictionary to modify
        path: Tuple of keys representing the path
        value: The value to set
    """
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def merge_image(
    existing: dict[str, Any] | None,
    new: dict[str, Any],
) -> dict[str, Any]:
    """Merge a newly generated image with an existing catalog entry.

    Merge strategy:
    1. If no existing entry, return the new image as-is
    2. Start with a deep copy of existing
    3. For AUTO_FIELDS, overwrite with new values
    4. For other fields, only copy from new if missing in existing
       (this allows gradual enrichment without overwriting curation)

    Args:
        existing: Existing catalog entry (None if this is a new image)
        new: Newly generated image data

    Returns:
        Merged image dictionary
    """
    if existing is None:
        return copy.deepcopy(new)

    # Start with existing data
    merged = copy.deepcopy(existing)

    # Update auto-managed fields from new data
    for path in AUTO_FIELDS:
        new_value = _get_nested(new, path)
        if new_value is not None:
            _set_nested(merged, path, new_value)

    # Preserve enrichment/audit fields when the new image has an explicit null
    # value. This prevents scraping runs from wiping data produced by separate
    # enrichers or the audit pipeline.
    for path in PRESERVE_IF_NEW_NULL:
        existing_value = _get_nested(existing, path)
        new_value = _get_nested(new, path)
        if existing_value is not None and new_value is None:
            _set_nested(merged, path, existing_value)

    # For top-level fields that exist in new but not in existing,
    # copy them over (gradual enrichment)
    for key in new:
        if key not in merged:
            merged[key] = copy.deepcopy(new[key])

    return merged


def merge_catalog(
    existing_images: list[dict[str, Any]],
    generated_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge generated images into an existing catalog.

    This function:
    1. Builds an index of existing images by ID
    2. For each generated image, merges with existing or adds new
    3. Preserves existing images not in the generated set (orphans)
    4. Returns sorted by ID for stable diffs

    Args:
        existing_images: Current catalog images
        generated_images: Newly generated images

    Returns:
        Merged list of images, sorted by ID
    """
    # Build index of existing images
    existing_by_id: dict[str, dict] = {img["id"]: img for img in existing_images}

    # Track which existing images were updated
    updated_ids: set[str] = set()

    # Check for ID collisions in generated images
    generated_ids: dict[str, str] = {}  # id -> name
    for img in generated_images:
        img_id = img["id"]
        img_name = img["name"]

        if img_id in generated_ids:
            if generated_ids[img_id] != img_name:
                logger.error(
                    "ID collision detected: '%s' used by both '%s' and '%s'",
                    img_id,
                    generated_ids[img_id],
                    img_name,
                )
                raise ValueError(f"ID collision: {img_id}")
        generated_ids[img_id] = img_name

    # Merge each generated image
    merged_images: list[dict] = []

    for gen_img in generated_images:
        img_id = gen_img["id"]
        existing_img = existing_by_id.get(img_id)

        merged = merge_image(existing_img, gen_img)
        merged_images.append(merged)
        updated_ids.add(img_id)

        if existing_img:
            logger.debug("Updated existing image: %s", img_id)
        else:
            logger.info("Added new image: %s", img_id)

    # Add orphaned existing images (not in generated set)
    for img_id, existing_img in existing_by_id.items():
        if img_id not in updated_ids:
            merged_images.append(copy.deepcopy(existing_img))
            logger.debug("Preserved orphan image: %s", img_id)

    # Sort by ID for stable diffs
    merged_images.sort(key=lambda x: x["id"])

    logger.info(
        "Merge complete: %d updated, %d new, %d orphans preserved",
        len(updated_ids & set(existing_by_id.keys())),
        len(updated_ids - set(existing_by_id.keys())),
        len(set(existing_by_id.keys()) - updated_ids),
    )

    return merged_images
