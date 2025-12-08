#!/usr/bin/env python
"""Update the GPU Doctor image catalog from container registries.

This script orchestrates the end-to-end flow:

1. Load tracked image seeds from data/tracked_images.yaml via config.py
2. For each Docker Hub seed:
   - Fetch tag metadata from the Docker Hub API
   - Filter to explicit tags plus optionally discovered tags matching a pattern
   - Parse tags into semantic metadata using tag_parsers.py
   - Build catalog image dicts using builders.py
3. Merge generated images into the existing catalog using merge.py
4. Validate the resulting catalog against data/schema.json using jsonschema
5. Optionally write the updated catalog to data/images.json

Usage:
    python scripts/update_catalog.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import date
from typing import Any, Dict, Iterable, List, Tuple

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from builders import get_builder
from config import (
    DockerHubSeed,
    GHCRSeed,
    NGCSeed,
    IMAGES_PATH,
    SCHEMA_PATH,
    load_dockerhub_seeds,
    load_ghcr_seeds,
    load_ngc_seeds,
)
from fetchers import TagInfo
from fetchers import dockerhub as dockerhub_client
from fetchers import ghcr as ghcr_client
from fetchers import ngc as ngc_client
from merge import merge_catalog
from tag_parsers import get_parser

logger = logging.getLogger(__name__)

SOURCE_CHOICES = ("dockerhub", "ghcr", "ngc", "all")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the GPU Doctor image catalog from container registries."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and validate catalog updates but do not write images.json",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="dockerhub",
        help=(
            "Sources to update. Use 'dockerhub', 'ghcr', or 'ngc' to limit to a single registry, "
            "or 'all' to update from all configured registries."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_existing_catalog() -> Dict[str, Any]:
    with IMAGES_PATH.open() as f:
        data = json.load(f)
    if "images" not in data or not isinstance(data["images"], list):
        raise ValueError(f"Existing catalog at {IMAGES_PATH} is missing 'images' array")
    return data


def fetch_tags_for_dockerhub_seed(seed: DockerHubSeed) -> Dict[str, TagInfo]:
    """Fetch TagInfo objects for a Docker Hub seed, combining explicit and discovered tags.

    Explicit tags are always fetched. If discover is configured, additional tags
    are pulled from the registry and filtered by the compiled regex until the
    discover limit is reached.
    """
    tags_by_name: Dict[str, TagInfo] = {}

    # Explicit tags
    for tag in seed.tags:
        info = dockerhub_client.get_tag(seed.namespace, seed.repo, tag)
        if info is None:
            logger.warning(
                "Seed %s: failed to fetch explicit tag %s/%s:%s",
                seed.id,
                seed.namespace,
                seed.repo,
                tag,
            )
            continue
        tags_by_name[info.name] = info

    # Discovery, if configured
    if seed.discover:
        discovered = 0
        pattern = seed.discover.pattern
        limit = seed.discover.limit

        logger.info(
            "Seed %s: discovering up to %d tags matching %r for %s/%s",
            seed.id,
            limit,
            pattern.pattern,
            seed.namespace,
            seed.repo,
        )

        for tag_info in dockerhub_client.iter_tags(
            seed.namespace,
            seed.repo,
        ):
            name = tag_info.name

            if name in tags_by_name:
                continue
            if not pattern.match(name):
                continue

            tags_by_name[name] = tag_info
            discovered += 1

            if discovered >= limit:
                break

        logger.info(
            "Seed %s: discovered %d additional matching tags",
            seed.id,
            discovered,
        )

    logger.info(
        "Seed %s: total %d tags selected for processing",
        seed.id,
        len(tags_by_name),
    )
    return tags_by_name


def build_images_for_dockerhub_seed(seed: DockerHubSeed) -> List[Dict[str, Any]]:
    """Build catalog images for a single Docker Hub seed."""
    parser_fn = get_parser(seed.parser)
    builder_fn = get_builder(seed.parser)

    tags_by_name = fetch_tags_for_dockerhub_seed(seed)
    images: List[Dict[str, Any]] = []

    for tag_name in sorted(tags_by_name.keys()):
        tag_info = tags_by_name[tag_name]

        parsed = parser_fn(tag_info.name)
        if parsed is None:
            logger.warning(
                "Seed %s: parser '%s' could not handle tag %s for %s/%s",
                seed.id,
                seed.parser,
                tag_info.name,
                seed.namespace,
                seed.repo,
            )
            continue

        try:
            image = builder_fn(
                tag_info,
                parsed,
                namespace=seed.namespace,
                repo=seed.repo,
            )
        except TypeError as exc:
            logger.error(
                "Seed %s: builder '%s' failed for tag %s: %s",
                seed.id,
                seed.parser,
                tag_info.name,
                exc,
            )
            continue

        images.append(image)

    logger.info(
        "Seed %s: built %d images for %s/%s",
        seed.id,
        len(images),
        seed.namespace,
        seed.repo,
    )
    return images


def generate_dockerhub_images() -> List[Dict[str, Any]]:
    """Generate catalog images from all configured Docker Hub seeds."""
    seeds = load_dockerhub_seeds()
    logger.info("Loaded %d Docker Hub seeds from configuration", len(seeds))

    images: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_images = build_images_for_dockerhub_seed(seed)
        images.extend(seed_images)

    logger.info("Generated %d images from Docker Hub seeds", len(images))
    return images

def generate_ghcr_images() -> List[Dict[str, Any]]:
    """Generate catalog images from all configured GHCR seeds."""
    seeds: List[GHCRSeed] = load_ghcr_seeds()
    logger.info("Loaded %d GHCR seeds from configuration", len(seeds))

    images: List[Dict[str, Any]] = []
    for seed in seeds:
        parser_fn = get_parser(seed.parser)
        builder_fn = get_builder(seed.parser)

        seed_images: List[Dict[str, Any]] = []

        for tag in seed.tags:
            tag_info = ghcr_client.get_tag(seed.org, seed.repo, tag)
            if tag_info is None:
                logger.warning(
                    "Seed %s: failed to fetch explicit tag ghcr.io/%s/%s:%s",
                    seed.id,
                    seed.org,
                    seed.repo,
                    tag,
                )
                continue

            parsed = parser_fn(tag_info.name)
            if parsed is None:
                logger.warning(
                    "Seed %s: parser '%s' could not handle tag %s for %s/%s",
                    seed.id,
                    seed.parser,
                    tag_info.name,
                    seed.org,
                    seed.repo,
                )
                continue

            try:
                image = builder_fn(
                    tag_info,
                    parsed,
                    org=seed.org,
                    repo=seed.repo,
                )
            except TypeError as exc:
                logger.error(
                    "Seed %s: builder '%s' failed for tag %s: %s",
                    seed.id,
                    seed.parser,
                    tag_info.name,
                    exc,
                )
                continue

            seed_images.append(image)

        logger.info(
            "Seed %s: built %d images for ghcr.io/%s/%s",
            seed.id,
            len(seed_images),
            seed.org,
            seed.repo,
        )
        images.extend(seed_images)

    logger.info("Generated %d images from GHCR seeds", len(images))
    return images

def generate_ngc_images() -> List[Dict[str, Any]]:
    """Generate catalog images from all configured NGC seeds."""
    seeds: List[NGCSeed] = load_ngc_seeds()
    logger.info("Loaded %d NGC seeds from configuration", len(seeds))

    images: List[Dict[str, Any]] = []
    for seed in seeds:
        parser_fn = get_parser(seed.parser)
        builder_fn = get_builder(seed.parser)

        seed_images: List[Dict[str, Any]] = []

        for tag in seed.tags:
            tag_info = ngc_client.get_tag(seed.org, seed.repo, tag, team=seed.team)
            if tag_info is None:
                # Build the full image path for logging
                if seed.team:
                    image_path = f"nvcr.io/{seed.org}/{seed.team}/{seed.repo}:{tag}"
                else:
                    image_path = f"nvcr.io/{seed.org}/{seed.repo}:{tag}"
                logger.warning(
                    "Seed %s: failed to fetch explicit tag %s",
                    seed.id,
                    image_path,
                )
                continue

            parsed = parser_fn(tag_info.name)
            if parsed is None:
                logger.warning(
                    "Seed %s: parser '%s' could not handle tag %s for %s/%s",
                    seed.id,
                    seed.parser,
                    tag_info.name,
                    seed.org,
                    seed.repo,
                )
                continue

            try:
                # For team-scoped images, combine org/team for the builder
                effective_org = f"{seed.org}/{seed.team}" if seed.team else seed.org
                image = builder_fn(
                    tag_info,
                    parsed,
                    org=effective_org,
                    repo=seed.repo,
                )
            except TypeError as exc:
                logger.error(
                    "Seed %s: builder '%s' failed for tag %s: %s",
                    seed.id,
                    seed.parser,
                    tag_info.name,
                    exc,
                )
                continue

            seed_images.append(image)

        # Build the full image path for logging
        if seed.team:
            image_path = f"nvcr.io/{seed.org}/{seed.team}/{seed.repo}"
        else:
            image_path = f"nvcr.io/{seed.org}/{seed.repo}"
        logger.info(
            "Seed %s: built %d images for %s",
            seed.id,
            len(seed_images),
            image_path,
        )
        images.extend(seed_images)

    logger.info("Generated %d images from NGC seeds", len(images))
    return images

def summarize_changes(
    existing_images: List[Dict[str, Any]],
    merged_images: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (added_ids, updated_ids, removed_ids) between two image lists."""
    existing_by_id: Dict[str, Dict[str, Any]] = {img["id"]: img for img in existing_images}
    merged_by_id: Dict[str, Dict[str, Any]] = {img["id"]: img for img in merged_images}

    existing_ids = set(existing_by_id.keys())
    merged_ids = set(merged_by_id.keys())

    added_ids = sorted(merged_ids - existing_ids)
    removed_ids = sorted(existing_ids - merged_ids)
    updated_ids = sorted(
        img_id
        for img_id in existing_ids & merged_ids
        if existing_by_id[img_id] != merged_by_id[img_id]
    )

    return added_ids, updated_ids, removed_ids


def build_catalog_document(
    existing_catalog: Dict[str, Any],
    images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the full catalog document to be written."""
    return {
        "$schema": existing_catalog.get("$schema", "./schema.json"),
        "version": existing_catalog.get("version", "0.1.0"),
        "last_updated": date.today().isoformat(),
        "images": images,
    }


def validate_catalog(catalog: Dict[str, Any]) -> None:
    """Validate the catalog against the JSON schema."""
    with SCHEMA_PATH.open() as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(catalog)


def write_catalog(catalog: Dict[str, Any]) -> None:
    with IMAGES_PATH.open("w") as f:
        json.dump(catalog, f, indent=2, sort_keys=False)
        f.write("\n")
    logger.info("Wrote updated catalog to %s", IMAGES_PATH)


def run(source: str, dry_run: bool) -> None:
    if source not in SOURCE_CHOICES:
        raise ValueError(f"Unsupported source '{source}'. Expected one of {SOURCE_CHOICES!r}.")

    existing_catalog = load_existing_catalog()
    existing_images: List[Dict[str, Any]] = existing_catalog.get("images", [])

    generated_images: List[Dict[str, Any]] = []

    if source in ("dockerhub", "all"):
        generated_images.extend(generate_dockerhub_images())

    if source in ("ghcr", "all"):
        generated_images.extend(generate_ghcr_images())

    if source in ("ngc", "all"):
        generated_images.extend(generate_ngc_images())

    if not generated_images:
        logger.warning("No images were generated from selected sources")

    merged_images = merge_catalog(existing_images, generated_images)

    added_ids, updated_ids, removed_ids = summarize_changes(existing_images, merged_images)

    catalog_document = build_catalog_document(existing_catalog, merged_images)

    # Validate before deciding whether to write so dry-run still checks schema correctness.
    validate_catalog(catalog_document)

    if dry_run:
        logger.info(
            "Dry run: %d images would be added, %d updated, %d removed",
            len(added_ids),
            len(updated_ids),
            len(removed_ids),
        )
        if added_ids:
            logger.info("Dry run: new image IDs: %s", ", ".join(added_ids))
        if updated_ids:
            logger.info("Dry run: updated image IDs: %s", ", ".join(updated_ids))
        if removed_ids:
            logger.info("Dry run: removed image IDs: %s", ", ".join(removed_ids))
        logger.info("Dry run complete; catalog was NOT written.")
        return

    logger.info(
        "Applying changes: %d images added, %d updated, %d removed",
        len(added_ids),
        len(updated_ids),
        len(removed_ids),
    )
    write_catalog(catalog_document)


def main(argv: Iterable[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    try:
        run(source=args.source, dry_run=bool(args.dry_run))
    except ValidationError as exc:
        logger.error("Schema validation failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unhandled exception while updating catalog")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())