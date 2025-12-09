#!/usr/bin/env python
"""Audit the GPU Doctor image catalog using skopeo + Trivy.

This script reads data/images.json, selects images for auditing, resolves
their digests with skopeo, optionally runs Trivy for vulnerability data,
and writes back small audit and security blocks while keeping large
artifacts under data/.audit/.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from config import IMAGES_PATH, SCHEMA_PATH
from audit.cache import AuditCache
from audit.models import AuditResult, InspectInfo, SecuritySummary, image_to_ref
from audit.rating import build_security_summary
from audit.selector import select_images
from audit.trivy_parser import extract_packages, summarize_vulnerabilities, write_trivy_artifacts
from audit.tools import ensure_audit_dirs, run_skopeo_inspect, run_trivy_scan
from audit.updater import apply_results_to_catalog

logger = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the GPU Doctor image catalog using skopeo + Trivy.",
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "security", "all"],
        default="security",
        help="Audit mode: inspect only, security (inspect+Trivy), or all (currently same as security).",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=str(IMAGES_PATH),
        help="Path to images.json catalog file (default: data/images.json).",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default=None,
        help="Directory for audit artifacts and index (default: data/.audit next to images.json).",
    )
    parser.add_argument(
        "--security-ttl-days",
        type=int,
        default=7,
        help="Minimum days between security scans for the same digest.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to audit in this run.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter images by simple metadata fields (e.g., provider=pytorch). Can be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scan of security even if TTL/backoff would skip it.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable audit cache (still writes artifacts but does not persist index).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute audit updates but do not write images.json.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="amd64",
        help="Target CPU architecture to audit (default: amd64).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_catalog(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Catalog at {path} is missing 'images' array")
    return data


def _load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open() as f:
        return json.load(f)


def _parse_filters(raw_filters: List[str]) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            logger.warning("Ignoring invalid filter %r (expected KEY=VALUE)", item)
            continue
        key, value = item.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def _build_inspect_info(data: Dict[str, Any]) -> InspectInfo:
    digest = data.get("Digest") or data.get("digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("Inspect data is missing a Digest field")

    # Env may appear either at the top level or nested under Config.
    env_list = data.get("Env")
    if not isinstance(env_list, list):
        config = data.get("Config") or {}
        env_list = config.get("Env") or []
    env: Dict[str, str] = {}
    for item in env_list:
        if not isinstance(item, str):
            continue
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        env[key] = value

    labels = data.get("Labels")
    if not isinstance(labels, dict):
        config = data.get("Config") or {}
        labels = config.get("Labels") or {}
    clean_labels: Dict[str, str] = {}
    for k, v in labels.items():
        if isinstance(k, str) and isinstance(v, str):
            clean_labels[k] = v

    arch = data.get("Architecture")
    if not isinstance(arch, str):
        config = data.get("Config") or {}
        arch_val = config.get("Architecture")
        arch = arch_val if isinstance(arch_val, str) else None

    inspected_at = datetime.now(timezone.utc).isoformat()

    return InspectInfo(
        digest=digest,
        env=env,
        labels=clean_labels,
        architecture=arch,
        inspected_at=inspected_at,
    )


def _security_from_cache_entry(entry: Dict[str, Any]) -> SecuritySummary | None:
    security = entry.get("security")
    if not isinstance(security, dict):
        return None

    try:
        last_scan = str(security.get("last_security_scan"))
        return SecuritySummary(
            total_cves=int(security.get("total_cves", 0)),
            critical=int(security.get("critical", 0)),
            high=int(security.get("high", 0)),
            medium=int(security.get("medium", 0)),
            low=int(security.get("low", 0)),
            rating=str(security.get("rating", "F")),
            last_scan=last_scan,
            scanner="trivy",
        )
    except (TypeError, ValueError):
        return None


def run(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    catalog_path = Path(args.catalog_path)
    if args.audit_dir:
        audit_root = Path(args.audit_dir)
    else:
        # Default to a sibling of images.json: data/.audit
        audit_root = IMAGES_PATH.parent / ".audit"

    # Ensure artifact directories exist early for a smoother experience.
    inspect_dir, trivy_dir, packages_dir = ensure_audit_dirs(audit_root)

    cache = AuditCache(audit_root=audit_root, enabled=not args.no_cache)

    try:
        catalog = _load_catalog(catalog_path)
    except Exception:
        logger.exception("Failed to load catalog from %s", catalog_path)
        return 1

    images: List[Dict[str, Any]] = catalog.get("images") or []

    filters = _parse_filters(args.filter or [])
    selections = select_images(
        images,
        arch=args.arch,
        max_images=args.max_images,
        filters=filters,
    )

    if not selections:
        logger.info("No images selected for audit (filters/limits may have excluded all images).")
        # Still validate catalog to ensure schema compatibility.
        schema = _load_schema()
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        validator.validate(catalog)
        return 0

    logger.info(
        "Selected %d images for audit (mode=%s, arch=%s)",
        len(selections),
        args.mode,
        args.arch,
    )

    do_security = args.mode in ("security", "all")
    today = date.today()

    # First pass: inspect all selected images and group by digest.
    inspect_by_digest: Dict[str, InspectInfo] = {}
    raw_inspect_by_digest: Dict[str, Dict[str, Any]] = {}
    images_by_digest: Dict[str, List[int]] = {}

    for sel in selections:
        image = sel.image
        try:
            ref = image_to_ref(image)
        except Exception as exc:
            logger.warning("Skipping image %s (%s): %s", sel.image_id, image.get("name"), exc)
            continue

        try:
            inspect_data = run_skopeo_inspect(ref)
            info = _build_inspect_info(inspect_data)
        except Exception as exc:
            logger.warning(
                "Failed to inspect image %s (%s): %s",
                sel.image_id,
                image.get("name"),
                exc,
            )
            continue

        digest = info.digest
        if digest not in inspect_by_digest:
            inspect_by_digest[digest] = info
            raw_inspect_by_digest[digest] = inspect_data

        images_by_digest.setdefault(digest, []).append(sel.index)

    if not inspect_by_digest:
        logger.warning("Inspection step produced no digests; nothing to do.")
        schema = _load_schema()
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        validator.validate(catalog)
        return 0

    logger.info(
        "Resolved %d unique digests from %d selected images",
        len(inspect_by_digest),
        len(selections),
    )

    # Second pass: persist inspect artifacts and, optionally, run security scans.
    results_by_digest: Dict[str, AuditResult] = {}

    for digest, info in inspect_by_digest.items():
        # Persist raw inspect JSON once per digest.
        inspect_data = raw_inspect_by_digest[digest]
        inspect_path = cache.inspect_path_for(digest)
        inspect_path.parent.mkdir(parents=True, exist_ok=True)
        with inspect_path.open("w") as f:
            json.dump(inspect_data, f, indent=2, sort_keys=True)
            f.write("\n")
        inspect_ref = cache.relative_ref(inspect_path)
        cache.update_inspect(digest, info, inspect_ref)

        security_summary: SecuritySummary | None = None
        trivy_ref: str | None = None
        packages_ref: str | None = None

        if do_security:
            # Decide whether to run a new Trivy scan or reuse cached data.
            entry = cache.get_entry(digest)
            should_run = cache.should_run_security_scan(
                digest=digest,
                today=today,
                ttl_days=int(args.security_ttl_days),
                force=bool(args.force),
            )

            if not should_run and entry and entry.get("security"):
                # Use cached summary and artifact refs.
                cached_summary = _security_from_cache_entry(entry)
                if cached_summary is not None:
                    security_summary = cached_summary
                    sec = entry["security"]
                    trivy_ref = sec.get("trivy_report_ref")
                    packages_ref = sec.get("packages_ref")
                    logger.info(
                        "Reusing cached security summary for %s (last_scan=%s)",
                        digest,
                        cached_summary.last_scan,
                    )
            elif should_run:
                # We need to run Trivy for this digest.
                # Use the first image reference associated with this digest.
                idx_list = images_by_digest.get(digest, [])
                if not idx_list:
                    logger.debug(
                        "No image index found for digest %s; skipping security scan", digest
                    )
                else:
                    image = images[idx_list[0]]
                    try:
                        ref = image_to_ref(image)
                    except Exception as exc:
                        logger.warning(
                            "Unable to build image ref for digest %s (%s): %s",
                            digest,
                            image.get("name"),
                            exc,
                        )
                        cache.register_failure(digest, f"ref-build-failed: {exc}", today=today)
                    else:
                        try:
                            report = run_trivy_scan(ref, digest=digest, arch=args.arch)
                            total, critical, high, medium, low = summarize_vulnerabilities(report)
                            packages = extract_packages(report)
                            trivy_ref, packages_ref = write_trivy_artifacts(
                                digest,
                                report,
                                packages,
                                trivy_dir,
                                packages_dir,
                            )
                            security_summary = build_security_summary(
                                total_cves=total,
                                critical=critical,
                                high=high,
                                medium=medium,
                                low=low,
                            )
                            cache.update_security(
                                digest,
                                security_summary,
                                trivy_report_ref=trivy_ref,
                                packages_ref=packages_ref,
                            )
                        except Exception as exc:
                            logger.warning("Security scan failed for %s: %s", digest, exc)
                            cache.register_failure(digest, f"trivy-failed: {exc}", today=today)

        results_by_digest[digest] = AuditResult(
            digest=digest,
            inspect=info,
            security=security_summary,
            trivy_report_ref=trivy_ref,
            packages_ref=packages_ref,
        )

    # Apply audit results back to the catalog images.
    apply_results_to_catalog(
        catalog,
        results_by_digest=results_by_digest,
        images_by_digest=images_by_digest,
        today=today,
    )

    # Validate against the JSON schema.
    schema = _load_schema()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    try:
        validator.validate(catalog)
    except ValidationError as exc:
        logger.error("Schema validation failed after audit updates: %s", exc)
        return 1

    image_count = sum(len(v) for v in images_by_digest.values())
    if args.dry_run:
        logger.info(
            "Dry run: audited %d digests across %d images; catalog was NOT written.",
            len(results_by_digest),
            image_count,
        )
    else:
        with catalog_path.open("w") as f:
            json.dump(catalog, f, indent=2, sort_keys=False)
            f.write("\n")
        logger.info(
            "Wrote updated catalog with audit data to %s (%d digests, %d images)",
            catalog_path,
            len(results_by_digest),
            image_count,
        )

    # Persist audit index regardless of dry-run state so TTL/backoff evolve.
    cache.save()
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        return run(argv)
    except Exception:
        logger.exception("Unhandled exception while auditing catalog")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())