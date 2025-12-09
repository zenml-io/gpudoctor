from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def summarize_vulnerabilities(report: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Return (total, critical, high, medium, low) counts from a Trivy report."""
    total = 0
    critical = high = medium = low = 0

    results = report.get("Results") or []
    for result in results:
        vulns = result.get("Vulnerabilities") or []
        for vuln in vulns:
            total += 1
            sev = str(vuln.get("Severity", "")).upper()
            if sev == "CRITICAL":
                critical += 1
            elif sev == "HIGH":
                high += 1
            elif sev == "MEDIUM":
                medium += 1
            elif sev == "LOW":
                low += 1
            # UNKNOWN and other severities are counted only in the total.

    return total, critical, high, medium, low


def extract_packages(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact package inventory from a Trivy report.

    This uses package names and versions visible in vulnerability entries.
    It is not a full SBOM but good enough for coarse inventory and debugging.
    """
    os_packages: Dict[str, str] = {}
    lang_packages: Dict[str, Dict[str, str]] = {}

    results = report.get("Results") or []
    for result in results:
        r_class = result.get("Class")
        r_type = str(result.get("Type") or "")

        language: str | None = None
        if r_class == "lang-pkgs":
            # Map common Trivy type identifiers to short language names.
            if r_type.startswith("python"):
                language = "python"
            elif r_type.startswith("node"):
                language = "node"
            elif r_type.startswith("go"):
                language = "go"
            else:
                language = r_type or "unknown"

        vulns = result.get("Vulnerabilities") or []
        for vuln in vulns:
            pkg = vuln.get("PkgName")
            version = vuln.get("InstalledVersion")
            if not pkg or not version:
                continue

            if r_class == "os-pkgs":
                # Keep the highest version string we have seen.
                os_packages[pkg] = str(version)
            elif language:
                lang = lang_packages.setdefault(language, {})
                lang[pkg] = str(version)

    inventory: Dict[str, Any] = {
        "os": [
            {"name": name, "version": version}
            for name, version in sorted(os_packages.items())
        ],
        "languages": {
            lang: [
                {"name": name, "version": version}
                for name, version in sorted(pkgs.items())
            ]
            for lang, pkgs in sorted(lang_packages.items())
        },
    }
    return inventory


def write_trivy_artifacts(
    digest: str,
    report_data: Dict[str, Any],
    packages: Dict[str, Any] | None,
    trivy_dir: Path,
    packages_dir: Path,
) -> tuple[str | None, str | None]:
    """Write raw Trivy report and optional package inventory to disk.

    Returns:
        (trivy_report_ref, packages_ref) as repository-relative paths,
        suitable for storing in the catalog's audit block.
    """
    base_name = digest.replace(":", "-")

    # Raw Trivy report
    trivy_path = trivy_dir / f"{base_name}.json"
    trivy_path.parent.mkdir(parents=True, exist_ok=True)
    with trivy_path.open("w") as f:
        json.dump(report_data, f, indent=2, sort_keys=True)
        f.write("\n")

    try:
        trivy_ref = str(trivy_path.relative_to(ROOT))
    except ValueError:
        trivy_ref = str(trivy_path)

    packages_ref: str | None = None
    if packages:
        packages_path = packages_dir / f"{base_name}.json.gz"
        packages_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(packages_path, "wt", encoding="utf-8") as f:
            json.dump(packages, f, indent=2, sort_keys=True)
            f.write("\n")

        try:
            packages_ref = str(packages_path.relative_to(ROOT))
        except ValueError:
            packages_ref = str(packages_path)

    logger.info(
        "Wrote Trivy artifacts for %s to %s (packages: %s)",
        digest,
        trivy_path,
        packages_ref,
    )
    return trivy_ref, packages_ref