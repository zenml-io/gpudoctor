from __future__ import annotations

from datetime import date

from .models import SecuritySummary


def compute_rating(critical: int, high: int) -> str:
    """Deterministic A–F rating based on critical and high CVE counts."""
    if critical == 0 and high <= 1:
        return "A"
    if critical == 0 and high <= 10:
        return "B"
    if critical <= 1:
        return "C"
    if critical <= 5:
        return "D"
    return "F"


def build_security_summary(
    total_cves: int,
    critical: int,
    high: int,
    medium: int,
    low: int,
    scanner: str = "trivy",
) -> SecuritySummary:
    """Create a SecuritySummary with today's date and computed rating."""
    today = date.today().isoformat()
    rating = compute_rating(critical, high)

    # SecuritySummary.scanner is typed as Literal["trivy"], so enforce that.
    _ = scanner  # scanner is currently fixed to "trivy" for compatibility.

    return SecuritySummary(
        total_cves=total_cves,
        critical=critical,
        high=high,
        medium=medium,
        low=low,
        rating=rating,
        last_scan=today,
        scanner="trivy",
    )