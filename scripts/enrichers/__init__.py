# Enricher modules for future auditing (security scans, CUDA compat, etc.)
#
# Each enricher should implement:
#   def enrich_catalog(catalog: dict) -> dict:
#       """Return a new catalog with additional fields populated."""
