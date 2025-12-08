#!/usr/bin/env python
"""Parse AWS available_images.md to extract DLC image URIs for GPU Doctor.

This script fetches or reads the AWS Deep Learning Containers availability
Markdown and extracts repository:tag pairs suitable for tracked_images.yaml.

The Markdown is treated as the canonical catalog since AWS doesn't expose
a clean "list all DLC repos" API - ECR is just for pulling.

Usage:
    python scripts/parse_aws_available_images.py [--fetch] [--output yaml|json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import httpx

AWS_MD_URL = "https://raw.githubusercontent.com/aws/deep-learning-containers/master/available_images.md"

# Repos we want to track for GPU Doctor (focused on GPU/ML workloads)
TRACKED_REPOS = {
    # Core frameworks - training
    "pytorch-training",
    "pytorch-training-arm64",
    "tensorflow-training",
    # Core frameworks - inference
    "pytorch-inference",
    "pytorch-inference-arm64",
    "tensorflow-inference",
    # HuggingFace variants (GPU-optimized)
    "huggingface-pytorch-training",
    "huggingface-pytorch-inference",
    "huggingface-tensorflow-training",
    "huggingface-tensorflow-inference",
    # AutoGluon (AutoML)
    "autogluon-training",
    "autogluon-inference",
    # DJL (Deep Java Library) - LLM inference
    "djl-inference",
    # StabilityAI
    "stabilityai-pytorch-inference",
}

# Repos we explicitly skip (Neuron = AWS Inferentia, not GPU)
SKIP_REPOS = {
    "pytorch-inference-neuron",
    "pytorch-inference-neuronx",
    "pytorch-training-neuronx",
    "huggingface-pytorch-inference-neuron",
    "huggingface-pytorch-inference-neuronx",
    "huggingface-pytorch-training-neuronx",
    "mxnet-inference-neuron",
    "sagemaker-tritonserver",
}


@dataclass
class ImageRef:
    """A parsed AWS DLC image reference."""
    account: str
    region: str
    repository: str
    tag: str

    @property
    def uri(self) -> str:
        return f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/{self.repository}:{self.tag}"


def fetch_markdown() -> str:
    """Fetch available_images.md from GitHub."""
    response = httpx.get(AWS_MD_URL, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.text


def parse_markdown(text: str) -> List[ImageRef]:
    """Extract all valid DLC image references from the Markdown.

    Returns deduplicated images, preferring us-east-1 region.
    """
    # Pattern matches: ACCOUNT.dkr.ecr.REGION.amazonaws.com/REPO:TAG
    # TAG must be complete (not cut off by Markdown table edge)
    pattern = re.compile(
        r"(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com/"
        r"([a-z0-9-]+):([a-z0-9._-]+(?:ubuntu\d+\.\d+)?(?:-(?:ec2|sagemaker|ecs|eks))?)"
    )

    images: List[ImageRef] = []
    seen: Set[str] = set()

    for match in pattern.finditer(text):
        account, region, repo, tag = match.groups()

        # Skip if tag looks truncated (ends with hyphen or is incomplete)
        if tag.endswith("-") or tag.endswith("tran"):
            continue

        # Normalize: use us-east-1 as canonical region
        key = f"{repo}:{tag}"
        if key in seen:
            continue
        seen.add(key)

        images.append(ImageRef(
            account=account,
            region=region,
            repository=repo,
            tag=tag,
        ))

    return images


def filter_images(images: List[ImageRef]) -> List[ImageRef]:
    """Filter to only tracked repos, excluding Neuron and other non-GPU variants."""
    filtered = []
    for img in images:
        if img.repository in SKIP_REPOS:
            continue
        if img.repository in TRACKED_REPOS:
            filtered.append(img)
    return filtered


def group_by_repo(images: List[ImageRef]) -> Dict[str, List[str]]:
    """Group tags by repository."""
    by_repo: Dict[str, List[str]] = defaultdict(list)
    for img in images:
        by_repo[img.repository].append(img.tag)

    # Sort tags for consistent output
    for repo in by_repo:
        by_repo[repo] = sorted(set(by_repo[repo]), reverse=True)

    return dict(by_repo)


def generate_yaml_seeds(by_repo: Dict[str, List[str]]) -> str:
    """Generate YAML for tracked_images.yaml aws: section."""
    lines = ["aws:"]

    # Map repos to parser names
    parser_map = {
        "pytorch-training": "aws_dlc",
        "pytorch-training-arm64": "aws_dlc",
        "pytorch-inference": "aws_dlc",
        "pytorch-inference-arm64": "aws_dlc",
        "tensorflow-training": "aws_dlc",
        "tensorflow-inference": "aws_dlc",
        "huggingface-pytorch-training": "aws_hf_dlc",
        "huggingface-pytorch-inference": "aws_hf_dlc",
        "huggingface-tensorflow-training": "aws_hf_dlc",
        "huggingface-tensorflow-inference": "aws_hf_dlc",
        "autogluon-training": "aws_autogluon",
        "autogluon-inference": "aws_autogluon",
        "djl-inference": "aws_djl",
        "stabilityai-pytorch-inference": "aws_stabilityai",
    }

    for repo in sorted(by_repo.keys()):
        tags = by_repo[repo]
        parser = parser_map.get(repo, "aws_dlc")
        seed_id = f"aws-{repo}"

        lines.append(f"  - id: {seed_id}")
        lines.append(f"    repo: {repo}")
        lines.append(f"    parser: {parser}")
        lines.append("    tags:")
        for tag in tags[:20]:  # Limit to 20 most recent tags
            lines.append(f'      - "{tag}"')
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse AWS DLC available_images.md")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch latest from GitHub (otherwise reads from /tmp/aws_dlc_available_images.md)",
    )
    parser.add_argument(
        "--output",
        choices=["yaml", "json", "summary"],
        default="summary",
        help="Output format",
    )
    args = parser.parse_args()

    # Get the Markdown
    if args.fetch:
        print("Fetching available_images.md from GitHub...", file=sys.stderr)
        text = fetch_markdown()
    else:
        cache_path = Path("/tmp/aws_dlc_available_images.md")
        if not cache_path.exists():
            print(f"Cache not found at {cache_path}, fetching...", file=sys.stderr)
            text = fetch_markdown()
            cache_path.write_text(text)
        else:
            text = cache_path.read_text()

    # Parse and filter
    all_images = parse_markdown(text)
    print(f"Found {len(all_images)} total unique images", file=sys.stderr)

    filtered = filter_images(all_images)
    print(f"After filtering: {len(filtered)} images in tracked repos", file=sys.stderr)

    by_repo = group_by_repo(filtered)

    # Output
    if args.output == "yaml":
        print(generate_yaml_seeds(by_repo))
    elif args.output == "json":
        print(json.dumps(by_repo, indent=2))
    else:
        # Summary
        print("\nImages by repository:")
        for repo in sorted(by_repo.keys()):
            tags = by_repo[repo]
            print(f"  {repo}: {len(tags)} tags")
            for tag in tags[:5]:
                print(f"    - {tag}")
            if len(tags) > 5:
                print(f"    ... and {len(tags) - 5} more")


if __name__ == "__main__":
    main()
