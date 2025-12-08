"""Tag parsers for extracting semantic information from image tags.

Each parser takes a tag string and returns a ParsedTag with extracted metadata.
Parsers are family-specific to handle different naming conventions.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ParsedTag:
    """Semantic information extracted from a tag string."""

    # Framework info
    framework: str | None = None
    framework_version: str | None = None

    # CUDA info
    cuda_version: str | None = None
    cudnn_version: str | None = None

    # Image classification
    image_type: str | None = None  # "runtime" | "devel" | "base"
    flavor: str | None = None  # "gpu", "gpu-jupyter", "cpu", etc.

    # OS info (when extractable from tag)
    os_name: str | None = None
    os_version: str | None = None

    # Release info (for NGC-style releases)
    release: str | None = None


def parse_pytorch_cuda(tag: str) -> ParsedTag | None:
    """Parse PyTorch official CUDA image tags.

    Examples:
        - "2.5.1-cuda12.4-cudnn9-runtime" -> pytorch 2.5.1, CUDA 12.4, cuDNN 9, runtime
        - "2.4.0-cuda11.8-cudnn9-devel" -> pytorch 2.4.0, CUDA 11.8, cuDNN 9, devel

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match expected pattern
    """
    # Pattern: VERSION-cudaVERSION-cudnnVERSION-TYPE
    pattern = r"^(\d+\.\d+\.\d+)-cuda(\d+\.\d+)-cudnn(\d+)-(runtime|devel)$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="pytorch",
        framework_version=match.group(1),
        cuda_version=match.group(2),
        cudnn_version=match.group(3),
        image_type=match.group(4),
    )


def parse_tensorflow_tf(tag: str) -> ParsedTag | None:
    """Parse TensorFlow official image tags.

    Examples:
        - "2.17.0" -> tensorflow 2.17.0, CPU
        - "2.17.0-gpu" -> tensorflow 2.17.0, GPU
        - "2.17.0-gpu-jupyter" -> tensorflow 2.17.0, GPU with Jupyter

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: VERSION[-gpu][-jupyter]
    pattern = r"^(\d+\.\d+\.\d+)(?:-(gpu))?(?:-(jupyter))?$"
    match = re.match(pattern, tag)

    if not match:
        return None

    version = match.group(1)
    is_gpu = match.group(2) is not None
    is_jupyter = match.group(3) is not None

    # Determine flavor
    if is_gpu and is_jupyter:
        flavor = "gpu-jupyter"
    elif is_gpu:
        flavor = "gpu"
    else:
        flavor = "cpu"

    # TensorFlow 2.17.0 uses CUDA 12.3 (hardcoded knowledge)
    cuda_version = "12.3" if is_gpu else None

    return ParsedTag(
        framework="tensorflow",
        framework_version=version,
        cuda_version=cuda_version,
        cudnn_version="8.9" if is_gpu else None,
        image_type="runtime",
        flavor=flavor,
    )


def parse_vllm_simple(tag: str) -> ParsedTag | None:
    """Parse vLLM image tags.

    Examples:
        - "latest" -> vllm, latest
        - "v0.6.4" -> vllm 0.6.4

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info
    """
    if tag == "latest":
        return ParsedTag(
            framework="vllm",
            framework_version="latest",
            image_type="runtime",
        )

    # Pattern: vVERSION
    pattern = r"^v?(\d+\.\d+\.\d+)$"
    match = re.match(pattern, tag)

    if match:
        return ParsedTag(
            framework="vllm",
            framework_version=match.group(1),
            image_type="runtime",
        )

    return ParsedTag(
        framework="vllm",
        framework_version=tag,
        image_type="runtime",
    )


def parse_ollama_simple(tag: str) -> ParsedTag | None:
    """Parse Ollama image tags.

    Examples:
        - "latest" -> ollama, latest
        - "0.5.4" -> ollama 0.5.4

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info
    """
    version = "latest" if tag == "latest" else tag

    return ParsedTag(
        framework="ollama",
        framework_version=version,
        image_type="runtime",
    )


def parse_tgi_simple(tag: str) -> ParsedTag | None:
    """Parse Hugging Face TGI image tags.

    Examples:
        - "3.3.5" -> TGI 3.3.5
        - "latest" -> TGI latest

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info
    """
    return ParsedTag(
        framework="text-generation-inference",
        framework_version=tag,
        image_type="runtime",
    )


def parse_nvidia_cuda(tag: str) -> ParsedTag | None:
    """Parse NVIDIA CUDA base image tags.

    Examples:
        - "12.4.0-runtime-ubuntu22.04" -> CUDA 12.4.0, runtime, Ubuntu 22.04
        - "12.4.0-devel-ubuntu22.04" -> CUDA 12.4.0, devel, Ubuntu 22.04
        - "12.4.0-cudnn-devel-ubuntu22.04" -> CUDA 12.4.0 with cuDNN, devel

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: VERSION[-cudnnN]-TYPE-OS (cudnn can be just "cudnn" or "cudnn9" etc.)
    pattern = r"^(\d+\.\d+\.\d+)(?:-(cudnn)(\d+)?)?-(runtime|devel|base)-(\w+)(\d+\.\d+)$"
    match = re.match(pattern, tag)

    if not match:
        return None

    cuda_version_full = match.group(1)
    # Normalize to X.Y format (schema expects 12.4, not 12.4.0)
    cuda_parts = cuda_version_full.split(".")
    cuda_version = f"{cuda_parts[0]}.{cuda_parts[1]}"

    has_cudnn = match.group(2) is not None
    cudnn_version = match.group(3) if has_cudnn else None  # e.g., "9" from cudnn9
    image_type = match.group(4)
    os_name = match.group(5)
    os_version = match.group(6)

    return ParsedTag(
        cuda_version=cuda_version,
        cudnn_version=cudnn_version if cudnn_version else ("9" if has_cudnn else None),
        image_type=image_type,
        os_name=os_name,
        os_version=os_version,
    )


def parse_ngc_pytorch(tag: str) -> ParsedTag | None:
    """Parse NVIDIA NGC PyTorch image tags.

    Examples:
        - "24.12-py3" -> Release 24.12, Python 3

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: YY.MM-py3
    pattern = r"^(\d+\.\d+)-py3$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="pytorch",
        release=match.group(1),
        image_type="devel",  # NGC images include compilers
    )


def parse_ngc_tensorflow(tag: str) -> ParsedTag | None:
    """Parse NVIDIA NGC TensorFlow image tags.

    Examples:
        - "24.12-tf2-py3" -> Release 24.12, TensorFlow 2, Python 3

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: YY.MM-tf2-py3
    pattern = r"^(\d+\.\d+)-tf2-py3$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="tensorflow",
        release=match.group(1),
        image_type="devel",
    )


def parse_ngc_triton(tag: str) -> ParsedTag | None:
    """Parse NVIDIA Triton Inference Server image tags.

    Examples:
        - "25.11-py3" -> Release 25.11, Python 3

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: YY.MM-py3
    pattern = r"^(\d+\.\d+)-py3$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="triton-inference-server",
        release=match.group(1),
        image_type="runtime",
    )


# Registry mapping parser names to functions
PARSER_REGISTRY: dict[str, type] = {
    "pytorch_cuda": parse_pytorch_cuda,
    "tensorflow_tf": parse_tensorflow_tf,
    "vllm_simple": parse_vllm_simple,
    "ollama_simple": parse_ollama_simple,
    "tgi_simple": parse_tgi_simple,
    "nvidia_cuda": parse_nvidia_cuda,
    "ngc_pytorch": parse_ngc_pytorch,
    "ngc_tensorflow": parse_ngc_tensorflow,
    "ngc_triton": parse_ngc_triton,
}


def get_parser(parser_name: str):
    """Get a parser function by name.

    Args:
        parser_name: Name of the parser (e.g., "pytorch_cuda")

    Returns:
        Parser function

    Raises:
        KeyError: If parser not found
    """
    return PARSER_REGISTRY[parser_name]
