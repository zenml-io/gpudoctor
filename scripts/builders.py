"""Image builders that combine TagInfo + ParsedTag into catalog-ready dicts.

Each builder creates a complete image dictionary matching the JSON schema.
Builders are family-specific to handle different naming conventions and defaults.
"""

from __future__ import annotations

from typing import Any

from fetchers import TagInfo
from tag_parsers import ParsedTag


def _version_to_id_part(version: str) -> str:
    """Convert a version string to an ID-safe format.

    Examples:
        "2.5.1" -> "2-5-1"
        "12.4" -> "12-4"
    """
    return version.replace(".", "-")


def _truncate_date(iso_datetime: str) -> str:
    """Truncate ISO datetime to date only (YYYY-MM-DD)."""
    if not iso_datetime:
        return ""
    return iso_datetime[:10]


def _build_cuda_object(
    version: str | None,
    cudnn: str | None = None,
    min_driver: str | None = None,
    compute_capabilities: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a CUDA metadata object, omitting any keys whose value is None.

    Returns None if version is None, allowing callers to represent the absence of
    CUDA support with a null cuda field instead of a partially-filled object.
    """
    if version is None:
        return None

    cuda: dict[str, Any] = {"version": version}
    if cudnn is not None:
        cuda["cudnn"] = cudnn
    if min_driver is not None:
        cuda["min_driver"] = min_driver
    if compute_capabilities is not None:
        cuda["compute_capabilities"] = compute_capabilities
    return cuda


def build_pytorch_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    namespace: str = "pytorch",
    repo: str = "pytorch",
) -> dict[str, Any]:
    """Build a catalog entry for PyTorch official images.

    ID patterns:
        - CUDA: pytorch-{version}-cuda{cuda_version}-{image_type}
        - CPU: pytorch-{version}-cpu-runtime
    Examples:
        - pytorch-2-5-1-cuda12-4-runtime
        - pytorch-2-6-0-cpu-runtime
    """
    version_id = _version_to_id_part(parsed.framework_version or "")
    image_type = parsed.image_type or "runtime"

    # Build ID based on whether this is a CPU or GPU image
    if parsed.cuda_version:
        cuda_id = _version_to_id_part(parsed.cuda_version)
        image_id = f"pytorch-{version_id}-cuda{cuda_id}-{image_type}"
        gpu_vendors = ["nvidia"]
    else:
        image_id = f"pytorch-{version_id}-cpu-{image_type}"
        gpu_vendors = ["none"]

    full_name = f"{namespace}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "pytorch",
            "registry": "dockerhub",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "BSD-3-Clause",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
            cudnn=parsed.cudnn_version,
        ),
        "runtime": {
            "python": "3.11",  # Default for recent PyTorch
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "pytorch", "version": parsed.framework_version},
        ],
        "capabilities": {
            "gpu_vendors": gpu_vendors,
            "image_type": image_type,
            "role": "training",
            "workloads": ["llm", "computer-vision", "nlp", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://hub.docker.com/r/{namespace}/{repo}",
            "documentation": "https://pytorch.org/docs/stable/",
            "source": "https://github.com/pytorch/pytorch",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_tensorflow_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    namespace: str = "tensorflow",
    repo: str = "tensorflow",
) -> dict[str, Any]:
    """Build a catalog entry for TensorFlow official images.

    ID patterns:
        - tensorflow-{version}-gpu
        - tensorflow-{version}-gpu-jupyter
        - tensorflow-{version}-cpu
    """
    version_id = _version_to_id_part(parsed.framework_version or "")
    flavor = parsed.flavor or "cpu"

    image_id = f"tensorflow-{version_id}-{flavor}"
    full_name = f"{namespace}/{repo}:{tag_info.name}"

    # Determine role based on flavor
    role = "notebook" if "jupyter" in flavor else "training"
    gpu_vendors = ["nvidia"] if "gpu" in flavor else ["none"]

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "tensorflow",
            "registry": "dockerhub",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
            cudnn=parsed.cudnn_version,
        ),
        "runtime": {
            "python": "3.11",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "tensorflow", "version": parsed.framework_version},
        ],
        "capabilities": {
            "gpu_vendors": gpu_vendors,
            "image_type": "runtime",
            "role": role,
            "workloads": ["classical-ml", "computer-vision", "nlp", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://hub.docker.com/r/{namespace}/{repo}",
            "documentation": "https://www.tensorflow.org/install/docker",
            "source": "https://github.com/tensorflow/tensorflow",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_vllm_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    namespace: str = "vllm",
    repo: str = "vllm-openai",
) -> dict[str, Any]:
    """Build a catalog entry for vLLM images.

    ID pattern: vllm-openai-{tag}
    """
    tag_id = _version_to_id_part(tag_info.name)
    image_id = f"vllm-openai-{tag_id}"
    full_name = f"{namespace}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "vllm",
            "registry": "dockerhub",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version="12.4",
            cudnn="9",
        ),
        "runtime": {
            "python": "3.12",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "vllm", "version": parsed.framework_version or tag_info.name},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "runtime",
            "role": "serving",
            "workloads": ["llm"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://hub.docker.com/r/{namespace}/{repo}",
            "documentation": "https://docs.vllm.ai/en/latest/",
            "source": "https://github.com/vllm-project/vllm",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ollama_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    namespace: str = "ollama",
    repo: str = "ollama",
) -> dict[str, Any]:
    """Build a catalog entry for Ollama images.

    ID pattern: ollama-{tag}
    """
    tag_id = _version_to_id_part(tag_info.name)
    image_id = f"ollama-{tag_id}"
    full_name = f"{namespace}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "ollama",
            "registry": "dockerhub",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "MIT",
        },
        "cuda": _build_cuda_object(
            version="12.4",
        ),
        "runtime": {
            "python": None,  # Ollama is Go-based
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64", "arm64"],
        },
        "frameworks": [
            {"name": "ollama", "version": parsed.framework_version or tag_info.name},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia", "amd"],
            "image_type": "runtime",
            "role": "serving",
            "workloads": ["llm"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://hub.docker.com/r/{namespace}/{repo}",
            "documentation": "https://ollama.ai/",
            "source": "https://github.com/ollama/ollama",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_nvidia_cuda_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    namespace: str = "nvidia",
    repo: str = "cuda",
) -> dict[str, Any]:
    """Build a catalog entry for NVIDIA CUDA base images.

    ID pattern: nvidia-cuda-{version}-{type}-{os}
    Example: nvidia-cuda-12-4-runtime-ubuntu22-04
    """
    cuda_id = _version_to_id_part(parsed.cuda_version or "")
    os_id = f"{parsed.os_name}{_version_to_id_part(parsed.os_version or '')}"
    image_type = parsed.image_type or "runtime"

    # Handle cudnn variant - prepend cudnn to type when cuDNN is included
    if parsed.cudnn_version:
        type_suffix = f"cudnn-{image_type}"
    else:
        type_suffix = image_type

    image_id = f"nvidia-cuda-{cuda_id}-{type_suffix}-{os_id}"
    full_name = f"{namespace}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "dockerhub",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
            cudnn=parsed.cudnn_version,
        ),
        "runtime": {
            "python": None,
            "os": {"name": parsed.os_name or "ubuntu", "version": parsed.os_version or "22.04"},
            "architectures": tag_info.architectures or ["amd64", "arm64"],
        },
        "frameworks": [],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": image_type,
            "role": "base",
            "workloads": ["generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://hub.docker.com/r/{namespace}/{repo}",
            "documentation": "https://docs.nvidia.com/cuda/",
            "source": None,
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_tgi_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "huggingface",
    repo: str = "text-generation-inference",
) -> dict[str, Any]:
    """Build a catalog entry for Hugging Face TGI images.

    ID patterns:
        - tgi-{version} for bare version tags
        - tgi-{version}-cuda{cuda} for CUDA-suffixed tags
    Examples:
        - tgi-3-3-5
        - tgi-3-3-5-cuda12-4
    """
    version_id = _version_to_id_part(parsed.framework_version or tag_info.name)

    # Use parsed CUDA version or default to 12.4
    cuda_version = parsed.cuda_version or "12.4"

    # Include CUDA in ID only if explicitly specified in tag
    if parsed.cuda_version:
        cuda_id = _version_to_id_part(parsed.cuda_version)
        image_id = f"tgi-{version_id}-cuda{cuda_id}"
    else:
        image_id = f"tgi-{version_id}"

    full_name = f"ghcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "huggingface",
            "registry": "ghcr",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version=cuda_version,
            cudnn="9",
        ),
        "runtime": {
            "python": "3.11",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "text-generation-inference", "version": tag_info.name},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "runtime",
            "role": "serving",
            "workloads": ["llm"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://github.com/{org}/{repo}/pkgs/container/{repo}",
            "documentation": "https://huggingface.co/docs/text-generation-inference",
            "source": f"https://github.com/{org}/{repo}",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_pytorch_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia",
    repo: str = "pytorch",
) -> dict[str, Any]:
    """Build a catalog entry for NGC PyTorch images.

    ID pattern: ngc-pytorch-{release}
    Example: ngc-pytorch-24-12
    """
    release_id = _version_to_id_part(parsed.release or tag_info.name.split("-")[0])
    image_id = f"ngc-pytorch-{release_id}"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=None,  # Varies by release, filled by enricher
        ),
        "runtime": {
            "python": "3.12",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "pytorch", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "devel",
            "role": "training",
            "workloads": ["llm", "computer-vision", "nlp", "multimodal", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/",
            "source": None,
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_tensorflow_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia",
    repo: str = "tensorflow",
) -> dict[str, Any]:
    """Build a catalog entry for NGC TensorFlow images.

    ID pattern: ngc-tensorflow-{release}
    Example: ngc-tensorflow-24-12
    """
    release_id = _version_to_id_part(parsed.release or tag_info.name.split("-")[0])
    image_id = f"ngc-tensorflow-{release_id}"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=None,
        ),
        "runtime": {
            "python": "3.10",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "tensorflow", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "devel",
            "role": "training",
            "workloads": ["classical-ml", "computer-vision", "nlp", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/",
            "source": None,
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_triton_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia",
    repo: str = "tritonserver",
) -> dict[str, Any]:
    """Build a catalog entry for NVIDIA Triton Inference Server images.

    ID pattern: tritonserver-{release}
    Example: tritonserver-25-11-py3
    """
    release_id = _version_to_id_part(tag_info.name.replace("-py3", ""))
    image_id = f"tritonserver-{release_id}-py3"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=None,
        ),
        "runtime": {
            "python": "3.12",
            "os": {"name": "ubuntu", "version": "24.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "triton-inference-server", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "runtime",
            "role": "serving",
            "workloads": ["llm", "computer-vision", "nlp", "multimodal", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.nvidia.com/deeplearning/triton-inference-server/",
            "source": "https://github.com/triton-inference-server/server",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_jax_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia",
    repo: str = "jax",
) -> dict[str, Any]:
    """Build a catalog entry for NVIDIA NGC JAX images.

    ID pattern: ngc-jax-{release}
    Example: ngc-jax-24-12
    """
    release_id = _version_to_id_part(parsed.release or tag_info.name.split("-")[0])
    image_id = f"ngc-jax-{release_id}"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=None,
        ),
        "runtime": {
            "python": "3.10",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "jax", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "devel",
            "role": "training",
            "workloads": ["llm", "scientific-computing", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/",
            "source": None,
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_nemo_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia",
    repo: str = "nemo",
) -> dict[str, Any]:
    """Build a catalog entry for NVIDIA NeMo Framework images.

    ID pattern: ngc-nemo-{release}
    Example: ngc-nemo-24-12
    """
    release_id = _version_to_id_part(parsed.release or tag_info.name)
    image_id = f"ngc-nemo-{release_id}"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "NVIDIA-proprietary",
        },
        "cuda": _build_cuda_object(
            version=None,
        ),
        "runtime": {
            "python": "3.10",
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "nemo", "version": "varies-by-release"},
            {"name": "pytorch", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "devel",
            "role": "training",
            "workloads": ["llm", "nlp", "audio", "multimodal"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.nvidia.com/nemo-framework/user-guide/latest/",
            "source": "https://github.com/NVIDIA/NeMo",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_ngc_rapids_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "nvidia/rapidsai",
    repo: str = "base",
) -> dict[str, Any]:
    """Build a catalog entry for NVIDIA RAPIDS images.

    ID pattern: ngc-rapids-{release}-cuda{cuda}-py{python}
    Example: ngc-rapids-24-10-cuda12-5-py3-12
    """
    release_id = _version_to_id_part(parsed.release or "")
    cuda_id = _version_to_id_part(parsed.cuda_version or "")
    # Extract Python version from tag (e.g., "24.10-cuda12.5-py3.12" -> "3.12")
    py_version = tag_info.name.split("-py")[-1] if "-py" in tag_info.name else "3.10"
    py_id = _version_to_id_part(py_version)

    image_id = f"ngc-rapids-{release_id}-cuda{cuda_id}-py{py_id}"
    full_name = f"nvcr.io/{org}/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "nvidia-ngc",
            "registry": "ngc",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
        ),
        "runtime": {
            "python": py_version,
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": "cudf", "version": "varies-by-release"},
            {"name": "cuml", "version": "varies-by-release"},
            {"name": "cugraph", "version": "varies-by-release"},
        ],
        "capabilities": {
            "gpu_vendors": ["nvidia"],
            "image_type": "runtime",
            "role": "training",
            "workloads": ["classical-ml", "scientific-computing", "generic"],
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://catalog.ngc.nvidia.com/orgs/{org}/containers/{repo}",
            "documentation": "https://docs.rapids.ai/",
            "source": "https://github.com/rapidsai",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


def build_aws_dlc_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    repo: str = "pytorch-training",
) -> dict[str, Any]:
    """Build a catalog entry for AWS Deep Learning Container images.

    ID pattern: aws-{repo}-{version}-{cuda|cpu}-{platform}
    Examples:
        - aws-pytorch-training-2-7-1-cuda12-8-ec2
        - aws-tensorflow-inference-2-18-0-cpu-sagemaker
    """
    version_id = _version_to_id_part(parsed.framework_version or "")

    # Determine framework from repo name
    if "pytorch" in repo:
        framework_name = "pytorch"
    elif "tensorflow" in repo:
        framework_name = "tensorflow"
    else:
        framework_name = "unknown"

    # Determine role from repo name
    if "training" in repo:
        role = "training"
    elif "inference" in repo:
        role = "serving"
    else:
        role = "base"

    # Build ID from flavor which includes platform (gpu-ec2, cpu-sagemaker, etc.)
    flavor = parsed.flavor or "gpu-ec2"
    flavor_parts = flavor.split("-")
    compute_type = flavor_parts[0]  # "gpu" or "cpu"
    platform = flavor_parts[1] if len(flavor_parts) > 1 else "ec2"

    if parsed.cuda_version:
        cuda_id = _version_to_id_part(parsed.cuda_version)
        compute_id = f"cuda{cuda_id}"
        gpu_vendors = ["nvidia"]
    else:
        compute_id = "cpu"
        gpu_vendors = ["none"]

    image_id = f"aws-{repo}-{version_id}-{compute_id}-{platform}"
    full_name = f"public.ecr.aws/deep-learning-containers/{repo}:{tag_info.name}"

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "aws-dlc",
            "registry": "ecr",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
            cudnn=parsed.cudnn_version,
        ),
        "runtime": {
            "python": "3.11",  # Default for recent AWS DLCs
            "os": {"name": parsed.os_name or "ubuntu", "version": parsed.os_version or "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": framework_name, "version": parsed.framework_version},
        ],
        "capabilities": {
            "gpu_vendors": gpu_vendors,
            "image_type": "runtime",
            "role": role,
            "workloads": ["llm", "computer-vision", "nlp", "generic"] if framework_name == "pytorch" else ["classical-ml", "computer-vision", "nlp", "generic"],
        },
        "cloud": {
            "affinity": ["aws"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://gallery.ecr.aws/deep-learning-containers/{repo}",
            "documentation": "https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/",
            "source": "https://github.com/aws/deep-learning-containers",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": f"Optimized for {platform.upper()} deployment on AWS.",
    }


def build_gcp_dlc_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    repo: str = "pytorch-gpu.2-2",
) -> dict[str, Any]:
    """Build a catalog entry for GCP Deep Learning Container images.

    ID pattern: gcp-{framework}[-{version}]-{cuda|cpu}[-py{python}][-{variant}]
    Examples:
        - gcp-pytorch-2-4-cuda12-1
        - gcp-pytorch-2-4-cuda12-1-py3-10
        - gcp-pytorch-2-4-cuda12-1-conda
        - gcp-tensorflow-2-17-cpu-slim
        - gcp-base-cuda12-1
        - gcp-r-4-4-cpu
        - gcp-rapids-21-12-cuda12-1
    """
    import re

    # Extract Python version from repo name if present (e.g., ".py310" -> "3.10")
    python_suffix = ""
    python_version = None
    python_match = re.search(r"\.py(\d)(\d+)$", repo)
    if python_match:
        python_version = f"{python_match.group(1)}.{python_match.group(2)}"
        python_suffix = f"-py{_version_to_id_part(python_version)}"

    # Handle base images specially (they have no framework)
    if parsed.image_type == "base" and parsed.framework is None:
        framework_name = "base"
        id_framework = "base"
        version_id = ""
    else:
        framework_name = parsed.framework or "unknown"
        # For TensorFlow, preserve tf vs tf2 prefix distinction in ID only
        # (frameworks array still uses "tensorflow")
        if framework_name == "tensorflow" and repo.startswith("tf2"):
            id_framework = "tf2"
        elif framework_name == "tensorflow":
            id_framework = "tf"
        else:
            id_framework = framework_name
        version_id = _version_to_id_part(parsed.framework_version or "")

    # Determine compute type for ID
    # Use "gpu" for generic -gpu repos, "cuda{version}" for explicit -cuXXX repos
    has_explicit_cuda = bool(re.search(r"-cu\d{2,3}", repo))
    if parsed.cuda_version:
        if has_explicit_cuda:
            cuda_id = _version_to_id_part(parsed.cuda_version)
            compute_id = f"cuda{cuda_id}"
        else:
            # Generic -gpu repo - use "gpu" instead of specific CUDA version
            compute_id = "gpu"
        gpu_vendors = ["nvidia"]
    else:
        compute_id = "cpu"
        gpu_vendors = ["none"]

    # Extract variant suffix directly from repo name (more reliable than parsed.flavor)
    variant_suffix = ""
    if "-conda" in repo:
        variant_suffix += "-conda"
    if "-slim" in repo:
        variant_suffix += "-slim"

    # Build ID, handling base images which have no version
    # Include Python suffix to ensure uniqueness for .pyXXX variants
    if version_id:
        image_id = f"gcp-{id_framework}-{version_id}-{compute_id}{python_suffix}{variant_suffix}"
    else:
        image_id = f"gcp-{id_framework}-{compute_id}{python_suffix}{variant_suffix}"
    full_name = f"gcr.io/deeplearning-platform-release/{repo}:{tag_info.name}"

    # Framework-specific role and workloads
    framework_config = {
        "pytorch": ("training", ["llm", "computer-vision", "nlp", "generic"]),
        "tensorflow": ("training", ["classical-ml", "computer-vision", "nlp", "generic"]),
        "rapids": ("training", ["classical-ml", "scientific-computing", "generic"]),
        "r": ("notebook", ["classical-ml", "scientific-computing", "generic"]),
        "scikit-learn": ("training", ["classical-ml", "generic"]),
        "xgboost": ("training", ["classical-ml", "generic"]),
        "spark": ("training", ["classical-ml", "generic"]),
        "base": ("base", ["generic"]),
    }
    role, workloads = framework_config.get(framework_name, ("base", ["generic"]))

    # Build notes with variant info
    notes = "Optimized for Google Cloud Platform (Vertex AI, GKE, Compute Engine)."
    if "conda" in (parsed.flavor or ""):
        notes += " Conda-based environment."
    if "slim" in (parsed.flavor or ""):
        notes += " Slim variant without development tools."

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "gcp-dlc",
            "registry": "gcr",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "Apache-2.0",
        },
        "cuda": _build_cuda_object(
            version=parsed.cuda_version,
        ),
        "runtime": {
            "python": python_version or "3.10",  # Use extracted version or default
            "os": {"name": "ubuntu", "version": "22.04"},
            "architectures": tag_info.architectures or ["amd64"],
        },
        "frameworks": [
            {"name": framework_name, "version": parsed.framework_version or "latest"},
        ] if framework_name not in ("unknown", "base") else [],
        "capabilities": {
            "gpu_vendors": gpu_vendors,
            "image_type": parsed.image_type or "runtime",
            "role": role,
            "workloads": workloads,
        },
        "cloud": {
            "affinity": ["gcp"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://gcr.io/deeplearning-platform-release/{repo}",
            "documentation": "https://cloud.google.com/deep-learning-containers/docs",
            "source": None,
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": notes,
    }


def build_jupyter_image(
    tag_info: TagInfo,
    parsed: ParsedTag,
    org: str = "jupyter",
    repo: str = "datascience-notebook",
) -> dict[str, Any]:
    """Build a catalog entry for Jupyter Docker Stack images.

    ID pattern: jupyter-{repo}-{tag}
    Examples:
        - jupyter-datascience-notebook-python-3-13
        - jupyter-scipy-notebook-latest
        - jupyter-tensorflow-notebook-python-3-11
    """
    # Sanitize repo and tag for ID
    repo_id = repo.replace("-", "-")
    tag_id = _version_to_id_part(tag_info.name)
    image_id = f"jupyter-{repo_id}-{tag_id}"
    full_name = f"quay.io/{org}/{repo}:{tag_info.name}"

    # Determine Python version from parsed data or tag
    python_version = None
    if parsed.flavor and parsed.flavor.startswith("python-"):
        python_version = parsed.flavor.replace("python-", "")
    elif parsed.framework_version and parsed.framework_version != "latest":
        # If framework_version looks like a Python version
        if parsed.framework == "jupyter" and "." in parsed.framework_version:
            python_version = parsed.framework_version

    # Determine OS info
    os_name = parsed.os_name or "ubuntu"
    os_version = parsed.os_version or "24.04"

    # Determine workloads based on repo type
    if "tensorflow" in repo:
        workloads = ["classical-ml", "computer-vision", "nlp", "generic"]
        frameworks = [
            {"name": "jupyter", "version": "varies-by-tag"},
            {"name": "tensorflow", "version": "varies-by-tag"},
        ]
    elif "pytorch" in repo:
        workloads = ["llm", "computer-vision", "nlp", "generic"]
        frameworks = [
            {"name": "jupyter", "version": "varies-by-tag"},
            {"name": "pytorch", "version": "varies-by-tag"},
        ]
    elif "pyspark" in repo:
        workloads = ["classical-ml", "generic"]
        frameworks = [
            {"name": "jupyter", "version": "varies-by-tag"},
            {"name": "pyspark", "version": "varies-by-tag"},
        ]
    elif "datascience" in repo:
        workloads = ["classical-ml", "scientific-computing", "generic"]
        frameworks = [
            {"name": "jupyter", "version": "varies-by-tag"},
            {"name": "numpy", "version": "varies-by-tag"},
            {"name": "pandas", "version": "varies-by-tag"},
            {"name": "scikit-learn", "version": "varies-by-tag"},
        ]
    elif "scipy" in repo:
        workloads = ["classical-ml", "scientific-computing", "generic"]
        frameworks = [
            {"name": "jupyter", "version": "varies-by-tag"},
            {"name": "scipy", "version": "varies-by-tag"},
        ]
    else:
        workloads = ["generic"]
        frameworks = [{"name": "jupyter", "version": "varies-by-tag"}]

    return {
        "id": image_id,
        "name": full_name,
        "metadata": {
            "status": "official",
            "provider": "jupyter",
            "registry": "quay",
            "maintenance": "active",
            "last_updated": _truncate_date(tag_info.last_updated),
            "license": "BSD-3-Clause",
        },
        "cuda": None,  # Jupyter stacks are CPU-only
        "runtime": {
            "python": python_version,
            "os": {"name": os_name, "version": os_version},
            "architectures": tag_info.architectures or ["amd64", "arm64"],
        },
        "frameworks": frameworks,
        "capabilities": {
            "gpu_vendors": ["none"],
            "image_type": "runtime",
            "role": "notebook",
            "workloads": workloads,
        },
        "cloud": {
            "affinity": ["any"],
            "exclusive_to": None,
            "aws_ami": None,
            "gcp_image": None,
            "azure_image": None,
        },
        "security": None,
        "size": {
            "compressed_mb": tag_info.compressed_size_mb,
            "uncompressed_mb": None,
        },
        "urls": {
            "registry": f"https://quay.io/repository/{org}/{repo}",
            "documentation": "https://jupyter-docker-stacks.readthedocs.io/en/latest/",
            "source": f"https://github.com/{org}/docker-stacks",
        },
        "recommended_for": [],
        "system_packages": [],
        "notes": None,
    }


# Registry mapping parser names to builder functions
BUILDER_REGISTRY: dict[str, type] = {
    "pytorch_cuda": build_pytorch_image,
    "pytorch_cpu": build_pytorch_image,
    "tensorflow_tf": build_tensorflow_image,
    "vllm_simple": build_vllm_image,
    "ollama_simple": build_ollama_image,
    "nvidia_cuda": build_nvidia_cuda_image,
    "tgi_simple": build_tgi_image,
    "ngc_pytorch": build_ngc_pytorch_image,
    "ngc_tensorflow": build_ngc_tensorflow_image,
    "ngc_triton": build_ngc_triton_image,
    "ngc_jax": build_ngc_jax_image,
    "ngc_nemo": build_ngc_nemo_image,
    "ngc_rapids": build_ngc_rapids_image,
    "jupyter_stack": build_jupyter_image,
    "aws_dlc": build_aws_dlc_image,
    "gcp_dlc": build_gcp_dlc_image,
}


def get_builder(builder_name: str):
    """Get a builder function by name.

    Args:
        builder_name: Name of the builder (matches parser name)

    Returns:
        Builder function

    Raises:
        KeyError: If builder not found
    """
    return BUILDER_REGISTRY[builder_name]
