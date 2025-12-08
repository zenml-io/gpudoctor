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
