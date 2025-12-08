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
        flavor="gpu",
    )


def parse_pytorch_cpu(tag: str) -> ParsedTag | None:
    """Parse PyTorch official CPU image tags.

    Examples:
        - "2.5.1-cpu" -> pytorch 2.5.1, CPU
        - "2.6.0-py3.11-cpu" -> pytorch 2.6.0, CPU with Python 3.11

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match expected pattern
    """
    # Pattern: VERSION[-pyPYTHON]-cpu
    pattern = r"^(\d+\.\d+\.\d+)(?:-py(\d+\.\d+))?-cpu$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="pytorch",
        framework_version=match.group(1),
        cuda_version=None,
        cudnn_version=None,
        image_type="runtime",
        flavor="cpu",
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
        - "3.3.5" -> TGI 3.3.5 (CUDA defaulted)
        - "3.3.5-cuda12.4" -> TGI 3.3.5, CUDA 12.4
        - "latest" -> TGI latest

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info
    """
    # Try to match version-cuda pattern first
    cuda_pattern = r"^(\d+\.\d+\.\d+)-cuda(\d+\.\d+)$"
    cuda_match = re.match(cuda_pattern, tag)
    if cuda_match:
        return ParsedTag(
            framework="text-generation-inference",
            framework_version=cuda_match.group(1),
            cuda_version=cuda_match.group(2),
            image_type="runtime",
        )

    # Otherwise, treat as bare version tag
    return ParsedTag(
        framework="text-generation-inference",
        framework_version=tag,
        cuda_version=None,  # Builder will default to 12.4
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
    # OS name is letters only, OS version is digits and dots
    pattern = r"^(\d+\.\d+\.\d+)(?:-(cudnn)(\d+)?)?-(runtime|devel|base)-([a-zA-Z]+)(\d+\.\d+)$"
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


def parse_ngc_jax(tag: str) -> ParsedTag | None:
    """Parse NVIDIA NGC JAX image tags.

    Examples:
        - "24.10-py3" -> Release 24.10, Python 3

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
        framework="jax",
        release=match.group(1),
        image_type="devel",
    )


def parse_ngc_nemo(tag: str) -> ParsedTag | None:
    """Parse NVIDIA NeMo Framework image tags.

    Examples:
        - "24.12" -> Release 24.12
        - "25.01" -> Release 25.01

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: YY.MM or YY.MM.N
    pattern = r"^(\d+\.\d+)(?:\.\d+)?$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="nemo",
        release=match.group(1),
        image_type="devel",
    )


def parse_ngc_rapids(tag: str) -> ParsedTag | None:
    """Parse NVIDIA RAPIDS image tags.

    Examples:
        - "24.10-cuda12.5-py3.12" -> Release 24.10, CUDA 12.5, Python 3.12
        - "24.06-cuda12.2-py3.11" -> Release 24.06, CUDA 12.2, Python 3.11

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # Pattern: YY.MM-cudaVER-pyVER
    pattern = r"^(\d+\.\d+)-cuda(\d+\.\d+)-py(\d+\.\d+)$"
    match = re.match(pattern, tag)

    if not match:
        return None

    return ParsedTag(
        framework="rapids",
        release=match.group(1),
        cuda_version=match.group(2),
        image_type="runtime",
    )


def parse_aws_dlc(tag: str) -> ParsedTag | None:
    """Parse AWS Deep Learning Container image tags.

    AWS DLC tags follow the format:
        {version}-{device}-py{python}-cu{cuda}-{os}-{platform}

    Examples:
        - "2.7.1-gpu-py312-cu128-ubuntu22.04-ec2"
        - "2.6.0-cpu-py311-ubuntu22.04-sagemaker"
        - "2.18.0-gpu-py310-cu125-ubuntu22.04-ec2"

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info, or None if tag doesn't match
    """
    # GPU pattern: VERSION-gpu-pyPYTHON-cuCUDA-OS-PLATFORM
    gpu_pattern = r"^(\d+\.\d+(?:\.\d+)?)-gpu-py(\d+)-cu(\d+)-ubuntu(\d+\.\d+)-(ec2|sagemaker)$"
    gpu_match = re.match(gpu_pattern, tag)

    if gpu_match:
        version = gpu_match.group(1)
        python_version = f"3.{gpu_match.group(2)[-2:]}" if len(gpu_match.group(2)) == 3 else f"3.{gpu_match.group(2)}"
        cuda_raw = gpu_match.group(3)
        # Convert cuda "128" to "12.8", "121" to "12.1", etc.
        cuda_version = f"{cuda_raw[:-1]}.{cuda_raw[-1]}" if len(cuda_raw) == 3 else f"{cuda_raw[0]}.{cuda_raw[1]}"
        os_version = gpu_match.group(4)
        platform = gpu_match.group(5)

        return ParsedTag(
            framework_version=version,
            cuda_version=cuda_version,
            cudnn_version="9",  # AWS DLCs typically use cuDNN 9
            image_type="runtime",
            flavor=f"gpu-{platform}",
            os_name="ubuntu",
            os_version=os_version,
        )

    # CPU pattern: VERSION-cpu-pyPYTHON-OS-PLATFORM
    cpu_pattern = r"^(\d+\.\d+(?:\.\d+)?)-cpu-py(\d+)-ubuntu(\d+\.\d+)-(ec2|sagemaker)$"
    cpu_match = re.match(cpu_pattern, tag)

    if cpu_match:
        version = cpu_match.group(1)
        python_version = f"3.{cpu_match.group(2)[-2:]}" if len(cpu_match.group(2)) == 3 else f"3.{cpu_match.group(2)}"
        os_version = cpu_match.group(3)
        platform = cpu_match.group(4)

        return ParsedTag(
            framework_version=version,
            cuda_version=None,
            image_type="runtime",
            flavor=f"cpu-{platform}",
            os_name="ubuntu",
            os_version=os_version,
        )

    return None


def parse_gcp_dlc(tag: str) -> ParsedTag | None:
    """Parse GCP Deep Learning Container repo names and tags.

    GCP DLC repo names encode framework and compute type:
        pytorch-gpu.2-2, tf-gpu.2-15, base-cu121

    The tag itself is usually "latest", a Python version like "py310",
    or a milestone like "m96".

    For GCP DLCs, we actually parse the *repo name* which is passed as the tag
    since the actual tag is typically just "latest" or "py310".

    This parser expects a combined format: REPO:TAG where REPO contains
    the framework info.

    Args:
        tag: Combined repo:tag string or just repo name

    Returns:
        ParsedTag with extracted info, or None if doesn't match
    """
    # This parser is designed for the full repo name which encodes framework/version
    # Examples: pytorch-gpu.2-2, tf-gpu.2-15, pytorch-cpu.2-2, base-cu121

    # PyTorch GPU pattern: pytorch-gpu.MAJOR-MINOR
    pytorch_gpu = re.match(r"^pytorch-gpu\.(\d+)-(\d+)$", tag)
    if pytorch_gpu:
        version = f"{pytorch_gpu.group(1)}.{pytorch_gpu.group(2)}"
        return ParsedTag(
            framework="pytorch",
            framework_version=version,
            cuda_version="12.1",  # Default for recent releases
            image_type="runtime",
            flavor="gpu",
        )

    # PyTorch CPU pattern: pytorch-cpu.MAJOR-MINOR
    pytorch_cpu = re.match(r"^pytorch-cpu\.(\d+)-(\d+)$", tag)
    if pytorch_cpu:
        version = f"{pytorch_cpu.group(1)}.{pytorch_cpu.group(2)}"
        return ParsedTag(
            framework="pytorch",
            framework_version=version,
            cuda_version=None,
            image_type="runtime",
            flavor="cpu",
        )

    # TensorFlow GPU pattern: tf-gpu.MAJOR-MINOR
    tf_gpu = re.match(r"^tf-gpu\.(\d+)-(\d+)$", tag)
    if tf_gpu:
        version = f"{tf_gpu.group(1)}.{tf_gpu.group(2)}"
        return ParsedTag(
            framework="tensorflow",
            framework_version=version,
            cuda_version="12.1",
            image_type="runtime",
            flavor="gpu",
        )

    # TensorFlow CPU pattern: tf-cpu.MAJOR-MINOR
    tf_cpu = re.match(r"^tf-cpu\.(\d+)-(\d+)$", tag)
    if tf_cpu:
        version = f"{tf_cpu.group(1)}.{tf_cpu.group(2)}"
        return ParsedTag(
            framework="tensorflow",
            framework_version=version,
            cuda_version=None,
            image_type="runtime",
            flavor="cpu",
        )

    # Base CUDA image pattern: base-cuXXX
    base_cuda = re.match(r"^base-cu(\d+)$", tag)
    if base_cuda:
        cuda_raw = base_cuda.group(1)
        # Convert "121" to "12.1", "113" to "11.3"
        cuda_version = f"{cuda_raw[:-1]}.{cuda_raw[-1]}" if len(cuda_raw) == 3 else cuda_raw
        return ParsedTag(
            framework=None,
            cuda_version=cuda_version,
            image_type="base",
            flavor="gpu",
        )

    return None


def parse_jupyter_stack(tag: str) -> ParsedTag | None:
    """Parse Jupyter Docker Stacks image tags.

    The Jupyter stacks use semantic tags like:
        - "latest" -> rolling release
        - "python-3.13" or "python-3.13.11" -> Python version pinned
        - "lab-4.5.0" -> JupyterLab version pinned
        - "ubuntu-24.04" -> Ubuntu version pinned
        - "notebook-7.5.0" -> Jupyter Notebook version pinned

    Args:
        tag: The image tag string

    Returns:
        ParsedTag with extracted info
    """
    # Handle "latest" tag
    if tag == "latest":
        return ParsedTag(
            framework="jupyter",
            framework_version="latest",
            image_type="runtime",
            flavor="notebook",
        )

    # Python version tag: python-3.13 or python-3.13.11
    python_pattern = r"^python-((\d+\.\d+)(?:\.\d+)?)$"
    python_match = re.match(python_pattern, tag)
    if python_match:
        return ParsedTag(
            framework="jupyter",
            framework_version=python_match.group(2),  # Major.minor only
            image_type="runtime",
            flavor=f"python-{python_match.group(1)}",
        )

    # JupyterLab version tag: lab-4.5.0
    lab_pattern = r"^lab-(\d+\.\d+\.\d+)$"
    lab_match = re.match(lab_pattern, tag)
    if lab_match:
        return ParsedTag(
            framework="jupyterlab",
            framework_version=lab_match.group(1),
            image_type="runtime",
            flavor="lab",
        )

    # Ubuntu version tag: ubuntu-24.04
    ubuntu_pattern = r"^ubuntu-(\d+\.\d+)$"
    ubuntu_match = re.match(ubuntu_pattern, tag)
    if ubuntu_match:
        return ParsedTag(
            framework="jupyter",
            framework_version=ubuntu_match.group(1),
            image_type="runtime",
            flavor="ubuntu",
            os_name="ubuntu",
            os_version=ubuntu_match.group(1),
        )

    # Notebook version tag: notebook-7.5.0
    notebook_pattern = r"^notebook-(\d+\.\d+\.\d+)$"
    notebook_match = re.match(notebook_pattern, tag)
    if notebook_match:
        return ParsedTag(
            framework="jupyter-notebook",
            framework_version=notebook_match.group(1),
            image_type="runtime",
            flavor="notebook",
        )

    # Fallback for other tags (e.g., commit hashes, dates)
    return ParsedTag(
        framework="jupyter",
        framework_version=tag,
        image_type="runtime",
        flavor="other",
    )


# Registry mapping parser names to functions
PARSER_REGISTRY: dict[str, type] = {
    "pytorch_cuda": parse_pytorch_cuda,
    "pytorch_cpu": parse_pytorch_cpu,
    "tensorflow_tf": parse_tensorflow_tf,
    "vllm_simple": parse_vllm_simple,
    "ollama_simple": parse_ollama_simple,
    "tgi_simple": parse_tgi_simple,
    "nvidia_cuda": parse_nvidia_cuda,
    "ngc_pytorch": parse_ngc_pytorch,
    "ngc_tensorflow": parse_ngc_tensorflow,
    "ngc_triton": parse_ngc_triton,
    "ngc_jax": parse_ngc_jax,
    "ngc_nemo": parse_ngc_nemo,
    "ngc_rapids": parse_ngc_rapids,
    "jupyter_stack": parse_jupyter_stack,
    "aws_dlc": parse_aws_dlc,
    "gcp_dlc": parse_gcp_dlc,
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
