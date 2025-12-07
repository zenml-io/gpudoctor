import type {
  ImageEntry,
  ImageProvider,
  Workload
} from '@/lib/types/images';

/**
 * Returns a sorted list of unique framework names across all images.
 * Names are normalized to lowercase so filtering can be case-insensitive.
 */
export function getAllFrameworkNames(images: ImageEntry[]): string[] {
  const names = new Set<string>();
  for (const image of images) {
    for (const framework of image.frameworks) {
      names.add(framework.name.toLowerCase());
    }
  }
  return Array.from(names).sort((a, b) => a.localeCompare(b));
}

/**
 * Returns a sorted list of unique providers found in image metadata.
 */
export function getAllProviders(images: ImageEntry[]): ImageProvider[] {
  const providers = new Set<ImageProvider>();
  for (const image of images) {
    providers.add(image.metadata.provider);
  }
  return Array.from(providers).sort((a, b) => a.localeCompare(b));
}

/**
 * Returns a sorted list of unique CUDA versions across all images.
 * CPU-only images are ignored.
 */
export function getAllCudaVersions(images: ImageEntry[]): string[] {
  const versions = new Set<string>();
  for (const image of images) {
    if (image.cuda?.version) {
      versions.add(image.cuda.version);
    }
  }
  return Array.from(versions).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

/**
 * Returns a sorted list of unique workloads across all images.
 */
export function getAllWorkloads(images: ImageEntry[]): Workload[] {
  const workloads = new Set<Workload>();
  for (const image of images) {
    for (const workload of image.capabilities.workloads) {
      workloads.add(workload);
    }
  }
  return Array.from(workloads).sort((a, b) => a.localeCompare(b));
}

export function getAllPythonVersions(images: ImageEntry[]): string[] {
  const versions = new Set<string>();
  for (const image of images) {
    const py = image.runtime.python;
    if (py) {
      versions.add(py);
    }
  }
  return Array.from(versions).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true })
  );
}

/**
 * Returns a short, human-friendly display name for an image.
 * This is intended for cards and tables where the full Docker tag is too verbose.
 */
export function getImageDisplayName(image: ImageEntry): string {
  const primaryFramework = image.frameworks[0];
  const cudaLabel = image.cuda?.version ? `CUDA ${image.cuda.version}` : 'CPU-only';

  if (primaryFramework) {
    const frameworkName = formatFrameworkName(primaryFramework.name);
    const version = primaryFramework.version;
    return `${frameworkName} ${version} · ${cudaLabel}`;
  }

  const providerLabel = formatProviderName(image.metadata.provider);
  return `${providerLabel} image · ${cudaLabel}`;
}

function formatFrameworkName(rawName: string): string {
  const normalized = rawName.toLowerCase();

  const knownNames: Record<string, string> = {
    pytorch: 'PyTorch',
    torchvision: 'TorchVision',
    torchaudio: 'Torchaudio',
    tensorflow: 'TensorFlow',
    keras: 'Keras',
    jax: 'JAX',
    vllm: 'vLLM',
    ollama: 'Ollama',
    'llama.cpp': 'llama.cpp',
    jupyter: 'Jupyter',
    transformers: 'Transformers',
    tensorrt: 'TensorRT',
    'tensorrt-llm': 'TensorRT-LLM',
    'text-generation-inference': 'Text Generation Inference',
    'triton-inference-server': 'Triton Inference Server',
    'transformer-engine': 'Transformer Engine',
    apex: 'APEX'
  };

  if (normalized in knownNames) {
    return knownNames[normalized];
  }

  // Fallback: title-case the name by splitting on dashes/underscores.
  return normalized
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatProviderName(provider: ImageProvider): string {
  const mapping: Record<ImageProvider, string> = {
    'nvidia-ngc': 'NVIDIA NGC',
    'aws-dlc': 'AWS DLC',
    'gcp-dlc': 'Google Cloud DLC',
    'azure-ml': 'Azure ML',
    pytorch: 'PyTorch',
    tensorflow: 'TensorFlow',
    jax: 'JAX',
    huggingface: 'Hugging Face',
    vllm: 'vLLM',
    ollama: 'Ollama',
    jupyter: 'Jupyter',
    community: 'Community'
  };

  return mapping[provider];
}