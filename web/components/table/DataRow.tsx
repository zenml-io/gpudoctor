'use client';

import { useRouter } from 'next/navigation';

import type { ImageEntry, ImageProvider, ImageRole, ImageType, RuntimeOs, CpuArchitecture, ImageSize, CudaConfig } from '@/lib/types/images';
import { StatusPill } from '@/components/ui/StatusPill';
import { getImageDisplayName } from '@/lib/data/images.client';

interface DataRowProps {
  image: ImageEntry;
}

/**
 * Single table row showing key image attributes.
 * The entire row is clickable and navigates to the image detail page.
 */
export function DataRow({ image }: DataRowProps) {
  const router = useRouter();

  const primaryFramework = image.frameworks[0] ?? null;
  const cudaVersion = image.cuda?.version ?? '—';
  const pythonVersion = image.runtime.python ?? '—';

  function handleNavigate() {
    router.push(`/images/${image.id}`);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTableRowElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleNavigate();
    }
  }

  return (
    <tr
      onClick={handleNavigate}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      className="cursor-pointer border-b border-neutral-200 last:border-b-0 hover:bg-primary-50/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400"
    >
      <td className="px-4 py-3 align-top">
        <div className="max-w-xs truncate font-mono text-xs text-neutral-900">
          {image.name}
        </div>
        <div className="mt-0.5 text-xs text-neutral-500">
          {getImageDisplayName(image)}
        </div>
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {primaryFramework
          ? `${formatFrameworkLabel(primaryFramework.name)} ${primaryFramework.version}`
          : '—'}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatRole(image.capabilities.role)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatImageType(image.capabilities.image_type)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {cudaVersion}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatCudnn(image.cuda)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {pythonVersion}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatOs(image.runtime.os)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatArchitectures(image.runtime.architectures)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800 text-right">
        {formatSize(image.size)}
      </td>
      <td className="px-4 py-3 align-top text-xs">
        <StatusPill status={image.metadata.maintenance} />
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatProviderLabel(image.metadata.provider)}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatLicense(image.metadata.license)}
      </td>
    </tr>
  );
}

function formatFrameworkLabel(rawName: string): string {
  const normalized = rawName.toLowerCase();

  const known: Record<string, string> = {
    pytorch: 'PyTorch',
    torchvision: 'TorchVision',
    torchaudio: 'Torchaudio',
    tensorflow: 'TensorFlow',
    keras: 'Keras',
    jax: 'JAX',
    vllm: 'vLLM',
    ollama: 'Ollama',
    jupyter: 'Jupyter',
    transformers: 'Transformers',
    tensorrt: 'TensorRT',
    'tensorrt-llm': 'TensorRT-LLM',
    'text-generation-inference': 'Text Generation Inference',
    'triton-inference-server': 'Triton Inference Server',
    'transformer-engine': 'Transformer Engine',
    apex: 'APEX'
  };

  if (normalized in known) {
    return known[normalized];
  }

  return normalized
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(' ');
}

function formatProviderLabel(provider: ImageProvider): string {
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

function formatRole(role: ImageRole): string {
  const labels: Record<ImageRole, string> = {
    base: 'Base',
    training: 'Training',
    inference: 'Inference',
    notebook: 'Notebook',
    serving: 'Serving'
  };
  return labels[role];
}

function formatImageType(t: ImageType): string {
  const labels: Record<ImageType, string> = {
    base: 'Base',
    runtime: 'Runtime',
    devel: 'Devel'
  };
  return labels[t];
}

function formatOs(os: RuntimeOs): string {
  const osNameMap: Record<RuntimeOs['name'], string> = {
    ubuntu: 'Ubuntu',
    debian: 'Debian',
    centos: 'CentOS',
    rhel: 'RHEL',
    alpine: 'Alpine',
    rockylinux: 'Rocky Linux'
  };
  const name = osNameMap[os.name] ?? os.name;
  return `${name} ${os.version}`;
}

function formatArchitecture(arch: CpuArchitecture): string {
  const labels: Record<CpuArchitecture, string> = {
    amd64: 'x86_64',
    arm64: 'ARM64'
  };
  return labels[arch];
}

function formatArchitectures(archs: CpuArchitecture[]): string {
  if (!archs || archs.length === 0) {
    return '—';
  }
  const sorted = [...archs].sort();
  return sorted.map(formatArchitecture).join(' / ');
}

function formatSize(size: ImageSize | null): string {
  if (!size || size.compressed_mb == null) {
    return '—';
  }
  const mb = size.compressed_mb;
  if (mb >= 1024) {
    const gb = mb / 1024;
    return `${gb.toFixed(1)} GB`;
  }
  return `${mb} MB`;
}

function formatCudnn(cuda: CudaConfig | null): string {
  if (!cuda || !cuda.cudnn) {
    return '—';
  }
  return cuda.cudnn;
}

function formatLicense(license?: string | null): string {
  return license ?? '—';
}