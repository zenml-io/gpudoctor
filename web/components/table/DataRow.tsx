'use client';

import { useRouter } from 'next/navigation';

import type { ImageEntry, ImageProvider } from '@/lib/types/images';
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
        {cudaVersion}
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {pythonVersion}
      </td>
      <td className="px-4 py-3 align-top text-xs">
        <StatusPill status={image.metadata.maintenance} />
      </td>
      <td className="px-4 py-3 align-top text-xs text-neutral-800">
        {formatProviderLabel(image.metadata.provider)}
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