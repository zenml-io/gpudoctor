import Link from 'next/link';
import clsx from 'clsx';

import type { ImageEntry, ImageProvider } from '@/lib/types/images';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StatusPill } from '@/components/ui/StatusPill';
import { getImageDisplayName } from '@/lib/data/images.client';

interface ImageCardProps {
  image: ImageEntry;
  /**
   * When set to "best", the card is visually highlighted as the top recommendation.
   */
  highlight?: 'best';
}

/**
 * Compact card representation of an image used in the guide results.
 * Shows the full Docker tag, a concise display name, key runtime badges,
 * maintenance status, and a link to view details.
 */
export function ImageCard({ image, highlight }: ImageCardProps) {
  const primaryFramework = image.frameworks[0];
  const cudaVersion = image.cuda?.version ?? null;
  const pythonVersion = image.runtime.python;
  const isBest = highlight === 'best';

  return (
    <Card
      padding="md"
      className={clsx(
        'flex flex-col gap-3 border',
        isBest
          ? 'border-primary-400 shadow-md ring-1 ring-primary-200'
          : 'border-neutral-200'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <p className="truncate font-mono text-xs text-neutral-900">
            {image.name}
          </p>
          <p className="text-xs text-neutral-500">
            {getImageDisplayName(image)}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          {isBest && (
            <Badge variant="purple" size="sm">
              Best match
            </Badge>
          )}
          <StatusPill status={image.metadata.maintenance} />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {primaryFramework && (
          <Badge size="sm">
            {formatFrameworkLabel(primaryFramework.name)}{' '}
            <span className="ml-1 text-[11px] text-neutral-500">
              {primaryFramework.version}
            </span>
          </Badge>
        )}
        {cudaVersion && <Badge size="sm">CUDA {cudaVersion}</Badge>}
        {pythonVersion && <Badge size="sm">Python {pythonVersion}</Badge>}
        <Badge size="sm" variant="default">
          {formatProviderLabel(image.metadata.provider)}
        </Badge>
      </div>

      <div className="flex items-center justify-between gap-3 text-xs">
        <p className="line-clamp-1 text-neutral-500">
          {image.recommended_for[0] ?? 'No description provided.'}
        </p>
        <Link
          href={`/images/${image.id}`}
          className="shrink-0 text-xs font-medium text-primary-600 hover:text-primary-700"
        >
          View details
        </Link>
      </div>
    </Card>
  );
}

function formatFrameworkLabel(rawName: string): string {
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

  return normalized
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
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