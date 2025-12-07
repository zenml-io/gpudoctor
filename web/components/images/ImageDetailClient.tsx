'use client';

import Link from 'next/link';

import type { ImageEntry } from '@/lib/types/images';
import { Badge } from '@/components/ui/Badge';
import { StatusPill } from '@/components/ui/StatusPill';
import { Card } from '@/components/ui/Card';
import { ImageQuickStart } from '@/components/images/ImageQuickStart';
import { ImageSpecs } from '@/components/images/ImageSpecs';
import { SimilarImages } from '@/components/images/SimilarImages';

interface ImageDetailClientProps {
  image: ImageEntry;
  allImages: ImageEntry[];
}

/**
 * Client-side container for the image detail page.
 * Shows metadata, quick start instructions, specifications, security, and similar images.
 */
export function ImageDetailClient({ image, allImages }: ImageDetailClientProps) {
  const similarImages = getSimilarImages(image, allImages);
  const primaryFramework = image.frameworks[0] ?? null;

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <Link
          href="/table"
          className="inline-flex items-center text-xs font-medium text-neutral-600 hover:text-neutral-900"
        >
          <span aria-hidden="true" className="mr-1">
            ←
          </span>
          Back to all images
        </Link>

        <div className="space-y-3">
          <div className="space-y-1">
            <p className="font-mono text-[11px] uppercase tracking-wide text-neutral-500">
              {image.metadata.registry.toUpperCase()} •{' '}
              {formatProviderLabel(image.metadata.provider)}
            </p>
            <h1 className="text-xl font-semibold tracking-tight text-neutral-900 sm:text-2xl">
              {image.name}
            </h1>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge variant="purple" size="sm">
              {formatStatusBadgeLabel(image.metadata.status)}
            </Badge>
            <StatusPill status={image.metadata.maintenance} />
            {image.size?.compressed_mb && (
              <Badge size="sm">
                {(image.size.compressed_mb / 1024).toFixed(1)} GB compressed
              </Badge>
            )}
            {primaryFramework && (
              <Badge size="sm">
                {formatFrameworkLabel(primaryFramework.name)}{' '}
                {primaryFramework.version}
              </Badge>
            )}
          </div>

          {image.recommended_for[0] && (
            <p className="max-w-2xl text-sm text-neutral-600">
              {image.recommended_for[0]}
            </p>
          )}
        </div>
      </div>

      <ImageQuickStart image={image} />

      <ImageSpecs image={image} />

      <Card className="space-y-3">
        <h2 className="text-sm font-semibold text-neutral-900">
          Security overview
        </h2>
        {image.security ? (
          <div className="grid gap-3 text-sm text-neutral-700 sm:grid-cols-2">
            <div className="space-y-1">
              <p>
                Rating:{' '}
                <span className="font-medium">{image.security.rating}</span>
              </p>
              <p>
                Total CVEs:{' '}
                <span className="font-medium">{image.security.total_cves}</span>
              </p>
            </div>
            <div className="space-y-1">
              <p>
                Last scan:{' '}
                <span className="font-medium">{image.security.last_scan}</span>
              </p>
              <p>
                Scanner:{' '}
                <span className="font-medium">
                  {image.security.scanner.toUpperCase()}
                </span>
              </p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-neutral-600">
            This image has not been scanned yet. Security data will appear here
            once a vulnerability scan has been performed.
          </p>
        )}
      </Card>

      <SimilarImages currentImage={image} images={similarImages} />
    </div>
  );
}

function getSimilarImages(image: ImageEntry, allImages: ImageEntry[]): ImageEntry[] {
  const currentFrameworks = new Set(
    image.frameworks.map((f) => f.name.toLowerCase())
  );

  const candidates = allImages.filter((other) => {
    if (other.id === image.id) {
      return false;
    }

    if (other.metadata.provider === image.metadata.provider) {
      return true;
    }

    const otherFrameworks = other.frameworks.map((f) =>
      f.name.toLowerCase()
    );
    return otherFrameworks.some((name) => currentFrameworks.has(name));
  });

  // Limit to a reasonable number for the UI.
  return candidates.slice(0, 6);
}

function formatStatusBadgeLabel(
  status: ImageEntry['metadata']['status']
): string {
  switch (status) {
    case 'official':
      return 'Official';
    case 'community':
      return 'Community';
    case 'verified':
      return 'Verified';
    default:
      return status;
  }
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

function formatProviderLabel(
  provider: ImageEntry['metadata']['provider']
): string {
  const mapping: Record<ImageEntry['metadata']['provider'], string> = {
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