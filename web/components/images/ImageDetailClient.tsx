'use client';

import Link from 'next/link';

import type { ImageEntry, SecurityRating } from '@/lib/types/images';
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

/** Map security rating to a badge variant. */
function ratingVariant(rating: SecurityRating): 'success' | 'warning' | 'error' | 'default' {
  switch (rating) {
    case 'A':
    case 'B':
      return 'success';
    case 'C':
      return 'warning';
    case 'D':
    case 'F':
      return 'error';
    default:
      return 'default';
  }
}

/** Format file size with appropriate unit. */
function formatSize(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${Math.round(mb)} MB`;
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
            {image.size && (
              <Badge size="sm">
                {formatSize(image.size.compressed_mb)} compressed
              </Badge>
            )}
            {image.metadata.license && (
              <Badge size="sm" variant="default">
                {image.metadata.license}
              </Badge>
            )}
            {primaryFramework && (
              <Badge size="sm">
                {formatFrameworkLabel(primaryFramework.name)}{' '}
                {primaryFramework.version}
              </Badge>
            )}
            {image.metadata.last_updated && (
              <Badge size="sm" variant="default">
                Updated {image.metadata.last_updated}
              </Badge>
            )}
          </div>

          {image.recommended_for.length > 0 && (
            <ul className="max-w-2xl list-inside list-disc space-y-0.5 text-sm text-neutral-600">
              {image.recommended_for.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <ImageQuickStart image={image} />

      <ImageSpecs image={image} />

      <Card className="space-y-4">
        <h2 className="text-sm font-semibold text-neutral-900">
          Security overview
        </h2>
        {image.security ? (
          <div className="space-y-4">
            {/* Rating and summary */}
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant={ratingVariant(image.security.rating)} size="md">
                Rating: {image.security.rating}
              </Badge>
              <span className="text-sm text-neutral-600">
                {image.security.total_cves} total CVEs found
              </span>
            </div>

            {/* CVE breakdown by severity */}
            {(image.security.critical !== undefined ||
              image.security.high !== undefined ||
              image.security.medium !== undefined ||
              image.security.low !== undefined) && (
              <div className="flex flex-wrap gap-2">
                {image.security.critical !== undefined && image.security.critical > 0 && (
                  <Badge variant="error" size="sm">
                    {image.security.critical} Critical
                  </Badge>
                )}
                {image.security.high !== undefined && image.security.high > 0 && (
                  <Badge variant="error" size="sm">
                    {image.security.high} High
                  </Badge>
                )}
                {image.security.medium !== undefined && image.security.medium > 0 && (
                  <Badge variant="warning" size="sm">
                    {image.security.medium} Medium
                  </Badge>
                )}
                {image.security.low !== undefined && image.security.low > 0 && (
                  <Badge variant="default" size="sm">
                    {image.security.low} Low
                  </Badge>
                )}
                {image.security.critical === 0 &&
                  image.security.high === 0 &&
                  image.security.medium === 0 &&
                  image.security.low === 0 && (
                    <Badge variant="success" size="sm">
                      No known vulnerabilities
                    </Badge>
                  )}
              </div>
            )}

            {/* Scan metadata */}
            <div className="flex flex-wrap gap-4 text-xs text-neutral-500">
              <span>Last scan: {image.security.last_scan}</span>
              <span>Scanner: {image.security.scanner.toUpperCase()}</span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-neutral-600">
            This image has not been scanned yet. Security data will appear here
            once a vulnerability scan has been performed.
          </p>
        )}
      </Card>

      {/* Metadata & Links */}
      <Card className="space-y-3">
        <h2 className="text-sm font-semibold text-neutral-900">
          Metadata & links
        </h2>
        <div className="flex flex-wrap gap-3">
          <a
            href={image.urls.registry}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 hover:underline"
          >
            Registry
            <ExternalLinkIcon />
          </a>
          {image.urls.documentation && (
            <a
              href={image.urls.documentation}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 hover:underline"
            >
              Documentation
              <ExternalLinkIcon />
            </a>
          )}
          {image.urls.source && (
            <a
              href={image.urls.source}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 hover:underline"
            >
              Source
              <ExternalLinkIcon />
            </a>
          )}
        </div>
        {image.size && (
          <div className="text-sm text-neutral-600">
            <span className="font-medium">Size:</span>{' '}
            {formatSize(image.size.compressed_mb)} compressed
            {image.size.uncompressed_mb && (
              <> / {formatSize(image.size.uncompressed_mb)} uncompressed</>
            )}
          </div>
        )}
      </Card>

      {/* System packages & notes */}
      {(image.system_packages.length > 0 || image.notes) && (
        <Card className="space-y-3">
          <h2 className="text-sm font-semibold text-neutral-900">
            System packages & notes
          </h2>
          {image.system_packages.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs font-medium text-neutral-500 uppercase tracking-wide">
                Notable packages
              </p>
              <div className="flex flex-wrap gap-1.5">
                {image.system_packages.map((pkg) => (
                  <Badge key={pkg} size="sm" variant="default">
                    {pkg}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {image.notes && (
            <div className="space-y-1">
              <p className="text-xs font-medium text-neutral-500 uppercase tracking-wide">
                Notes
              </p>
              <p className="text-sm text-neutral-700">{image.notes}</p>
            </div>
          )}
        </Card>
      )}

      <SimilarImages currentImage={image} images={similarImages} />
    </div>
  );
}

/** Small external link icon for links. */
function ExternalLinkIcon() {
  return (
    <svg
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
      />
    </svg>
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