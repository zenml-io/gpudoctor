import Link from 'next/link';

import type { ImageEntry } from '@/lib/types/images';
import type {
  GuideState,
  GuideWorkload,
  GuidePriorityKey
} from '@/lib/url/guideSearchParams';
import { EmptyState } from '@/components/ui/EmptyState';
import { ImageCard } from '@/components/guide/ImageCard';

interface GuideResultsProps {
  images: ImageEntry[];
  state: GuideState;
}

/**
 * Displays the recommended images for the current guide state.
 * The highest-scoring image is highlighted as the best match, with a short
 * summary of the filters used to derive the recommendations.
 */
export function GuideResults({ images, state }: GuideResultsProps) {
  const hasResults = images.length > 0;

  if (!hasResults) {
    return (
      <section id="guide-results" aria-label="Recommended images" className="space-y-3">
        <h2 className="text-lg font-semibold text-neutral-900">
          Recommended images
        </h2>
        <EmptyState
          title="No images match your current selections"
          description="Try removing one or more filters or choose a more general workload."
        />
      </section>
    );
  }

  const [best, ...rest] = images;

  const summaryParts: string[] = [];

  if (state.workload) {
    summaryParts.push(formatWorkloadLabel(state.workload));
  }
  if (state.frameworks.length > 0) {
    summaryParts.push(formatFrameworkSummary(state.frameworks));
  }
  if (state.clouds.length > 0) {
    summaryParts.push(formatCloudSummary(state.clouds));
  }

  if (state.gpuPreference === 'gpu-required') {
    summaryParts.push('GPU-enabled images');
  } else if (state.gpuPreference === 'cpu-only') {
    summaryParts.push('CPU-only images');
  }

  if (state.licensePreference === 'oss-only') {
    summaryParts.push('Open-source licenses only');
  }

  if (state.pythonVersion) {
    summaryParts.push(`Python ${state.pythonVersion}`);
  }

  if (state.minSecurityRating) {
    summaryParts.push(`Security rating ≥ ${state.minSecurityRating}`);
  }

  // Show priorities if customized - check first two priorities
  const defaultTopTwo: GuidePriorityKey[] = ['security', 'gpu'];
  const currentTopTwo = state.priorities.slice(0, 2);
  const prioritiesCustomized =
    currentTopTwo[0] !== defaultTopTwo[0] ||
    currentTopTwo[1] !== defaultTopTwo[1];

  if (prioritiesCustomized) {
    const topLabels = currentTopTwo.map(formatPriorityLabel).join(' & ');
    summaryParts.push(`Prioritizing ${topLabels}`);
  }

  const summary =
    summaryParts.length > 0 ? summaryParts.join(' + ') : 'All available images';

  return (
    <section id="guide-results" aria-label="Recommended images" className="space-y-4">
      <div className="flex items-baseline justify-between gap-3">
        <div className="space-y-1">
          <h2 className="text-lg font-semibold text-neutral-900">
            Recommended images
          </h2>
          <p className="text-sm text-neutral-600">Based on: {summary}</p>
        </div>
        <Link
          href="/guide"
          className="text-xs font-medium text-neutral-500 hover:text-neutral-800"
        >
          Modify filters
        </Link>
      </div>

      <div className="space-y-4">
        <ImageCard image={best} highlight="best" />
        {rest.slice(0, 4).map((image) => (
          <ImageCard key={image.id} image={image} />
        ))}

        {images.length > 5 && (
          <div>
            <Link
              href="/table"
              className="text-sm font-medium text-primary-600 hover:text-primary-700"
            >
              View all {images.length} matching images in table →
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}

function formatWorkloadLabel(workload: GuideWorkload): string {
  switch (workload) {
    case 'computer-vision':
      return 'Computer Vision';
    case 'llm-train':
      return 'LLM / Text Generation';
    case 'llm-inference':
      return 'LLM Inference / Serving';
    case 'multimodal':
      return 'Multimodal';
    case 'classical-ml':
      return 'Classical ML';
    case 'general':
      return 'General / Experimentation';
    default:
      return workload;
  }
}

function formatFrameworkSummary(frameworks: string[]): string {
  const unique = Array.from(new Set(frameworks.map((fw) => fw.toLowerCase())));
  const labels = unique.map(formatFrameworkLabel);
  if (labels.length <= 2) {
    return labels.join(', ');
  }
  const shown = labels.slice(0, 2).join(', ');
  const extra = labels.length - 2;
  return `${shown} + ${extra} more`;
}

function formatFrameworkLabel(rawName: string): string {
  const normalized = rawName.toLowerCase();

  const knownNames: Record<string, string> = {
    pytorch: 'PyTorch',
    tensorflow: 'TensorFlow',
    jax: 'JAX',
    vllm: 'vLLM',
    ollama: 'Ollama',
    jupyter: 'Jupyter',
    transformers: 'Transformers',
    tensorrt: 'TensorRT',
    'tensorrt-llm': 'TensorRT-LLM',
    'text-generation-inference': 'Text Generation Inference',
    'triton-inference-server': 'Triton Inference Server'
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

function formatCloudSummary(clouds: string[]): string {
  const labels: Record<string, string> = {
    aws: 'AWS',
    gcp: 'Google Cloud',
    azure: 'Azure',
    any: 'Any / self-hosted'
  };

  const unique = Array.from(new Set(clouds));
  return unique
    .map((cloud) => labels[cloud] ?? cloud.toUpperCase())
    .join(', ');
}

function formatPriorityLabel(key: GuidePriorityKey): string {
  const labels: Record<GuidePriorityKey, string> = {
    security: 'Security',
    size: 'Size',
    license: 'Licensing',
    gpu: 'GPU',
    cloud: 'Cloud',
    freshness: 'Freshness'
  };
  return labels[key] ?? key;
}