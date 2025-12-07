import type { ReactNode } from 'react';

import type { ImageEntry } from '@/lib/types/images';
import { Card } from '@/components/ui/Card';

interface ImageSpecsProps {
  image: ImageEntry;
}

/**
 * Specifications grid for an image, including framework/runtime on the left
 * and compatibility/requirements on the right.
 */
export function ImageSpecs({ image }: ImageSpecsProps) {
  const primaryFramework = image.frameworks[0] ?? null;
  const cuda = image.cuda;
  const runtime = image.runtime;
  const osLabel = `${runtime.os.name} ${runtime.os.version}`;
  const archLabel = runtime.architectures.join(', ');
  const cloud = image.cloud;

  const cloudAffinity =
    cloud?.affinity && cloud.affinity.length > 0
      ? cloud.affinity.join(', ').toUpperCase()
      : 'Any';
  const exclusive =
    cloud?.exclusive_to && cloud.exclusive_to !== null
      ? cloud.exclusive_to.toUpperCase()
      : null;

  const driverRequirement = cuda?.min_driver
    ? `NVIDIA driver ${cuda.min_driver}+`
    : 'See framework documentation';

  const computeCaps =
    cuda?.compute_capabilities && cuda.compute_capabilities.length > 0
      ? cuda.compute_capabilities.join(', ')
      : 'Not specified';

  const vendors =
    image.capabilities.gpu_vendors.length > 0
      ? image.capabilities.gpu_vendors.join(', ')
      : 'Not specified';

  return (
    <Card className="space-y-4">
      <h2 className="text-sm font-semibold text-neutral-900">
        Specifications & compatibility
      </h2>
      <div className="grid gap-6 md:grid-cols-2">
        <dl className="space-y-2 text-sm">
          <SpecItem label="Framework">
            {primaryFramework
              ? `${formatFrameworkLabel(primaryFramework.name)} ${primaryFramework.version}`
              : 'Not specified'}
          </SpecItem>
          <SpecItem label="CUDA">
            {cuda?.version ??
              (image.capabilities.gpu_vendors.includes('none')
                ? 'CPU-only'
                : 'Not specified')}
          </SpecItem>
          <SpecItem label="cuDNN">
            {cuda?.cudnn ?? 'Not specified'}
          </SpecItem>
          <SpecItem label="Python">{runtime.python ?? 'Not specified'}</SpecItem>
          <SpecItem label="Base OS">{osLabel}</SpecItem>
          <SpecItem label="Architecture">{archLabel}</SpecItem>
        </dl>

        <dl className="space-y-2 text-sm">
          <SpecItem label="Cloud compatibility">
            {exclusive ? `${exclusive} only` : cloudAffinity}
          </SpecItem>
          <SpecItem label="GPU vendors">{vendors}</SpecItem>
          <SpecItem label="Driver requirements">{driverRequirement}</SpecItem>
          <SpecItem label="Compute capabilities">{computeCaps}</SpecItem>
        </dl>
      </div>
    </Card>
  );
}

interface SpecItemProps {
  label: string;
  children: ReactNode;
}

function SpecItem({ label, children }: SpecItemProps) {
  return (
    <div className="flex flex-col">
      <dt className="text-xs font-medium uppercase tracking-wide text-neutral-500">
        {label}
      </dt>
      <dd className="text-sm text-neutral-900">{children}</dd>
    </div>
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