import type { ReactNode } from 'react';

import type { ImageEntry, Workload, ImageRole } from '@/lib/types/images';
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

  const license = image.metadata.license ?? 'Not specified';
  const roleLabel = formatRoleLabel(image.capabilities.role);
  const workloadsLabel =
    image.capabilities.workloads.length > 0
      ? image.capabilities.workloads.map(formatWorkloadLabel).join(', ')
      : 'Not specified';
  const imageTypeLabel = formatImageType(image.capabilities.image_type);
  const awsAmi = image.cloud?.aws_ami;
  const gcpImage = image.cloud?.gcp_image;
  const azureImage = image.cloud?.azure_image;

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
          <SpecItem label="Image role">{roleLabel}</SpecItem>
          <SpecItem label="Image type">{imageTypeLabel}</SpecItem>
          <SpecItem label="Workloads">{workloadsLabel}</SpecItem>
          <SpecItem label="License">{license}</SpecItem>
        </dl>
      </div>

      {(awsAmi || gcpImage || azureImage) && (
        <div className="border-t border-neutral-100 pt-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Cloud-specific images
          </h3>
          <dl className="space-y-1 text-sm">
            {awsAmi && (
              <div className="flex gap-2">
                <dt className="text-neutral-500">AWS AMI:</dt>
                <dd className="font-mono text-xs text-neutral-700">{awsAmi}</dd>
              </div>
            )}
            {gcpImage && (
              <div className="flex gap-2">
                <dt className="text-neutral-500">GCP image:</dt>
                <dd className="font-mono text-xs text-neutral-700">{gcpImage}</dd>
              </div>
            )}
            {azureImage && (
              <div className="flex gap-2">
                <dt className="text-neutral-500">Azure image:</dt>
                <dd className="font-mono text-xs text-neutral-700">{azureImage}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
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

function formatRoleLabel(role: ImageRole): string {
  switch (role) {
    case 'base':
      return 'Base';
    case 'training':
      return 'Training';
    case 'inference':
      return 'Inference';
    case 'notebook':
      return 'Notebook';
    case 'serving':
      return 'Serving';
  }
}

function formatWorkloadLabel(workload: Workload): string {
  switch (workload) {
    case 'classical-ml':
      return 'Classical ML';
    case 'llm':
      return 'LLM';
    case 'multimodal':
      return 'Multimodal';
    case 'computer-vision':
      return 'Computer Vision';
    case 'nlp':
      return 'NLP';
    case 'audio':
      return 'Audio';
    case 'reinforcement-learning':
      return 'Reinforcement Learning';
    case 'scientific-computing':
      return 'Scientific Computing';
    case 'generic':
      return 'Generic';
  }
}

function formatImageType(imageType: string): string {
  switch (imageType) {
    case 'devel':
      return 'Development (includes compilers)';
    case 'runtime':
      return 'Runtime';
    case 'base':
      return 'Base';
    default:
      return imageType;
  }
}