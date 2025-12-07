import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

import { PageShell } from '@/components/layout/PageShell';
import { getAllImages, getImageById } from '@/lib/data/images';
import type { ImageProvider } from '@/lib/types/images';
import { ImageDetailClient } from '@/components/images/ImageDetailClient';

export const dynamic = 'error';

interface ImagePageParams {
  params: { id: string };
}

export function generateStaticParams() {
  const images = getAllImages();
  return images.map((image) => ({ id: image.id }));
}

export function generateMetadata({ params }: ImagePageParams): Metadata {
  const image = getImageById(params.id);

  if (!image) {
    return {
      title: 'Image not found | GPU Doctor',
      description:
        'The requested image could not be found in the GPU Doctor image catalog.'
    };
  }

  const primaryFramework = image.frameworks[0];
  const frameworkLabel = primaryFramework
    ? `${formatFrameworkLabel(primaryFramework.name)} ${primaryFramework.version}`
    : 'ML';
  const cudaLabel = image.cuda?.version ? `CUDA ${image.cuda.version}` : 'CPU-only';
  const providerLabel = formatProviderLabel(image.metadata.provider);

  const title = `${frameworkLabel} Docker image | GPU Doctor`;
  const description = `Details for ${image.name} – ${cudaLabel} image maintained by ${providerLabel}. View specs, compatibility, and security overview.`;

  return {
    title,
    description
  };
}

export default function ImagePage({ params }: ImagePageParams) {
  const image = getImageById(params.id);
  if (!image) {
    return notFound();
  }

  const allImages = getAllImages();

  return (
    <PageShell activeTab={null}>
      <ImageDetailClient image={image} allImages={allImages} />
    </PageShell>
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