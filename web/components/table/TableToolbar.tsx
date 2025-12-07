import { useMemo } from 'react';

import type {
  ImageEntry,
  ImageProvider,
  MaintenanceStatus,
  Workload
} from '@/lib/types/images';
import type { TableState } from '@/lib/url/tableSearchParams';
import {
  getAllFrameworkNames,
  getAllProviders,
  getAllCudaVersions,
  getAllWorkloads
} from '@/lib/data/images.client';
import { SearchInput } from '@/components/ui/SearchInput';
import { FilterDropdown, type FilterOption } from '@/components/table/FilterDropdown';

interface TableToolbarProps {
  state: TableState;
  onStateChange: (next: TableState) => void;
  images: ImageEntry[];
}

/**
 * Top-of-page filter toolbar for the table view.
 * Includes the main search input and a grid of dropdown filters.
 */
export function TableToolbar({
  state,
  onStateChange,
  images
}: TableToolbarProps) {
  const frameworkOptions = useMemo<FilterOption[]>(
    () =>
      getAllFrameworkNames(images).map((fw) => ({
        value: fw,
        label: formatFrameworkLabel(fw)
      })),
    [images]
  );

  const providerOptions = useMemo<FilterOption[]>(
    () =>
      getAllProviders(images).map((provider) => ({
        value: provider,
        label: formatProviderLabel(provider)
      })),
    [images]
  );

  const workloadOptions = useMemo<FilterOption[]>(
    () =>
      getAllWorkloads(images).map((wk) => ({
        value: wk,
        label: formatWorkloadLabel(wk)
      })),
    [images]
  );

  const statusOptions: FilterOption[] = [
    { value: 'active', label: 'Active' },
    { value: 'deprecated', label: 'Deprecated' },
    { value: 'end-of-life', label: 'EOL' }
  ];

  const cudaOptions = useMemo<FilterOption[]>(
    () =>
      getAllCudaVersions(images).map((version) => ({
        value: version,
        label: `CUDA ${version}`
      })),
    [images]
  );

  function handleQueryChange(query: string) {
    onStateChange({
      ...state,
      query
    });
  }

  return (
    <div className="space-y-4 rounded-lg border border-neutral-200 bg-white p-4 shadow-card">
      <SearchInput
        value={state.query}
        onChange={handleQueryChange}
        placeholder="Search images, frameworks, versions..."
      />
      <div className="grid gap-3 sm:grid-cols-3">
        <FilterDropdown
          label="Framework"
          options={frameworkOptions}
          selectedValues={state.frameworks}
          onChange={(next) =>
            onStateChange({
              ...state,
              frameworks: next
            })
          }
        />
        <FilterDropdown
          label="Provider"
          options={providerOptions}
          selectedValues={state.providers}
          onChange={(next) =>
            onStateChange({
              ...state,
              providers: next as ImageProvider[]
            })
          }
        />
        <FilterDropdown
          label="Workload"
          options={workloadOptions}
          selectedValues={state.workloads}
          onChange={(next) =>
            onStateChange({
              ...state,
              workloads: next as Workload[]
            })
          }
        />
        <FilterDropdown
          label="Status"
          options={statusOptions}
          selectedValues={state.status}
          onChange={(next) =>
            onStateChange({
              ...state,
              status: next as MaintenanceStatus[]
            })
          }
        />
        <FilterDropdown
          label="CUDA"
          options={cudaOptions}
          selectedValues={state.cudaVersions}
          onChange={(next) =>
            onStateChange({
              ...state,
              cudaVersions: next
            })
          }
        />
      </div>
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
    'triton-inference-server': 'Triton Inference Server'
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
      return 'Reinforcement learning';
    case 'scientific-computing':
      return 'Scientific computing';
    case 'generic':
      return 'General';
    default:
      return workload;
  }
}