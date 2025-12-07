import { X } from 'lucide-react';

import type {
  ImageProvider,
  MaintenanceStatus,
  Workload
} from '@/lib/types/images';
import type { TableState } from '@/lib/url/tableSearchParams';

interface FilterChipsProps {
  state: TableState;
  onStateChange: (next: TableState) => void;
}

/**
 * Renders active filters as removable chips plus a "Clear all" action.
 * Helps users understand which filters are currently applied.
 */
export function FilterChips({ state, onStateChange }: FilterChipsProps) {
  const chips = buildChips(state, onStateChange);

  if (chips.length === 0) {
    return null;
  }

  function handleClearAll() {
    onStateChange({
      ...state,
      frameworks: [],
      providers: [],
      workloads: [],
      status: [],
      cudaVersions: [],
      ids: []
    });
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-neutral-500">Active filters:</span>
      {chips.map((chip) => {
        const isFromGuide = chip.key === 'ids:guide';

        const baseClasses =
          'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium';
        const guideClasses =
          'border-primary-200 bg-primary-50 text-primary-700 hover:bg-primary-100';
        const defaultClasses =
          'border-neutral-200 bg-neutral-100 text-neutral-700 hover:bg-neutral-200';

        return (
          <button
            key={chip.key}
            type="button"
            onClick={chip.onRemove}
            className={`${baseClasses} ${
              isFromGuide ? guideClasses : defaultClasses
            }`}
          >
            <span>{chip.label}</span>
            <X className="h-3 w-3" aria-hidden="true" />
          </button>
        );
      })}
      <button
        type="button"
        onClick={handleClearAll}
        className="ml-1 text-xs font-medium text-primary-600 hover:text-primary-700"
      >
        Clear all
      </button>
    </div>
  );
}

interface Chip {
  key: string;
  label: string;
  onRemove: () => void;
}

function buildChips(
  state: TableState,
  onStateChange: (next: TableState) => void
): Chip[] {
  const chips: Chip[] = [];

  if (state.ids.length > 0) {
    chips.push({
      key: 'ids:guide',
      label:
        state.ids.length > 1
          ? `From guide (${state.ids.length})`
          : 'From guide',
      onRemove: () =>
        onStateChange({
          ...state,
          ids: []
        })
    });
  }

  for (const fw of state.frameworks) {
    chips.push({
      key: `fw:${fw}`,
      label: formatFrameworkLabel(fw),
      onRemove: () =>
        onStateChange({
          ...state,
          frameworks: state.frameworks.filter((value) => value !== fw)
        })
    });
  }

  for (const provider of state.providers) {
    chips.push({
      key: `prov:${provider}`,
      label: formatProviderLabel(provider),
      onRemove: () =>
        onStateChange({
          ...state,
          providers: state.providers.filter((value) => value !== provider)
        })
    });
  }

  for (const wk of state.workloads) {
    chips.push({
      key: `wk:${wk}`,
      label: formatWorkloadLabel(wk),
      onRemove: () =>
        onStateChange({
          ...state,
          workloads: state.workloads.filter((value) => value !== wk)
        })
    });
  }

  for (const st of state.status) {
    chips.push({
      key: `st:${st}`,
      label: formatStatusLabel(st),
      onRemove: () =>
        onStateChange({
          ...state,
          status: state.status.filter((value) => value !== st)
        })
    });
  }

  for (const cu of state.cudaVersions) {
    chips.push({
      key: `cu:${cu}`,
      label: `CUDA ${cu}`,
      onRemove: () =>
        onStateChange({
          ...state,
          cudaVersions: state.cudaVersions.filter((value) => value !== cu)
        })
    });
  }

  return chips;
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

function formatStatusLabel(status: MaintenanceStatus): string {
  switch (status) {
    case 'active':
      return 'Active';
    case 'deprecated':
      return 'Deprecated';
    case 'end-of-life':
      return 'EOL';
    default:
      return status;
  }
}