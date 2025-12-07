import clsx from 'clsx';

import type { GuideState } from '@/lib/url/guideSearchParams';
import type { ImageRole } from '@/lib/types/images';
import { Checkbox } from '@/components/ui/Checkbox';
import { Button } from '@/components/ui/Button';

interface FrameworkRoleStepProps {
  state: GuideState;
  availableFrameworks: string[];
  onChange: (next: GuideState) => void;
  onNext: () => void;
  onBack: () => void;
}

/**
 * Wizard step for selecting preferred ML frameworks and the primary image role.
 * Frameworks act as coarse filters while the role biases images towards training,
 * serving, notebooks, or base images.
 */
export function FrameworkRoleStep({
  state,
  availableFrameworks,
  onChange,
  onNext,
  onBack
}: FrameworkRoleStepProps) {
  function toggleFramework(rawFramework: string) {
    const framework = rawFramework.toLowerCase();
    const exists = state.frameworks.includes(framework);
    const nextFrameworks = exists
      ? state.frameworks.filter((fw) => fw !== framework)
      : [...state.frameworks, framework];

    onChange({
      ...state,
      frameworks: nextFrameworks
    });
  }

  function setRole(role: ImageRole | null) {
    if (state.role === role) return;

    onChange({
      ...state,
      role
    });
  }

  const frameworkOptions = availableFrameworks;

  const roleOptions: { id: ImageRole | null; label: string }[] = [
    { id: null, label: 'Any' },
    { id: 'training', label: 'Training' },
    { id: 'inference', label: 'Inference' },
    { id: 'serving', label: 'Serving' },
    { id: 'notebook', label: 'Notebook' },
    { id: 'base', label: 'Base image' }
  ];

  return (
    <section
      aria-label="Framework and image role preferences"
      className="space-y-6"
    >
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-neutral-900 sm:text-xl">
          Which frameworks and image type do you prefer?
        </h2>
        <p className="text-sm text-neutral-600">
          Pick the ML frameworks you plan to use and how you&apos;ll use the image.
          You can leave everything flexible if you&apos;re not sure yet.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Frameworks */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Frameworks
          </h3>
          <p className="text-xs text-neutral-500">
            Select any frameworks you&apos;d like the image to include.
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3">
            {frameworkOptions.map((fw) => {
              const normalized = fw.toLowerCase();
              return (
                <Checkbox
                  key={normalized}
                  id={`fw-step-${normalized}`}
                  checked={state.frameworks.includes(normalized)}
                  onCheckedChange={() => toggleFramework(normalized)}
                  label={formatFrameworkLabel(normalized)}
                />
              );
            })}
          </div>
        </div>

        {/* Image role */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Image role
          </h3>
          <p className="text-xs text-neutral-500">
            Choose the image&apos;s main purpose, or leave it as "Any" if you&apos;re
            open to multiple roles.
          </p>
          <div className="flex flex-wrap gap-2">
            {roleOptions.map((roleOption) => {
              const isActive =
                roleOption.id === null
                  ? state.role === null
                  : state.role === roleOption.id;

              return (
                <button
                  key={roleOption.label}
                  type="button"
                  onClick={() => setRole(roleOption.id)}
                  className={clsx(
                    'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                    isActive
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50'
                  )}
                >
                  {roleOption.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-neutral-100 pt-4">
        <Button
          type="button"
          variant="secondary"
          size="md"
          onClick={onBack}
        >
          Back
        </Button>
        <Button type="button" size="md" onClick={onNext}>
          Next
        </Button>
      </div>
    </section>
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
    'llama.cpp': 'llama.cpp',
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