import type { ComponentType, SVGProps } from 'react';
import { Cpu, Image, MessageCircle, Server, Layers, BarChart3 } from 'lucide-react';
import clsx from 'clsx';
import type { GuideWorkload } from '@/lib/url/guideSearchParams';

interface UseCaseSelectorProps {
  value: GuideWorkload | null;
  onChange: (value: GuideWorkload | null) => void;
}

interface UseCaseOption {
  id: GuideWorkload;
  label: string;
  description: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

const OPTIONS: UseCaseOption[] = [
  {
    id: 'computer-vision',
    label: 'Computer Vision',
    description: 'Object detection, segmentation, and image models.',
    icon: Image
  },
  {
    id: 'llm-train',
    label: 'LLM / Text Generation',
    description: 'Training or fine-tuning language models.',
    icon: MessageCircle
  },
  {
    id: 'llm-inference',
    label: 'LLM Inference / Serving',
    description: 'Deploying models with vLLM, TGI, Triton, and similar stacks.',
    icon: Server
  },
  {
    id: 'multimodal',
    label: 'Multimodal',
    description: 'Vision-language models and multi-input systems.',
    icon: Layers
  },
  {
    id: 'classical-ml',
    label: 'Classical ML',
    description: 'Scikit-learn, XGBoost, and traditional ML workloads.',
    icon: BarChart3
  },
  {
    id: 'general',
    label: 'General / Experimentation',
    description: 'Flexible base images for a mix of workloads.',
    icon: Cpu
  }
];

/**
 * Card-based selector for the primary workload / use case.
 * Cards behave like a segmented control: clicking the active card clears the selection.
 */
export function UseCaseSelector({ value, onChange }: UseCaseSelectorProps) {
  function handleSelect(id: GuideWorkload) {
    if (value === id) {
      onChange(null);
    } else {
      onChange(id);
    }
  }

  return (
    <section aria-label="Primary use case" className="space-y-4">
      <div className="space-y-1">
        <h2 className="text-sm font-semibold text-neutral-900">
          What do you want to build?
        </h2>
        <p className="text-xs text-neutral-500">
          Select the primary workload you&apos;re targeting. You can refine details
          with frameworks and cloud preferences.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {OPTIONS.map((option) => {
          const Icon = option.icon;
          const selected = value === option.id;

          return (
            <button
              key={option.id}
              type="button"
              onClick={() => handleSelect(option.id)}
              className={clsx(
                'flex w-full items-start gap-3 rounded-lg border px-3 py-3 text-left transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                selected
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-neutral-200 bg-white hover:border-primary-200 hover:bg-neutral-50'
              )}
            >
              <span
                className={clsx(
                  'mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl',
                  'bg-gradient-to-br from-primary-50 to-primary-100/60',
                  'ring-1 ring-primary-200/50',
                  selected && 'from-primary-100 to-primary-200/60 ring-primary-300/50'
                )}
              >
                <Icon className="h-4 w-4 text-primary-600" aria-hidden="true" />
              </span>
              <span className="flex min-w-0 flex-col">
                <span className="text-sm font-medium text-neutral-900">
                  {option.label}
                </span>
                <span className="text-xs text-neutral-500 line-clamp-2">
                  {option.description}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}