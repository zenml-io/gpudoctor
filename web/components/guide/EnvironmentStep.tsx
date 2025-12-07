import clsx from 'clsx';

import type { GuideState } from '@/lib/url/guideSearchParams';
import type { CloudProviderAffinity } from '@/lib/types/images';
import { Checkbox } from '@/components/ui/Checkbox';
import { Button } from '@/components/ui/Button';

interface EnvironmentStepProps {
  state: GuideState;
  onChange: (next: GuideState) => void;
  onNext: () => void;
  onBack: () => void;
}

/**
 * Wizard step for environment preferences:
 *  • GPU requirement
 *  • Cloud providers
 *  • Cloud specificity (portable vs cloud-optimized)
 */
export function EnvironmentStep({
  state,
  onChange,
  onNext,
  onBack
}: EnvironmentStepProps) {
  function setGpuPreference(pref: GuideState['gpuPreference']) {
    if (state.gpuPreference === pref) return;
    onChange({
      ...state,
      gpuPreference: pref
    });
  }

  function toggleCloud(cloud: CloudProviderAffinity) {
    const exists = state.clouds.includes(cloud);
    const nextClouds = exists
      ? state.clouds.filter((c) => c !== cloud)
      : [...state.clouds, cloud];

    onChange({
      ...state,
      clouds: nextClouds
    });
  }

  function setCloudSpecificity(spec: GuideState['cloudSpecificity']) {
    if (state.cloudSpecificity === spec) return;
    onChange({
      ...state,
      cloudSpecificity: spec
    });
  }

  const cloudOptions: { id: CloudProviderAffinity; label: string }[] = [
    { id: 'aws', label: 'AWS' },
    { id: 'gcp', label: 'Google Cloud' },
    { id: 'azure', label: 'Azure' },
    { id: 'any', label: 'Any / Self-hosted' }
  ];

  const gpuOptions: {
    id: GuideState['gpuPreference'];
    label: string;
    description: string;
  }[] = [
    {
      id: 'gpu-required',
      label: 'I need GPU acceleration',
      description: 'For training, large models, or GPU-only libraries.'
    },
    {
      id: 'cpu-only',
      label: 'CPU only is fine',
      description: 'For lighter workloads, experimentation, or CI pipelines.'
    },
    {
      id: 'any',
      label: "I'm flexible",
      description: 'Show both GPU and CPU images.'
    }
  ];

  const cloudSpecificityOptions: {
    id: GuideState['cloudSpecificity'];
    label: string;
    description: string;
  }[] = [
    {
      id: 'either',
      label: 'Either works',
      description: 'Show both portable and cloud-optimized images.'
    },
    {
      id: 'portable',
      label: 'Prefer portable / generic',
      description: 'Images that work across clouds or on-prem.'
    },
    {
      id: 'cloud-optimized',
      label: 'Prefer cloud-optimized',
      description: 'Images tuned for specific cloud providers.'
    }
  ];

  return (
    <section aria-label="Environment preferences" className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-neutral-900 sm:text-xl">
          Where will you run this?
        </h2>
        <p className="text-sm text-neutral-600">
          Tell us about your hardware and cloud setup so we can recommend images
          that fit your environment.
        </p>
      </div>

      <div className="space-y-5">
        {/* GPU requirement */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            GPU requirement
          </h3>
          <div className="grid gap-2 md:grid-cols-3">
            {gpuOptions.map((option) => {
              const isActive = state.gpuPreference === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setGpuPreference(option.id)}
                  className={clsx(
                    'flex h-full flex-col items-start gap-1 rounded-lg border px-3 py-2 text-left text-xs transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                    isActive
                      ? 'border-primary-500 bg-primary-50 text-primary-800'
                      : 'border-neutral-200 bg-white text-neutral-700 hover:border-primary-200 hover:bg-neutral-50'
                  )}
                >
                  <span className="font-medium">{option.label}</span>
                  <span className="text-[11px] text-neutral-500">
                    {option.description}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Cloud providers */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Cloud providers
          </h3>
          <p className="text-xs text-neutral-500">
            Pick all the platforms you want this image to work well on.
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4">
            {cloudOptions.map((cloud) => (
              <Checkbox
                key={cloud.id}
                id={`env-cloud-${cloud.id}`}
                checked={state.clouds.includes(cloud.id)}
                onCheckedChange={() => toggleCloud(cloud.id)}
                label={cloud.label}
              />
            ))}
          </div>
        </div>

        {/* Cloud specificity */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Cloud specificity
          </h3>
          <div className="grid gap-2 md:grid-cols-3">
            {cloudSpecificityOptions.map((option) => {
              const isActive = state.cloudSpecificity === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setCloudSpecificity(option.id)}
                  className={clsx(
                    'flex h-full flex-col items-start gap-1 rounded-lg border px-3 py-2 text-left text-xs transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                    isActive
                      ? 'border-primary-500 bg-primary-50 text-primary-800'
                      : 'border-neutral-200 bg-white text-neutral-700 hover:border-primary-200 hover:bg-neutral-50'
                  )}
                >
                  <span className="font-medium">{option.label}</span>
                  <span className="text-[11px] text-neutral-500">
                    {option.description}
                  </span>
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