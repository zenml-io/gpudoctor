import clsx from 'clsx';
import type { GuideState } from '@/lib/url/guideSearchParams';
import type { CloudProviderAffinity, ImageRole } from '@/lib/types/images';
import { Checkbox } from '@/components/ui/Checkbox';
import { formatFrameworkLabel } from '@/lib/format/frameworkLabels';

interface RequirementsSectionProps {
  state: GuideState;
  onChange: (next: GuideState) => void;
  /**
   * Framework options derived from the image catalog (normalized to lowercase).
   */
  availableFrameworks: string[];
}

/**
 * Requirements section for selecting frameworks, cloud provider preferences,
 * and an optional image role. Acts as a coarse filter before ranking.
 */
export function RequirementsSection({
  state,
  onChange,
  availableFrameworks
}: RequirementsSectionProps) {
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

  function setRole(role: ImageRole | null) {
    onChange({
      ...state,
      role
    });
  }

  const cloudOptions: { id: CloudProviderAffinity; label: string }[] = [
    { id: 'aws', label: 'AWS' },
    { id: 'gcp', label: 'Google Cloud' },
    { id: 'azure', label: 'Azure' },
    { id: 'any', label: 'Any / Self-hosted' }
  ];

  const roleOptions: { id: ImageRole | null; label: string }[] = [
    { id: null, label: 'Any' },
    { id: 'training', label: 'Training' },
    { id: 'serving', label: 'Inference / Serving' },
    { id: 'notebook', label: 'Notebook' },
    { id: 'base', label: 'Base image' }
  ];

  // Framework options are expected to be normalized to lowercase already,
  // but we keep the label formatting separate for user-facing text.
  const frameworkOptions = availableFrameworks;

  return (
    <section aria-label="Requirements" className="space-y-4">
      <div className="space-y-1">
        <h2 className="text-sm font-semibold text-neutral-900">
          What are your requirements?
        </h2>
        <p className="text-xs text-neutral-500">
          Select any that apply—leave everything empty if you&apos;re flexible.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Frameworks
          </h3>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3">
            {frameworkOptions.map((fw) => {
              const normalized = fw.toLowerCase();
              return (
                <Checkbox
                  key={normalized}
                  id={`fw-${normalized}`}
                  checked={state.frameworks.includes(normalized)}
                  onCheckedChange={() => toggleFramework(normalized)}
                  label={formatFrameworkLabel(normalized)}
                />
              );
            })}
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Cloud provider
          </h3>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3">
            {cloudOptions.map((cloud) => (
              <Checkbox
                key={cloud.id}
                id={`cloud-${cloud.id}`}
                checked={state.clouds.includes(cloud.id)}
                onCheckedChange={() => toggleCloud(cloud.id)}
                label={cloud.label}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
          Image role (optional)
        </h3>
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
    </section>
  );
}

