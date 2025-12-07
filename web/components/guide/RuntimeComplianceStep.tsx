import clsx from 'clsx';

import type { GuideState } from '@/lib/url/guideSearchParams';
import { Button } from '@/components/ui/Button';

interface RuntimeComplianceStepProps {
  state: GuideState;
  pythonVersions: string[];
  onChange: (next: GuideState) => void;
  onNext: () => void;
  onBack: () => void;
}

/**
 * Wizard step for runtime and compliance preferences:
 *  • Preferred Python version
 *  • License preference
 *  • Minimum acceptable security rating
 */
export function RuntimeComplianceStep({
  state,
  pythonVersions,
  onChange,
  onNext,
  onBack
}: RuntimeComplianceStepProps) {
  function setPythonVersion(version: string | null) {
    if (state.pythonVersion === version) return;

    onChange({
      ...state,
      pythonVersion: version
    });
  }

  function setLicensePreference(pref: GuideState['licensePreference']) {
    if (state.licensePreference === pref) return;

    onChange({
      ...state,
      licensePreference: pref
    });
  }

  function setMinSecurityRating(
    rating: GuideState['minSecurityRating']
  ) {
    if (state.minSecurityRating === rating) return;

    onChange({
      ...state,
      minSecurityRating: rating
    });
  }

  const licenseOptions: {
    id: GuideState['licensePreference'];
    label: string;
    description: string;
  }[] = [
    {
      id: 'any',
      label: 'Any license',
      description: 'Include both open-source and proprietary images.'
    },
    {
      id: 'oss-only',
      label: 'Open-source only',
      description: 'Only images with non-proprietary licenses.'
    }
  ];

  const securityOptions: {
    id: GuideState['minSecurityRating'];
    label: string;
    description: string;
  }[] = [
    {
      id: null,
      label: 'No minimum',
      description: 'Include images regardless of security scan rating.'
    },
    {
      id: 'B',
      label: 'B or better',
      description: 'Require at least a B security rating when available.'
    },
    {
      id: 'A',
      label: 'A only',
      description: 'Prefer the most secure images with an A rating.'
    }
  ];

  return (
    <section
      aria-label="Runtime and compliance preferences"
      className="space-y-6"
    >
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-neutral-900 sm:text-xl">
          Runtime & compliance preferences
        </h2>
        <p className="text-sm text-neutral-600">
          Tell us which Python versions, licenses, and security levels you&apos;re
          comfortable with so we can narrow down the options.
        </p>
      </div>

      <div className="space-y-6">
        {/* Python version */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Python version
          </h3>
          <p className="text-xs text-neutral-500">
            Choose a specific Python version if your code depends on it, or keep
            it flexible.
          </p>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setPythonVersion(null)}
              className={clsx(
                'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                state.pythonVersion === null
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50'
              )}
            >
              No preference
            </button>
            {pythonVersions.map((version) => {
              const isActive = state.pythonVersion === version;
              return (
                <button
                  key={version}
                  type="button"
                  onClick={() => setPythonVersion(version)}
                  className={clsx(
                    'rounded-full border px-3 py-1 text-xs font-medium transition-colors',
                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
                    isActive
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-neutral-200 bg-white text-neutral-700 hover:bg-neutral-50'
                  )}
                >
                  Python {version}
                </button>
              );
            })}
          </div>
        </div>

        {/* License preference */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            License preference
          </h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {licenseOptions.map((option) => {
              const isActive = state.licensePreference === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setLicensePreference(option.id)}
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

        {/* Security rating */}
        <div className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Security rating
          </h3>
          <p className="text-xs text-neutral-500">
            Use the security scan rating as a minimum bar for recommended images.
          </p>
          <div className="grid gap-2 md:grid-cols-3">
            {securityOptions.map((option) => {
              const isActive = state.minSecurityRating === option.id;
              return (
                <button
                  key={option.label}
                  type="button"
                  onClick={() => setMinSecurityRating(option.id)}
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