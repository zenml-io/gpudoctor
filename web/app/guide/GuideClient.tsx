'use client';

import { useMemo, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

import type { ImageEntry } from '@/lib/types/images';
import {
  parseGuideState,
  serializeGuideState,
  type GuideState
} from '@/lib/url/guideSearchParams';
import { filterImages } from '@/lib/filters/guideFilters';
import { getAllFrameworkNames, getAllPythonVersions } from '@/lib/data/images.client';
import { Card } from '@/components/ui/Card';
import { GuideStepper } from '@/components/guide/GuideStepper';
import { UseCaseStep } from '@/components/guide/UseCaseStep';
import { EnvironmentStep } from '@/components/guide/EnvironmentStep';
import { FrameworkRoleStep } from '@/components/guide/FrameworkRoleStep';
import { RuntimeComplianceStep } from '@/components/guide/RuntimeComplianceStep';
import { PrioritiesStep } from '@/components/guide/PrioritiesStep';
import { GuideResults } from '@/components/guide/GuideResults';

interface GuideClientProps {
  images: ImageEntry[];
}

export function GuideClient({ images }: GuideClientProps) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [state, setState] = useState<GuideState>(() =>
    parseGuideState(searchParams)
  );

  const [currentStep, setCurrentStep] = useState<number>(1);
  const totalSteps = 5;

  const availableFrameworks = useMemo(
    () => getAllFrameworkNames(images),
    [images]
  );

  const pythonVersions = useMemo(
    () => getAllPythonVersions(images),
    [images]
  );

  const matchingImages = useMemo(
    () => filterImages(images, state),
    [images, state]
  );

  function updateState(next: GuideState) {
    setState(next);
    const query = serializeGuideState(next);
    const href = query ? `/guide?${query}` : '/guide';
    router.replace(href, { scroll: false });
  }

  function goToStep(step: number) {
    setCurrentStep(Math.min(Math.max(step, 1), totalSteps));
  }

  function goNext() {
    goToStep(currentStep + 1);
  }

  function goBack() {
    goToStep(currentStep - 1);
  }

  function scrollToResults() {
    const el = document.getElementById('guide-results');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          Find your perfect GPU image
        </h1>
        <p className="max-w-2xl text-sm text-neutral-600">
          Answer a few quick questions and we&apos;ll recommend Docker images
          that match your requirements.
        </p>
      </section>

      <Card className="space-y-6">
        <GuideStepper currentStep={currentStep} totalSteps={totalSteps} onStepClick={goToStep} />

        {currentStep === 1 && (
          <UseCaseStep
            workload={state.workload}
            onChange={(workload) => updateState({ ...state, workload })}
            onNext={goNext}
          />
        )}

        {currentStep === 2 && (
          <EnvironmentStep
            state={state}
            onChange={updateState}
            onNext={goNext}
            onBack={goBack}
          />
        )}

        {currentStep === 3 && (
          <FrameworkRoleStep
            state={state}
            availableFrameworks={availableFrameworks}
            onChange={updateState}
            onNext={goNext}
            onBack={goBack}
          />
        )}

        {currentStep === 4 && (
          <RuntimeComplianceStep
            state={state}
            pythonVersions={pythonVersions}
            onChange={updateState}
            onNext={goNext}
            onBack={goBack}
          />
        )}

        {currentStep === 5 && (
          <PrioritiesStep
            state={state}
            onChange={updateState}
            onBack={goBack}
            onSubmit={scrollToResults}
          />
        )}
      </Card>

      <GuideResults images={matchingImages} state={state} />
    </div>
  );
}