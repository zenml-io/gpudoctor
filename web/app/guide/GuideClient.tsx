'use client';

import { useMemo, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

import type { ImageEntry } from '@/lib/types/images';
import {
  parseGuideState,
  serializeGuideState,
  type GuideState,
  type GuideWorkload
} from '@/lib/url/guideSearchParams';
import { filterImages } from '@/lib/filters/guideFilters';
import { getAllFrameworkNames } from '@/lib/data/images.client';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { UseCaseSelector } from '@/components/guide/UseCaseSelector';
import { RequirementsSection } from '@/components/guide/RequirementsSection';
import { GuideResults } from '@/components/guide/GuideResults';

interface GuideClientProps {
  images: ImageEntry[];
}

/**
 * Client-side wrapper for the interactive guide experience.
 * Manages URL-backed state for workload and filter selections.
 */
export function GuideClient({ images }: GuideClientProps) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [state, setState] = useState<GuideState>(() =>
    parseGuideState(searchParams)
  );

  const availableFrameworks = useMemo(
    () => getAllFrameworkNames(images),
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

  function handleWorkloadChange(workload: GuideWorkload | null) {
    updateState({
      ...state,
      workload
    });
  }

  function handleRequirementsChange(next: GuideState) {
    updateState(next);
  }

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          Find your perfect GPU image
        </h1>
        <p className="max-w-2xl text-sm text-neutral-600">
          Answer a few quick questions and we&apos;ll recommend Docker images
          that match your frameworks, cloud environment, and workload.
        </p>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1.8fr)]">
        <Card className="h-full">
          <UseCaseSelector
            value={state.workload}
            onChange={handleWorkloadChange}
          />
        </Card>
        <Card className="flex h-full flex-col justify-between gap-4">
          <RequirementsSection
            state={state}
            onChange={handleRequirementsChange}
            availableFrameworks={availableFrameworks}
          />
          <div className="mt-4 flex justify-end">
            <Button
              type="button"
              size="lg"
              className="inline-flex items-center gap-2"
            >
              <span>Show matching images ({matchingImages.length})</span>
            </Button>
          </div>
        </Card>
      </div>

      <GuideResults images={matchingImages} state={state} />
    </div>
  );
}