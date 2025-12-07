import type { GuideWorkload } from '@/lib/url/guideSearchParams';
import { UseCaseSelector } from '@/components/guide/UseCaseSelector';
import { Button } from '@/components/ui/Button';

interface UseCaseStepProps {
  workload: GuideWorkload | null;
  onChange: (workload: GuideWorkload | null) => void;
  onNext: () => void;
}

/**
 * Wizard step for selecting the primary ML task.
 * Wraps the card-based UseCaseSelector and provides navigation controls.
 */
export function UseCaseStep({ workload, onChange, onNext }: UseCaseStepProps) {
  const canProceed = workload !== null;

  return (
    <section aria-label="ML task selection" className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-neutral-900 sm:text-xl">
          What are you building?
        </h2>
        <p className="text-sm text-neutral-600">
          Choose the main type of workload you&apos;re targeting. You can refine
          frameworks and environment details in the next steps.
        </p>
      </div>

      <UseCaseSelector value={workload} onChange={onChange} />

      <div className="flex justify-end pt-2">
        <Button
          type="button"
          size="lg"
          onClick={onNext}
          disabled={!canProceed}
          className="min-w-[96px]"
        >
          Next
        </Button>
      </div>
    </section>
  );
}