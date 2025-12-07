import { ChevronUp, ChevronDown } from 'lucide-react';
import clsx from 'clsx';

import type {
  GuideState,
  GuidePriorityKey
} from '@/lib/url/guideSearchParams';
import { Button } from '@/components/ui/Button';

interface PrioritiesStepProps {
  state: GuideState;
  onChange: (next: GuideState) => void;
  onBack: () => void;
  /** Called when the user clicks "Find Images" to scroll to results. */
  onSubmit?: () => void;
}

const PRIORITY_LABELS: Record<GuidePriorityKey, string> = {
  security: 'Security & vulnerability posture',
  size: 'Smaller image size',
  license: 'Open licensing',
  gpu: 'GPU support & performance',
  cloud: 'Cloud alignment',
  freshness: 'Most recently updated'
};

/**
 * Wizard step for ranking how results should be scored.
 * Users can reorder priority keys so the guide can weigh trade-offs
 * between security, size, licensing, GPU capabilities, cloud alignment,
 * and freshness.
 */
export function PrioritiesStep({
  state,
  onChange,
  onBack,
  onSubmit
}: PrioritiesStepProps) {
  function movePriority(index: number, direction: 'up' | 'down') {
    const priorities = state.priorities;
    const lastIndex = priorities.length - 1;

    if (
      (direction === 'up' && index === 0) ||
      (direction === 'down' && index === lastIndex)
    ) {
      return;
    }

    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex > lastIndex) {
      return;
    }

    const next = [...priorities];
    const [moved] = next.splice(index, 1);
    next.splice(targetIndex, 0, moved);

    onChange({
      ...state,
      priorities: next
    });
  }

  return (
    <section
      aria-label="Result ranking priorities"
      className="space-y-6"
    >
      <div className="space-y-1">
        <h2 className="text-lg font-semibold text-neutral-900 sm:text-xl">
          What matters most in your results?
        </h2>
        <p className="text-sm text-neutral-600">
          Drag the most important factors to the top using the arrows. We&apos;ll
          use this order to break ties and rank images that all meet your
          requirements.
        </p>
      </div>

      <ol className="space-y-2">
        {state.priorities.map((key, index) => {
          const isFirst = index === 0;
          const isLast = index === state.priorities.length - 1;
          const label = PRIORITY_LABELS[key];

          return (
            <li
              key={key}
              className={clsx(
                'flex items-center justify-between gap-3 rounded-lg border bg-white px-3 py-2 text-xs sm:text-sm',
                isFirst
                  ? 'border-primary-200 bg-primary-50'
                  : 'border-neutral-200'
              )}
            >
              <div className="flex items-center gap-3">
                <span
                  className={clsx(
                    'flex h-6 w-6 items-center justify-center rounded-full text-[11px] font-semibold',
                    isFirst
                      ? 'bg-primary-500 text-white'
                      : 'bg-neutral-100 text-neutral-700'
                  )}
                >
                  {index + 1}
                </span>
                <div className="flex flex-col">
                  <span className="text-xs font-medium text-neutral-900 sm:text-sm">
                    {label}
                  </span>
                  {isFirst && (
                    <span className="text-[11px] text-primary-700">
                      Most important
                    </span>
                  )}
                  {!isFirst && (
                    <span className="text-[11px] text-neutral-500">
                      Lower priority than items above
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => movePriority(index, 'up')}
                  disabled={isFirst}
                  aria-label="Move up"
                  className={clsx(
                    'inline-flex h-7 w-7 items-center justify-center rounded border text-neutral-500 transition-colors',
                    'hover:bg-neutral-100 hover:text-neutral-800',
                    'disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent',
                    isFirst ? 'border-neutral-200' : 'border-neutral-300'
                  )}
                >
                  <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
                </button>
                <button
                  type="button"
                  onClick={() => movePriority(index, 'down')}
                  disabled={isLast}
                  aria-label="Move down"
                  className={clsx(
                    'inline-flex h-7 w-7 items-center justify-center rounded border text-neutral-500 transition-colors',
                    'hover:bg-neutral-100 hover:text-neutral-800',
                    'disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent',
                    isLast ? 'border-neutral-200' : 'border-neutral-300'
                  )}
                >
                  <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
                </button>
              </div>
            </li>
          );
        })}
      </ol>

      <div className="mt-4 flex items-center justify-between gap-3 border-t border-neutral-100 pt-4">
        <Button
          type="button"
          variant="secondary"
          size="md"
          onClick={onBack}
        >
          Back
        </Button>
        <Button
          type="button"
          size="md"
          className="min-w-[120px]"
          onClick={onSubmit}
        >
          Find Images
        </Button>
      </div>
    </section>
  );
}