import { Check } from 'lucide-react';
import clsx from 'clsx';

interface GuideStepperProps {
  currentStep: number;
  totalSteps: number;
  /** Called when a user clicks a completed step to navigate backward. */
  onStepClick?: (step: number) => void;
}

const STEP_LABELS: string[] = [
  '1. ML Task',
  '2. Environment',
  '3. Framework',
  '4. Runtime',
  '5. Priorities'
];

/**
 * Horizontal progress indicator for the guide wizard.
 * Highlights the current step, marks completed steps with a checkmark,
 * and mutes future steps.
 */
export function GuideStepper({ currentStep, totalSteps, onStepClick }: GuideStepperProps) {
  const maxSteps = STEP_LABELS.length;
  const clampedTotal = Math.max(1, Math.min(totalSteps, maxSteps));
  const clampedCurrent = Math.max(1, Math.min(currentStep, clampedTotal));
  const steps = STEP_LABELS.slice(0, clampedTotal);

  return (
    <nav aria-label="Guide progress">
      <ol className="flex items-center justify-between gap-4">
        {steps.map((label, index) => {
          const stepNumber = index + 1;
          const isComplete = stepNumber < clampedCurrent;
          const isCurrent = stepNumber === clampedCurrent;
          const isLast = index === steps.length - 1;
          const isClickable = isComplete && onStepClick;

          const stepContent = (
            <>
              <span
                className={clsx(
                  'flex h-7 w-7 items-center justify-center rounded-full border text-[11px] font-semibold sm:h-8 sm:w-8 sm:text-xs',
                  isComplete &&
                    'border-primary-500 bg-primary-500 text-white shadow-sm',
                  isCurrent &&
                    !isComplete &&
                    'border-primary-500 bg-primary-50 text-primary-700',
                  !isComplete &&
                    !isCurrent &&
                    'border-neutral-200 bg-neutral-50 text-neutral-400'
                )}
                aria-current={isCurrent ? 'step' : undefined}
              >
                {isComplete ? (
                  <Check className="h-3.5 w-3.5" aria-hidden="true" />
                ) : (
                  stepNumber
                )}
              </span>
              <span
                className={clsx(
                  'whitespace-nowrap text-[11px] sm:text-xs',
                  isCurrent
                    ? 'font-medium text-neutral-900'
                    : isComplete
                      ? 'text-neutral-700'
                      : 'text-neutral-400'
                )}
              >
                {label}
              </span>
            </>
          );

          return (
            <li
              key={label}
              className="flex flex-1 items-center gap-2 text-xs sm:text-sm"
            >
              {isClickable ? (
                <button
                  type="button"
                  onClick={() => onStepClick(stepNumber)}
                  className="flex items-center gap-2 rounded-md transition-opacity hover:opacity-80 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2"
                  aria-label={`Go back to step ${stepNumber}`}
                >
                  {stepContent}
                </button>
              ) : (
                <div className="flex items-center gap-2">
                  {stepContent}
                </div>
              )}
              {!isLast && (
                <span className="ml-2 hidden flex-1 border-t border-dashed border-neutral-200 sm:block" />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}