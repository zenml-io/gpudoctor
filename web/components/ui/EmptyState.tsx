import type { ReactNode } from 'react';

interface EmptyStateProps {
  title: string;
  description: string;
  action?: ReactNode;
}

/**
 * Simple centered empty state used when filters produce no results or
 * when a section has nothing to display yet.
 */
export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-neutral-200 bg-neutral-50 px-6 py-10 text-center">
      <h2 className="text-sm font-semibold text-neutral-900">{title}</h2>
      <p className="max-w-md text-sm text-neutral-500">{description}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}