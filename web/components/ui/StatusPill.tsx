import clsx from 'clsx';
import type { MaintenanceStatus } from '@/lib/types/images';

interface StatusPillProps {
  status: MaintenanceStatus;
  className?: string;
}

/**
 * Compact pill indicator for maintenance status.
 * Examples:
 *  • Active      → green
 *  • Deprecated  → orange
 *  • EOL         → red
 */
export function StatusPill({ status, className }: StatusPillProps) {
  const label = LABELS[status];
  const styles = STYLES[status];

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        styles.container,
        className
      )}
    >
      <span
        className={clsx('h-1.5 w-1.5 rounded-full', styles.dot)}
        aria-hidden="true"
      />
      <span>{label}</span>
    </span>
  );
}

const LABELS: Record<MaintenanceStatus, string> = {
  active: 'Active',
  deprecated: 'Deprecated',
  'end-of-life': 'EOL'
};

const STYLES: Record<MaintenanceStatus, { container: string; dot: string }> = {
  active: {
    container: 'bg-success-100 text-success-600',
    dot: 'bg-success-500'
  },
  deprecated: {
    container: 'bg-warning-100 text-warning-600',
    dot: 'bg-warning-500'
  },
  'end-of-life': {
    container: 'bg-error-100 text-error-600',
    dot: 'bg-error-500'
  }
};