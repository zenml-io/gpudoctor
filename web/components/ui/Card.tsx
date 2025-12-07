import type { ReactNode } from 'react';
import clsx from 'clsx';

export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

interface CardProps {
  children: ReactNode;
  padding?: CardPadding;
  className?: string;
}

/**
 * Generic card surface used across the app for content grouping.
 * Uses the gpudoctor-card utility class defined in globals for consistent
 * radius, border, and shadow, with optional Tailwind padding presets.
 */
export function Card({ children, padding = 'md', className }: CardProps) {
  const paddingClasses = getPaddingClasses(padding);

  return (
    <div className={clsx('gpudoctor-card bg-white', paddingClasses, className)}>
      {children}
    </div>
  );
}

function getPaddingClasses(padding: CardPadding): string {
  switch (padding) {
    case 'none':
      return 'p-0';
    case 'sm':
      return 'p-3';
    case 'lg':
      return 'p-6';
    case 'md':
    default:
      return 'p-4';
  }
}