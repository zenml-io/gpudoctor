import type { ReactNode } from 'react';
import clsx from 'clsx';

export type BadgeVariant =
  | 'default'
  | 'success'
  | 'warning'
  | 'error'
  | 'info'
  | 'purple';

export type BadgeSize = 'sm' | 'md';

interface BadgeProps {
  children: ReactNode;
  variant?: BadgeVariant;
  size?: BadgeSize;
  className?: string;
}

/**
 * Generic pill-shaped badge used for small status and tag labels
 * (e.g. "Official", "PyTorch", "CUDA 12.4").
 */
export function Badge({
  children,
  variant = 'default',
  size = 'md',
  className
}: BadgeProps) {
  const baseClasses =
    'inline-flex items-center rounded-full border text-xs font-medium';

  const sizeClasses =
    size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-0.5';

  const variantClasses = getVariantClasses(variant);

  return (
    <span className={clsx(baseClasses, sizeClasses, variantClasses, className)}>
      {children}
    </span>
  );
}

function getVariantClasses(variant: BadgeVariant): string {
  switch (variant) {
    case 'success':
      return 'bg-success-100 text-success-600 border-success-100';
    case 'warning':
      return 'bg-warning-100 text-warning-600 border-warning-100';
    case 'error':
      return 'bg-error-100 text-error-600 border-error-100';
    case 'info':
      return 'bg-info-100 text-info-600 border-info-100';
    case 'purple':
      return 'bg-primary-50 text-primary-700 border-primary-100';
    case 'default':
    default:
      return 'bg-neutral-100 text-neutral-700 border-neutral-200';
  }
}