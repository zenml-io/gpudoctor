import type {
  ButtonHTMLAttributes,
  DetailedHTMLProps
} from 'react';
import clsx from 'clsx';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export type ButtonProps = {
  variant?: ButtonVariant;
  size?: ButtonSize;
} & DetailedHTMLProps<ButtonHTMLAttributes<HTMLButtonElement>, HTMLButtonElement>;

/**
 * Base button component used across the app.
 * Variants and sizes are aligned with the ZenML design tokens so visual updates
 * can be made in one place without hunting through feature code.
 */
export function Button({
  variant = 'primary',
  size = 'md',
  className,
  ...props
}: ButtonProps) {
  const baseClasses =
    'inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors ' +
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 ' +
    'disabled:opacity-50 disabled:cursor-not-allowed';

  const variantClasses = getVariantClasses(variant);
  const sizeClasses = getSizeClasses(size);

  return (
    <button
      className={clsx(baseClasses, variantClasses, sizeClasses, className)}
      {...props}
    />
  );
}

function getVariantClasses(variant: ButtonVariant): string {
  switch (variant) {
    case 'secondary':
      return 'bg-neutral-100 text-neutral-900 hover:bg-neutral-200 border border-neutral-200';
    case 'ghost':
      return 'bg-transparent text-neutral-700 hover:bg-neutral-100';
    case 'primary':
    default:
      return 'bg-primary-500 text-white hover:bg-primary-600 shadow-sm';
  }
}

function getSizeClasses(size: ButtonSize): string {
  switch (size) {
    case 'sm':
      return 'text-xs px-3 py-1.5';
    case 'lg':
      return 'text-sm px-5 py-2.5';
    case 'md':
    default:
      return 'text-sm px-4 py-2';
  }
}