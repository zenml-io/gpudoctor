'use client';

import * as CheckboxPrimitive from '@radix-ui/react-checkbox';
import { Check } from 'lucide-react';
import clsx from 'clsx';

interface CheckboxProps {
  id?: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label?: string;
  className?: string;
}

/**
 * Accessible checkbox built on Radix UI with a purple checkmark.
 * When a label is provided, the control and text are linked for better usability.
 */
export function Checkbox({
  id,
  checked,
  onCheckedChange,
  label,
  className
}: CheckboxProps) {
  const control = (
    <CheckboxPrimitive.Root
      id={id}
      checked={checked}
      onCheckedChange={(value) => onCheckedChange(Boolean(value))}
      className={clsx(
        'flex h-4 w-4 items-center justify-center rounded border border-neutral-300 bg-white shadow-sm',
        'data-[state=checked]:border-primary-500 data-[state=checked]:bg-primary-500',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white',
        className
      )}
    >
      <CheckboxPrimitive.Indicator className="text-white">
        <Check className="h-3 w-3" aria-hidden="true" />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  );

  if (!label) {
    return control;
  }

  return (
    <label
      htmlFor={id}
      className="inline-flex cursor-pointer items-center gap-2 text-sm text-neutral-700"
    >
      {control}
      <span>{label}</span>
    </label>
  );
}