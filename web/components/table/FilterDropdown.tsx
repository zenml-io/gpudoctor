'use client';

import * as Popover from '@radix-ui/react-popover';
import { ChevronDown } from 'lucide-react';
import clsx from 'clsx';

import { Checkbox } from '@/components/ui/Checkbox';

export interface FilterOption {
  value: string;
  label: string;
}

interface FilterDropdownProps {
  label: string;
  options: FilterOption[];
  selectedValues: string[];
  onChange: (next: string[]) => void;
  className?: string;
}

/**
 * Generic multi-select dropdown built on Radix Popover.
 * Used by the table toolbar to select frameworks, providers, workloads, etc.
 */
export function FilterDropdown({
  label,
  options,
  selectedValues,
  onChange,
  className
}: FilterDropdownProps) {
  const count = selectedValues.length;

  function toggleValue(value: string) {
    const exists = selectedValues.includes(value);
    const next = exists
      ? selectedValues.filter((v) => v !== value)
      : [...selectedValues, value];
    onChange(next);
  }

  function handleClear() {
    if (selectedValues.length > 0) {
      onChange([]);
    }
  }

  function handleSelectAll() {
    if (selectedValues.length !== options.length) {
      onChange(options.map((opt) => opt.value));
    }
  }

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          className={clsx(
            'inline-flex w-full items-center justify-between gap-2 rounded-md border border-neutral-200 bg-white px-3 py-1.5 text-xs font-medium text-neutral-700 shadow-sm',
            'hover:bg-neutral-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
            className
          )}
        >
          <span>{label}</span>
          <span className="flex items-center gap-1 text-[11px] text-neutral-500">
            {count > 0 && (
              <span className="inline-flex min-w-[1.5rem] items-center justify-center rounded-full bg-neutral-100 px-1 py-0.5 text-[10px] font-semibold text-neutral-700">
                {count}
              </span>
            )}
            <ChevronDown className="h-3 w-3" aria-hidden="true" />
          </span>
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          sideOffset={6}
          align="start"
          className="z-50 w-56 rounded-lg border border-neutral-200 bg-white p-2 shadow-card"
        >
          <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
            {options.map((option) => (
              <div key={option.value} className="flex items-center gap-2 py-1">
                <Checkbox
                  id={`${label}-${option.value}`}
                  checked={selectedValues.includes(option.value)}
                  onCheckedChange={() => toggleValue(option.value)}
                />
                <label
                  htmlFor={`${label}-${option.value}`}
                  className="cursor-pointer text-xs text-neutral-800"
                >
                  {option.label}
                </label>
              </div>
            ))}
            {options.length === 0 && (
              <p className="px-1 py-1 text-xs text-neutral-500">
                No options available.
              </p>
            )}
          </div>
          {options.length > 0 && (
            <div className="mt-2 flex items-center justify-between gap-2 border-t border-neutral-100 pt-2">
              <button
                type="button"
                onClick={handleClear}
                className="text-[11px] font-medium text-neutral-500 hover:text-neutral-800"
              >
                Clear
              </button>
              <button
                type="button"
                onClick={handleSelectAll}
                className="text-[11px] font-medium text-primary-600 hover:text-primary-700"
              >
                Select all
              </button>
            </div>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}