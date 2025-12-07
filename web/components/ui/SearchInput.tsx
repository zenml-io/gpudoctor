'use client';

import type { ChangeEvent } from 'react';
import { Search } from 'lucide-react';
import clsx from 'clsx';

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

/**
 * Text input with a search icon, used for free-text filtering in the table
 * and other search surfaces.
 */
export function SearchInput({
  value,
  onChange,
  placeholder,
  className
}: SearchInputProps) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    onChange(event.target.value);
  }

  return (
    <div className={clsx('relative', className)}>
      <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-neutral-400">
        <Search className="h-4 w-4" aria-hidden="true" />
      </span>
      <input
        type="text"
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        className="block w-full rounded-md border border-neutral-200 bg-white py-2 pl-10 pr-3 text-sm text-neutral-900 placeholder-neutral-400 shadow-sm focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-400"
      />
    </div>
  );
}