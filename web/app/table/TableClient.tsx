'use client';

import { useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import type { ImageEntry } from '@/lib/types/images';
import type {
  TableState,
  TableSortBy,
  TableSortDir
} from '@/lib/url/tableSearchParams';
import {
  parseTableState,
  serializeTableState
} from '@/lib/url/tableSearchParams';
import { filterTableImages } from '@/lib/filters/tableFilters';
import { TableToolbar } from '@/components/table/TableToolbar';
import { FilterChips } from '@/components/table/FilterChips';
import { DataTable } from '@/components/table/DataTable';

interface TableClientProps {
  images: ImageEntry[];
}

/**
 * Client-side wrapper for the data explorer table.
 * Manages URL-backed state for filters, search, and sorting.
 */
export function TableClient({ images }: TableClientProps) {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [state, setState] = useState<TableState>(() =>
    parseTableState(searchParams)
  );

  const filteredImages = useMemo(
    () => filterTableImages(images, state),
    [images, state]
  );

  function updateState(next: TableState) {
    setState(next);
    const query = serializeTableState(next);
    const href = query ? `/table?${query}` : '/table';
    router.replace(href, { scroll: false });
  }

  function handleSort(column: TableSortBy) {
    const isSameColumn = state.sortBy === column;
    const nextDir: TableSortDir =
      isSameColumn && state.sortDir === 'asc' ? 'desc' : 'asc';

    updateState({
      ...state,
      sortBy: column,
      sortDir: nextDir
    });
  }

  const totalCount = images.length;
  const showingCount = filteredImages.length;
  const isGuideConstrained = state.ids.length > 0;

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          {isGuideConstrained ? 'Guide-matched images' : 'Browse all images'}
        </h1>
        <p className="max-w-2xl text-sm text-neutral-600">
          {isGuideConstrained
            ? `Showing ${showingCount} of ${state.ids.length} guide-matched images.`
            : `Showing ${showingCount} of ${totalCount} images in the catalog.`}
        </p>
      </section>

      <TableToolbar state={state} onStateChange={updateState} images={images} />
      <FilterChips state={state} onStateChange={updateState} />
      <DataTable
        images={filteredImages}
        sortBy={state.sortBy}
        sortDir={state.sortDir}
        onSort={handleSort}
      />
    </div>
  );
}