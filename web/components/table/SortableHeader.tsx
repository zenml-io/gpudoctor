import type { TableSortBy, TableSortDir } from '@/lib/url/tableSearchParams';

type SortAlignment = 'left' | 'center' | 'right';

interface SortableHeaderProps {
  label: string;
  sortKey: TableSortBy;
  currentSortBy: TableSortBy;
  currentSortDir: TableSortDir;
  onSort: (column: TableSortBy) => void;
  align?: SortAlignment;
}

/**
 * Table header cell that can toggle sorting for a given column.
 * Uses aria-sort for accessibility and Unicode arrows for sort direction.
 */
export function SortableHeader({
  label,
  sortKey,
  currentSortBy,
  currentSortDir,
  onSort,
  align = 'left'
}: SortableHeaderProps) {
  const isActive = currentSortBy === sortKey;
  const ariaSort: 'ascending' | 'descending' | 'none' = isActive
    ? currentSortDir === 'asc'
      ? 'ascending'
      : 'descending'
    : 'none';

  const alignmentClass =
    align === 'right'
      ? 'text-right'
      : align === 'center'
      ? 'text-center'
      : 'text-left';

  const buttonColorClasses = isActive
    ? 'text-neutral-900'
    : 'text-neutral-500 hover:text-neutral-900';

  return (
    <th
      scope="col"
      aria-sort={ariaSort}
      className={`px-4 py-2 ${alignmentClass}`}
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 select-none focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-300 rounded-sm ${buttonColorClasses}`}
      >
        <span>{label}</span>
        {isActive && (
          <span aria-hidden="true" className="text-[0.7rem] leading-none">
            {currentSortDir === 'asc' ? '▲' : '▼'}
          </span>
        )}
      </button>
    </th>
  );
}