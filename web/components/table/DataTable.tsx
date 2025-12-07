import type { ImageEntry } from '@/lib/types/images';
import type { TableSortBy, TableSortDir } from '@/lib/url/tableSearchParams';
import { EmptyState } from '@/components/ui/EmptyState';
import { ImageCard } from '@/components/guide/ImageCard';
import { DataRow } from '@/components/table/DataRow';
import { SortableHeader } from '@/components/table/SortableHeader';

interface DataTableProps {
  images: ImageEntry[];
  sortBy: TableSortBy;
  sortDir: TableSortDir;
  onSort: (column: TableSortBy) => void;
}

/**
 * Responsive data table for the table view.
 * On desktop it renders a classic HTML table; on mobile it falls back to
 * the compact ImageCard list layout used in the guide.
 */
export function DataTable({
  images,
  sortBy,
  sortDir,
  onSort
}: DataTableProps) {
  if (images.length === 0) {
    return (
      <EmptyState
        title="No images match your filters"
        description="Try adjusting the filters or clearing them to see more results."
      />
    );
  }

  return (
    <>
      <div className="hidden rounded-lg border border-neutral-200 bg-white shadow-card md:block">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-neutral-100 text-xs font-semibold uppercase tracking-wide text-neutral-500">
              <tr>
                <SortableHeader
                  label="Image"
                  sortKey="name"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <th className="px-4 py-2 text-left">Framework</th>
                <SortableHeader
                  label="Role"
                  sortKey="role"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="Type"
                  sortKey="imageType"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="CUDA"
                  sortKey="cuda"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <th className="px-4 py-2 text-left">cuDNN</th>
                <SortableHeader
                  label="Python"
                  sortKey="python"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="OS"
                  sortKey="os"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="Arch"
                  sortKey="arch"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="Size"
                  sortKey="size"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="right"
                />
                <SortableHeader
                  label="Status"
                  sortKey="status"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <SortableHeader
                  label="Provider"
                  sortKey="provider"
                  currentSortBy={sortBy}
                  currentSortDir={sortDir}
                  onSort={onSort}
                  align="left"
                />
                <th className="px-4 py-2 text-left">License</th>
              </tr>
            </thead>
            <tbody className="bg-white">
              {images.map((image) => (
                <DataRow key={image.id} image={image} />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="space-y-3 md:hidden">
        {images.map((image) => (
          <ImageCard key={image.id} image={image} />
        ))}
      </div>
    </>
  );
}