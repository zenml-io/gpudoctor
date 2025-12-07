import type { ImageEntry } from '@/lib/types/images';
import { EmptyState } from '@/components/ui/EmptyState';
import { ImageCard } from '@/components/guide/ImageCard';
import { DataRow } from '@/components/table/DataRow';

interface DataTableProps {
  images: ImageEntry[];
}

/**
 * Responsive data table for the table view.
 * On desktop it renders a classic HTML table; on mobile it falls back to
 * the compact ImageCard list layout used in the guide.
 */
export function DataTable({ images }: DataTableProps) {
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
      <div className="hidden overflow-hidden rounded-lg border border-neutral-200 bg-white shadow-card md:block">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-neutral-100 text-xs font-semibold uppercase tracking-wide text-neutral-500">
            <tr>
              <th className="px-4 py-2 text-left">Image</th>
              <th className="px-4 py-2 text-left">Framework</th>
              <th className="px-4 py-2 text-left">CUDA</th>
              <th className="px-4 py-2 text-left">Python</th>
              <th className="px-4 py-2 text-left">Status</th>
              <th className="px-4 py-2 text-left">Provider</th>
            </tr>
          </thead>
          <tbody className="bg-white">
            {images.map((image) => (
              <DataRow key={image.id} image={image} />
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-3 md:hidden">
        {images.map((image) => (
          <ImageCard key={image.id} image={image} />
        ))}
      </div>
    </>
  );
}