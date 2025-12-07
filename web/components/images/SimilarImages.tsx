import Link from 'next/link';

import type { ImageEntry } from '@/lib/types/images';
import { ImageCard } from '@/components/guide/ImageCard';
import { EmptyState } from '@/components/ui/EmptyState';

interface SimilarImagesProps {
  currentImage: ImageEntry;
  images: ImageEntry[];
}

/**
 * Displays a horizontal, scrollable list of images related to the current one.
 */
export function SimilarImages({ currentImage, images }: SimilarImagesProps) {
  if (images.length === 0) {
    return (
      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-neutral-900">
          Similar images
        </h2>
        <EmptyState
          title="No closely related images found"
          description="This image is relatively unique in the catalog. Browse all images to explore other options."
        />
      </section>
    );
  }

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-neutral-900">
          Similar images
        </h2>
        <Link
          href="/table"
          className="text-xs font-medium text-primary-600 hover:text-primary-700"
        >
          View all images →
        </Link>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-1">
        {images.map((image) => (
          <div
            key={image.id}
            className="min-w-[260px] max-w-xs flex-shrink-0"
          >
            <ImageCard image={image} />
          </div>
        ))}
      </div>
    </section>
  );
}