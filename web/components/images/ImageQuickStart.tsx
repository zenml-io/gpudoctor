'use client';

import type { ImageEntry } from '@/lib/types/images';
import { CopyButton } from '@/components/ui/CopyButton';

interface ImageQuickStartProps {
  image: ImageEntry;
}

/**
 * Quick start section showing a docker pull command with copy-to-clipboard.
 */
export function ImageQuickStart({ image }: ImageQuickStartProps) {
  const command = `docker pull ${image.name}`;

  return (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold text-neutral-900">Quick start</h2>
      <div className="relative rounded-lg bg-neutral-900 px-4 py-3 text-neutral-50">
        <code className="block overflow-x-auto font-mono text-xs sm:text-sm">
          {command}
        </code>
        <div className="absolute right-3 top-2">
          <CopyButton text={command} />
        </div>
      </div>
    </section>
  );
}