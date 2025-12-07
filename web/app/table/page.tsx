import { Suspense } from 'react';
import { PageShell } from '@/components/layout/PageShell';
import { getAllImages } from '@/lib/data/images';
import { TableClient } from './TableClient';

export const dynamic = 'error';

export default function TablePage() {
  const images = getAllImages();

  return (
    <PageShell activeTab="table">
      <Suspense fallback={<div className="animate-pulse text-neutral-400">Loading table...</div>}>
        <TableClient images={images} />
      </Suspense>
    </PageShell>
  );
}