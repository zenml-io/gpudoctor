import { PageShell } from '@/components/layout/PageShell';
import { getAllImages } from '@/lib/data/images';
import { TableClient } from './TableClient';

export const dynamic = 'error';

export default function TablePage() {
  const images = getAllImages();

  return (
    <PageShell activeTab="table">
      <TableClient images={images} />
    </PageShell>
  );
}