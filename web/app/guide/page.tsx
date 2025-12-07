import { PageShell } from '@/components/layout/PageShell';
import { getAllImages } from '@/lib/data/images';
import { GuideClient } from './GuideClient';

export const dynamic = 'error';

export default function GuidePage() {
  const images = getAllImages();

  return (
    <PageShell activeTab="guide">
      <GuideClient images={images} />
    </PageShell>
  );
}