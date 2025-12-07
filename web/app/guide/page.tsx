import { Suspense } from 'react';
import { PageShell } from '@/components/layout/PageShell';
import { getAllImages } from '@/lib/data/images';
import { GuideClient } from './GuideClient';

export const dynamic = 'error';

export default function GuidePage() {
  const images = getAllImages();

  return (
    <PageShell activeTab="guide">
      <Suspense fallback={<div className="animate-pulse text-neutral-400">Loading guide...</div>}>
        <GuideClient images={images} />
      </Suspense>
    </PageShell>
  );
}