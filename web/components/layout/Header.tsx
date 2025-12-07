import Link from 'next/link';
import { Cpu } from 'lucide-react';

/**
 * Site header with GPU DOCTOR branding and a link back to ZenML.
 * Kept visually lightweight so content and data remain the primary focus.
 */
export function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-neutral-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-primary-100 bg-primary-50 text-primary-600">
            <Cpu className="h-4 w-4" aria-hidden="true" />
          </div>
          <span className="text-lg font-semibold tracking-tight text-neutral-900">
            GPU DOCTOR
          </span>
        </Link>
{/* ZenML branding moved to CornerRibbon component */}
      </div>
    </header>
  );
}