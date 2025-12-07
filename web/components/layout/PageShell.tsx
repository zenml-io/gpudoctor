import type { ReactNode } from 'react';

import { Header, type TabKey } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';

interface PageShellProps {
  /**
   * Which primary tab is active for this page.
   * Use null on detail pages where the tab navigation should be hidden.
   */
  activeTab: TabKey | null;
  children: ReactNode;
}

/**
 * High-level page chrome that applies the global header (with integrated navigation),
 * constrained content width, and footer. Individual routes focus purely on their
 * content while PageShell keeps the outer layout consistent.
 */
export function PageShell({ activeTab, children }: PageShellProps) {
  return (
    <div className="flex min-h-screen flex-col bg-neutral-50">
      <Header activeTab={activeTab} />
      <main className="flex-1">
        <div className="mx-auto max-w-6xl px-4 py-8">{children}</div>
      </main>
      <Footer />
    </div>
  );
}

export type { TabKey };
