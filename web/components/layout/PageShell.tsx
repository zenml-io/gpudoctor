import type { ReactNode } from 'react';

import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { TabNav, type TabKey } from '@/components/layout/TabNav';

interface PageShellProps {
  /**
   * Which primary tab is active for this page.
   * Use null on detail pages where the tab navigation should be hidden.
   */
  activeTab: TabKey | null;
  children: ReactNode;
}

/**
 * High-level page chrome that applies the global header, optional tab navigation,
 * constrained content width, and footer. Individual routes focus purely on their
 * content while PageShell keeps the outer layout consistent.
 */
export function PageShell({ activeTab, children }: PageShellProps) {
  return (
    <div className="flex min-h-screen flex-col bg-neutral-50">
      <Header />
      <main className="flex-1">
        {activeTab && (
          <div className="border-b border-neutral-200 bg-white">
            <div className="mx-auto max-w-6xl px-4 py-4">
              <TabNav activeTab={activeTab} />
            </div>
          </div>
        )}
        <div className="mx-auto max-w-6xl px-4 py-8">{children}</div>
      </main>
      <Footer />
    </div>
  );
}