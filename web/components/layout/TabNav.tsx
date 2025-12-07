import Link from 'next/link';
import type { Route } from 'next';
import clsx from 'clsx';

export type TabKey = 'guide' | 'table';

interface TabNavProps {
  activeTab: TabKey;
}

/**
 * Primary navigation between the Guide and Table views.
 * Implemented as pill-style links so URLs stay shareable and indexable.
 */
export function TabNav({ activeTab }: TabNavProps) {
  return (
    <nav aria-label="Primary">
      <div className="inline-flex rounded-full bg-neutral-100 p-1 shadow-inner">
        <TabLink href="/guide" isActive={activeTab === 'guide'}>
          Guide
        </TabLink>
        <TabLink href="/table" isActive={activeTab === 'table'}>
          Table
        </TabLink>
      </div>
    </nav>
  );
}

interface TabLinkProps {
  href: Route;
  isActive: boolean;
  children: React.ReactNode;
}

function TabLink({ href, isActive, children }: TabLinkProps) {
  return (
    <Link
      href={href}
      className={clsx(
        'rounded-full px-4 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400',
        isActive
          ? 'bg-primary-500 text-white shadow-sm'
          : 'text-neutral-600 hover:bg-white hover:text-neutral-900'
      )}
      aria-current={isActive ? 'page' : undefined}
    >
      {children}
    </Link>
  );
}