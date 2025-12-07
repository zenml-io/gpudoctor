'use client';

import Link from 'next/link';
import type { Route } from 'next';
import { Cpu } from 'lucide-react';
import clsx from 'clsx';

export type TabKey = 'guide' | 'table';

interface HeaderProps {
  activeTab?: TabKey | null;
}

/**
 * Site header with GPU Doctor branding and integrated navigation.
 * When activeTab is provided, the Guide/Table toggle appears in the header.
 */
export function Header({ activeTab }: HeaderProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-neutral-200 bg-white/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        {/* Logo and wordmark */}
        <Link
          href="/"
          className="group flex items-center gap-2.5 transition-opacity hover:opacity-80"
        >
          {/* Icon container with subtle hover effect */}
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-sm transition-transform duration-200 group-hover:scale-105">
            <Cpu className="h-4.5 w-4.5" aria-hidden="true" strokeWidth={2} />
            {/* Subtle inner glow */}
            <div className="absolute inset-0 rounded-xl bg-white/10" />
          </div>

          {/* Wordmark - refined typography */}
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-bold uppercase tracking-wide text-primary-600">
              GPU
            </span>
            <span className="text-lg font-semibold tracking-tight text-neutral-800">
              Doctor
            </span>
          </div>
        </Link>

        {/* Navigation toggle - only shown when activeTab is provided */}
        {activeTab && (
          <nav aria-label="Primary" className="flex items-center">
            <div className="flex rounded-lg border border-neutral-200 bg-neutral-50 p-0.5">
              <NavLink href="/guide" isActive={activeTab === 'guide'}>
                Guide
              </NavLink>
              <NavLink href="/table" isActive={activeTab === 'table'}>
                Table
              </NavLink>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
}

interface NavLinkProps {
  href: Route;
  isActive: boolean;
  children: React.ReactNode;
}

function NavLink({ href, isActive, children }: NavLinkProps) {
  return (
    <Link
      href={href}
      className={clsx(
        'relative rounded-md px-4 py-1.5 text-sm font-medium transition-all duration-150',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-400 focus-visible:ring-offset-1',
        isActive
          ? 'bg-white text-neutral-900 shadow-sm'
          : 'text-neutral-500 hover:text-neutral-700'
      )}
      aria-current={isActive ? 'page' : undefined}
    >
      {children}
    </Link>
  );
}
