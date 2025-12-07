import Link from 'next/link';

/**
 * Simple neutral footer anchored to the bottom of the layout shell.
 * The year is fixed to 2025 to match the initial launch timeframe.
 */
export function Footer() {
  return (
    <footer className="border-t border-neutral-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 text-xs text-neutral-500 sm:text-sm">
        <span>© 2025 ZenML</span>
        <Link
          href="https://github.com/zenml-io/gpudoctor"
          target="_blank"
          rel="noreferrer"
          className="transition-colors hover:text-neutral-900"
        >
          GitHub
        </Link>
      </div>
    </footer>
  );
}