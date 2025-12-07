'use client';

import Link from 'next/link';

/**
 * A refined diagonal corner ribbon fixed to the viewport's upper-right.
 * Links to ZenML as a subtle branding element that doesn't distract from content.
 */
export function CornerRibbon() {
  return (
    <Link
      href="https://zenml.io"
      target="_blank"
      rel="noreferrer"
      className="
        fixed z-50
        top-0 right-0
        pointer-events-none
        hidden sm:block
      "
      aria-label="Made by ZenML - visit zenml.io"
    >
      {/* Ribbon container - positions the diagonal band */}
      <div
        className="
          relative
          w-[200px] h-[200px]
          overflow-hidden
        "
      >
        {/* The actual ribbon band */}
        <div
          className="
            pointer-events-auto
            absolute
            top-[35px] right-[-60px]
            w-[240px]
            py-1.5
            text-center
            font-medium text-xs tracking-wide
            text-white/95
            bg-gradient-to-r from-primary-600 via-primary-500 to-primary-400
            shadow-md
            transform rotate-45
            origin-center
            transition-all duration-300 ease-out
            hover:brightness-110 hover:shadow-lg
            hover:from-primary-500 hover:via-primary-400 hover:to-primary-300
            cursor-pointer
          "
          style={{
            boxShadow: '0 2px 8px rgba(124, 58, 237, 0.3)',
          }}
        >
          <span className="drop-shadow-sm">Made by ZenML</span>
        </div>
      </div>
    </Link>
  );
}
