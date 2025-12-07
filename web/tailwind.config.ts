import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}'
  ],
  theme: {
    extend: {
      colors: {
        // ZenML primary purple scale
        primary: {
          900: '#1e1b4b',
          800: '#2e1065',
          700: '#4c1d95',
          600: '#6d28d9',
          500: '#7c3aed',
          400: '#8b5cf6',
          300: '#a78bfa',
          200: '#c4b5fd',
          100: '#ddd6fe',
          50: '#f5f3ff'
        },
        // Neutral gray scale
        neutral: {
          900: '#111827',
          800: '#1f2937',
          700: '#374151',
          600: '#4b5563',
          500: '#6b7280',
          400: '#9ca3af',
          300: '#d1d5db',
          200: '#e5e7eb',
          100: '#f3f4f6',
          50: '#f9fafb'
        },
        // Semantic colors
        success: {
          600: '#16a34a',
          500: '#22c55e',
          100: '#dcfce7'
        },
        warning: {
          600: '#d97706',
          500: '#f59e0b',
          100: '#fef3c7'
        },
        error: {
          600: '#dc2626',
          500: '#ef4444',
          100: '#fee2e2'
        },
        info: {
          600: '#2563eb',
          500: '#3b82f6',
          100: '#dbeafe'
        },
        // Extended UI accent colors
        teal: {
          500: '#14b8a6',
          100: '#ccfbf1'
        },
        turquoise: {
          500: '#06b6d4',
          100: '#cffafe'
        },
        lime: {
          500: '#84cc16',
          100: '#ecfccb'
        },
        magenta: {
          500: '#d946ef',
          100: '#fae8ff'
        },
        orange: {
          500: '#f97316',
          100: '#ffedd5'
        }
      },
      fontFamily: {
        // Wired to CSS variables set by next/font in layout.tsx
        sans: [
          'var(--font-sans)',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          'sans-serif'
        ],
        mono: [
          'var(--font-mono)',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Monaco',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace'
        ]
      },
      boxShadow: {
        // Subtle card shadow from design spec
        card: '0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px -1px rgba(0, 0, 0, 0.08)'
      }
    }
  },
  plugins: []
};

export default config;