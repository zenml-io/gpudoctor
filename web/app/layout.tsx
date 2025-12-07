import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap'
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap'
});

export const metadata: Metadata = {
  title: {
    default: 'GPU Doctor',
    template: '%s | GPU Doctor'
  },
  description:
    'A professional interface for ML engineers to find the perfect GPU-enabled Docker image.',
  metadataBase: new URL('https://gpudoctor.zenml.io'),
  openGraph: {
    title: 'GPU Doctor',
    description:
      'Browse and compare GPU-optimized Docker images for PyTorch, TensorFlow, LLM inference and more.',
    url: 'https://gpudoctor.zenml.io',
    siteName: 'GPU Doctor',
    type: 'website'
  },
  twitter: {
    card: 'summary_large_image',
    title: 'GPU Doctor',
    description:
      'Find the right GPU Docker image for your ML workloads across major providers and frameworks.'
  }
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen bg-neutral-50 text-neutral-900 antialiased`}
      >
        {children}
      </body>
    </html>
  );
}