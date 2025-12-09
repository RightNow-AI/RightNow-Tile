import type { Metadata, Viewport } from 'next';
import { Analytics } from '@vercel/analytics/next';
import './globals.css';
import { ThemeProvider } from './components/ThemeProvider';

const siteUrl = 'https://tile.rightnowai.co';
const siteName = 'RightNow Tile';
const siteDescription = 'Convert CUDA SIMT kernels to NVIDIA cuTile Python code. Production-grade transpiler with pattern detection, semantic analysis, and optimized code generation for Blackwell GPUs.';

export const metadata: Metadata = {
  // Basic metadata
  title: {
    default: 'RightNow Tile - CUDA to cuTile Transpiler for Blackwell GPUs',
    template: '%s | RightNow Tile',
  },
  description: siteDescription,
  keywords: [
    'CUDA',
    'cuTile',
    'transpiler',
    'GPU',
    'NVIDIA',
    'Blackwell',
    'SIMT',
    'GPU programming',
    'CUDA converter',
    'cuTile Python',
    'GPU kernel',
    'tile-based programming',
    'CUDA to Python',
    'GPU optimization',
    'parallel computing',
    'HPC',
    'high performance computing',
    'matrix multiplication',
    'GEMM',
    'reduction kernel',
    'stencil computation',
    'RightNow AI',
  ],
  authors: [{ name: 'RightNow AI', url: 'https://rightnowai.co' }],
  creator: 'RightNow AI',
  publisher: 'RightNow AI',

  // Canonical URL
  metadataBase: new URL(siteUrl),
  alternates: {
    canonical: '/',
  },

  // Open Graph (Facebook, LinkedIn, etc.)
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: siteUrl,
    siteName: siteName,
    title: 'RightNow Tile - CUDA to cuTile Transpiler',
    description: siteDescription,
    images: [
      {
        url: '/og-image.webp',
        width: 1200,
        height: 630,
        alt: 'RightNow Tile - CUDA to cuTile Transpiler for NVIDIA Blackwell GPUs',
      },
    ],
  },

  // Twitter Card
  twitter: {
    card: 'summary_large_image',
    title: 'RightNow Tile - CUDA to cuTile Transpiler',
    description: siteDescription,
    images: ['/og-image.webp'],
    creator: '@RightNowAI',
  },

  // Robots
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },

  // Icons
  icons: {
    icon: '/favicon.ico',
  },

  // Manifest
  manifest: '/site.webmanifest',

  // Category
  category: 'technology',

  // Other
  applicationName: siteName,
  referrer: 'origin-when-cross-origin',
  generator: 'Next.js',
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#14120b' },
  ],
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Structured Data (JSON-LD) */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'WebApplication',
              name: 'RightNow Tile',
              description: siteDescription,
              url: siteUrl,
              applicationCategory: 'DeveloperApplication',
              operatingSystem: 'Any',
              offers: {
                '@type': 'Offer',
                price: '0',
                priceCurrency: 'USD',
              },
              author: {
                '@type': 'Organization',
                name: 'RightNow AI',
                url: 'https://rightnowai.co',
              },
              publisher: {
                '@type': 'Organization',
                name: 'RightNow AI',
                url: 'https://rightnowai.co',
                logo: {
                  '@type': 'ImageObject',
                  url: `${siteUrl}/logo.webp`,
                },
              },
              screenshot: `${siteUrl}/og-image.webp`,
              featureList: [
                'CUDA to cuTile transpilation',
                'Pattern detection (GEMM, Reduction, Scan, Stencil)',
                'Semantic analysis',
                'Memory access optimization',
                'Real-time code generation',
              ],
              keywords: 'CUDA, cuTile, transpiler, NVIDIA, Blackwell, GPU programming',
            }),
          }}
        />
        {/* Software Application Schema */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'SoftwareSourceCode',
              name: 'RightNow Tile',
              description: 'CUDA SIMT to cuTile Python Transpiler',
              codeRepository: 'https://github.com/RightNow-AI/RightNow-Tile',
              programmingLanguage: ['TypeScript', 'CUDA', 'Python'],
              runtimePlatform: 'Node.js',
              targetProduct: {
                '@type': 'SoftwareApplication',
                name: 'cuTile Python',
                operatingSystem: 'NVIDIA Blackwell GPU',
              },
              author: {
                '@type': 'Organization',
                name: 'RightNow AI',
                url: 'https://rightnowai.co',
              },
              license: 'https://opensource.org/licenses/MIT',
            }),
          }}
        />
      </head>
      <body className="font-jetbrains antialiased">
        <ThemeProvider>
          {children}
        </ThemeProvider>
        <Analytics />
      </body>
    </html>
  );
}
