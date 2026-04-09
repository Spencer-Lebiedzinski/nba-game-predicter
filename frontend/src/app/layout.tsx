import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AstroHoops Analytics',
  description: 'AI-powered NBA prediction platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen flex flex-col relative`}>
        {/* Starry Background Animation */}
        <div className="fixed inset-0 z-[-1] stars-bg opacity-40"></div>

        {/* Top Navigation */}
        <nav className="glass-panel sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-opacity-20 border-white">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-black tracking-tighter neon-text-gradient">AstroHoops</h1>
          </div>
          <div className="hidden md:flex space-x-8 text-sm font-semibold tracking-wide text-gray-300">
            <a href="/" className="hover:text-[#45f3ff] transition-colors">Dashboard</a>
            <a href="/predictions" className="hover:text-[#45f3ff] transition-colors">Predictions</a>
            <a href="/live" className="hover:text-[#45f3ff] transition-colors">Live Games</a>
            <a href="/insights" className="hover:text-[#45f3ff] transition-colors">Model Insights</a>
            <a href="/history" className="hover:text-[#45f3ff] transition-colors">Historical Accuracy</a>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 w-full max-w-7xl mx-auto p-6 md:p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
