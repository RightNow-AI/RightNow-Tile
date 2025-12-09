'use client';

import { useTheme } from './ThemeProvider';
import { Sun, Moon } from 'lucide-react';

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="flex items-center justify-center w-8 h-8 rounded-md border transition-colors
        border-gray-200 hover:border-gray-300 hover:bg-gray-50
        dark:border-white/20 dark:hover:border-white/40 dark:hover:bg-white/5"
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      {theme === 'light' ? (
        <Moon className="h-4 w-4 text-gray-600 dark:text-white/70" />
      ) : (
        <Sun className="h-4 w-4 text-gray-600 dark:text-white/70" />
      )}
    </button>
  );
}
