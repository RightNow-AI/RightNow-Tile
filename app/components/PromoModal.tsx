'use client';

import { useState, useEffect, useRef } from 'react';
import { X, ArrowRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface PromoModalProps {
  isDark: boolean;
}

export default function PromoModal({ isDark }: PromoModalProps) {
  const [isOpen, setIsOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Show the modal after a short delay
  useEffect(() => {
    const dismissed = sessionStorage.getItem('promo-modal-dismissed');
    if (dismissed) return;

    const timer = setTimeout(() => {
      setIsOpen(true);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  // Auto-play video when modal opens
  useEffect(() => {
    if (isOpen && videoRef.current) {
      videoRef.current.play().catch(() => {});
    }
  }, [isOpen]);

  const handleClose = () => {
    setIsOpen(false);
    sessionStorage.setItem('promo-modal-dismissed', 'true');
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 40 }}
          transition={{ type: 'spring', damping: 30, stiffness: 400 }}
          className="fixed bottom-6 right-6 z-50"
          style={{ width: '380px' }}
        >
          <div
            className="rounded-lg overflow-hidden"
            style={{
              backgroundColor: isDark ? '#1a1814' : '#ffffff',
              boxShadow: isDark
                ? '0 25px 50px -12px rgba(0,0,0,0.7), 0 0 0 1px rgba(118,185,0,0.2)'
                : '0 25px 50px -12px rgba(0,0,0,0.25), 0 0 0 1px rgba(0,0,0,0.05)',
            }}
          >
            {/* Video - Full width at top, auto-playing */}
            <div className="relative">
              <video
                ref={videoRef}
                src="/videos/ai-that-know-your-gpu.mp4"
                className="w-full aspect-video object-cover"
                muted
                loop
                playsInline
                autoPlay
              />

              {/* Close button overlaid on video */}
              <button
                onClick={handleClose}
                className="absolute top-2 right-2 w-7 h-7 rounded-full flex items-center justify-center transition-all hover:scale-110"
                style={{
                  backgroundColor: 'rgba(0,0,0,0.6)',
                  backdropFilter: 'blur(4px)',
                }}
                aria-label="Close"
              >
                <X className="w-4 h-4 text-white" />
              </button>

              {/* NVIDIA green accent line */}
              <div
                className="absolute bottom-0 left-0 right-0 h-0.5"
                style={{ backgroundColor: '#76B900' }}
              />
            </div>

            {/* Content */}
            <div className="p-4">
              {/* Title */}
              <h3
                className="text-base font-bold mb-1"
                style={{ color: isDark ? '#ffffff' : '#1a1a1a' }}
              >
                Try RightNow Editor
              </h3>
              <p
                className="text-sm mb-4"
                style={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}
              >
                Full GPU kernel development with AI
              </p>

              {/* CTA Button */}
              <a
                href="https://rightnowai.co"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded text-sm font-bold uppercase transition-all hover:brightness-110"
                style={{
                  backgroundColor: '#76B900',
                  color: '#000000',
                }}
              >
                <span>Get Started Free</span>
                <ArrowRight className="w-4 h-4" />
              </a>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
