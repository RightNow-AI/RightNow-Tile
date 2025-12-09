'use client';

import { useState, useEffect, useRef } from 'react';
import { ArrowRight, Copy, Check, Github, Maximize2, Minimize2 } from 'lucide-react';
import dynamic from 'next/dynamic';
import Image from 'next/image';
import { motion } from 'framer-motion';
import ScientificVisualization from './components/ScientificVisualization';
import ThemeToggle from './components/ThemeToggle';
import PromoModal from './components/PromoModal';
import { useTheme } from './components/ThemeProvider';

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

const DiscordIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.0371c-1.4712.2492-3.0103.8227-4.8852 1.5152a.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276c-.598.3428-1.2205.6447-1.8733.8923a.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6601a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1835 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.4189 0 1.3333-.9555 2.419-2.1569 2.419zm7.9748 0c-1.1835 0-2.1568-1.0857-2.1568-2.419 0-1.3332.9554-2.4189 2.1568-2.4189 1.2108 0 2.1757 1.0952 2.1569 2.4189 0 1.3333-.9461 2.419-2.1569 2.419Z" />
  </svg>
);

const EXAMPLE_CUDA = `__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}`;

interface TranspileResult {
  tileCode: string;
  pattern: {
    archetype: string;
    variant?: string;
    confidence: number;
    evidence: Array<{ type: string; weight: number; description: string }>;
    warnings: string[];
  };
  variant?: string;
  validation: {
    isValid: boolean;
    adjustedConfidence: number;
    warnings: string[];
    errors?: string[];
  };
  diagnostics?: Array<{
    code: string;
    severity: 'error' | 'warning' | 'info' | 'hint';
    category: string;
    message: string;
    suggestions?: string[];
  }>;
  semanticAnalysis?: {
    reductionVariables: Array<{ name: string; operation: string }>;
    parallelismType: string;
  };
  memoryAnalysis?: {
    globalMemory: { coalescingScore: number };
    tileRecommendation: { recommended: { tileSize?: number } };
  };
}

export default function Home() {
  const { isDark } = useTheme();
  const [cudaCode, setCudaCode] = useState(EXAMPLE_CUDA);
  const [tileCode, setTileCode] = useState('');
  const [displayedCode, setDisplayedCode] = useState('');
  const [isTranspiling, setIsTranspiling] = useState(false);
  const [hasTranspiled, setHasTranspiled] = useState(false);
  const [result, setResult] = useState<TranspileResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isOutputExpanded, setIsOutputExpanded] = useState(false);
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const totalStages = 9; // Enhanced pipeline: Extractor, Parser, Semantic, Memory, Pattern, IR, Optimizer, Templates, Diagnostics

  // Stream code line by line
  const streamCode = (code: string) => {
    // Trim the code to remove any leading/trailing whitespace
    const trimmedCode = code.trim();
    const lines = trimmedCode.split('\n');
    let currentLine = 0;
    setDisplayedCode('');
    setIsStreaming(true);

    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
    }

    streamIntervalRef.current = setInterval(() => {
      if (currentLine < lines.length) {
        const line = lines[currentLine] ?? '';
        setDisplayedCode(prev => prev + (currentLine > 0 ? '\n' : '') + line);
        currentLine++;
      } else {
        if (streamIntervalRef.current) {
          clearInterval(streamIntervalRef.current);
        }
        setIsStreaming(false);
      }
    }, 40);
  };

  useEffect(() => {
    return () => {
      if (streamIntervalRef.current) {
        clearInterval(streamIntervalRef.current);
      }
    };
  }, []);

  const handleTranspile = async () => {
    setIsTranspiling(true);
    setHasTranspiled(true);
    setCurrentStage(0);
    setDisplayedCode('');
    setTileCode('');

    // Animate through stages
    const stageInterval = setInterval(() => {
      setCurrentStage(prev => {
        if (prev >= totalStages - 1) {
          clearInterval(stageInterval);
          return prev;
        }
        return prev + 1;
      });
    }, 350);

    try {
      const response = await fetch('/api/transpile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: cudaCode }),
      });
      const data = await response.json();

      // Wait for visualization
      await new Promise(resolve => setTimeout(resolve, totalStages * 350 + 300));
      clearInterval(stageInterval);
      setCurrentStage(totalStages);

      if (data.success) {
        const cleanCode = (data.result.tileCode || '').trim();
        setTileCode(cleanCode);
        setResult(data.result);
        setTimeout(() => streamCode(cleanCode), 200);
      } else {
        const errorCode = '# Error: ' + (data.error || 'Transpilation failed');
        setTileCode(errorCode);
        setResult(null);
        streamCode(errorCode);
      }
    } catch {
      const errorCode = '# Error: Could not connect to server';
      setTileCode(errorCode);
      setResult(null);
      streamCode(errorCode);
    }
    setIsTranspiling(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(tileCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* Header */}
      <header className="border-b" style={{ borderColor: 'var(--border-primary)', backgroundColor: 'var(--bg-secondary)' }}>
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-14 items-center justify-between">
            <a href="https://rightnowai.co" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
              <Image
                src="/logo.webp"
                alt="RightNow AI logo"
                width={28}
                height={28}
                className={`select-none ${isDark ? '' : 'invert'}`}
              />
              <div>
                <h1 className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>CUDA Tile Transpiler</h1>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Convert CUDA kernels to cuTile Python for Blackwell GPUs · by <span className="text-[#76B900]">RightNow AI</span></p>
              </div>
            </a>
            <div className="flex items-center gap-3">
              <ThemeToggle />
              <a
                href="https://github.com/RightNow-AI/RightNow-Tile"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 transition-colors text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                <Github className="h-4 w-4" />
                <span className="hidden sm:inline">GitHub</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="py-6 flex-1">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          {/* Editors - Always visible, animate sizes */}
          <div className="flex gap-4">
            {/* CUDA Input - shrinks when transpiled, hides when output expanded */}
            <motion.div
              className="editor-container overflow-hidden"
              animate={{
                flex: isOutputExpanded ? '0 0 0%' : hasTranspiled ? '0 0 35%' : '1 1 50%',
                opacity: isOutputExpanded ? 0 : 1,
                marginRight: isOutputExpanded ? 0 : undefined,
              }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ display: isOutputExpanded ? 'none' : 'block' }}
            >
              <div className="editor-header">
                <span className="editor-label">CUDA SIMT</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Input</span>
              </div>
              <motion.div
                animate={{ height: hasTranspiled ? 250 : 400 }}
                transition={{ duration: 0.6, ease: 'easeInOut' }}
                style={{ backgroundColor: 'var(--bg-editor)' }}
              >
                <MonacoEditor
                  height="100%"
                  language="cpp"
                  theme={isDark ? 'vs-dark' : 'vs'}
                  value={cudaCode}
                  onChange={(value) => setCudaCode(value || '')}
                  options={{
                    fontFamily: 'JetBrains Mono, Menlo, Monaco, monospace',
                    fontSize: hasTranspiled ? 11 : 13,
                    lineHeight: hasTranspiled ? 16 : 20,
                    fontLigatures: false,
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    padding: { top: 12, bottom: 12 },
                    renderLineHighlight: 'none',
                    overviewRulerBorder: false,
                    hideCursorInOverviewRuler: true,
                    scrollbar: { vertical: 'hidden', horizontal: 'hidden' },
                    readOnly: isTranspiling,
                    cursorBlinking: 'smooth',
                    cursorSmoothCaretAnimation: 'on',
                  }}
                />
              </motion.div>
            </motion.div>

            {/* cuTile Output - expands when transpiled, can be expanded further */}
            <motion.div
              className="editor-container overflow-hidden"
              animate={{
                flex: isOutputExpanded ? '1 1 100%' : hasTranspiled ? '0 0 65%' : '1 1 50%',
              }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
            >
              <div className="editor-header">
                <span className="editor-label flex items-center gap-2">
                  cuTile Python
                  {isStreaming && (
                    <span className="flex items-center gap-1 text-[#76B900]">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#76B900] animate-pulse" />
                      <span className="text-xs font-normal">streaming</span>
                    </span>
                  )}
                </span>
                <div className="flex items-center gap-3">
                  {tileCode && !isStreaming && (
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1 transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      {copied ? (
                        <Check className="h-3.5 w-3.5 text-[#76B900]" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                      <span className="text-xs">{copied ? 'Copied' : 'Copy'}</span>
                    </button>
                  )}
                  {hasTranspiled && (
                    <button
                      onClick={() => setIsOutputExpanded(!isOutputExpanded)}
                      className="flex items-center gap-1 transition-colors hover:opacity-70"
                      style={{ color: 'var(--text-muted)' }}
                      title={isOutputExpanded ? 'Collapse editor' : 'Expand editor'}
                    >
                      {isOutputExpanded ? (
                        <Minimize2 className="h-3.5 w-3.5" />
                      ) : (
                        <Maximize2 className="h-3.5 w-3.5" />
                      )}
                      <span className="text-xs">{isOutputExpanded ? 'Collapse' : 'Expand'}</span>
                    </button>
                  )}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Output</span>
                </div>
              </div>
              <motion.div
                animate={{ height: isOutputExpanded ? 600 : hasTranspiled ? 450 : 400 }}
                transition={{ duration: 0.4, ease: 'easeInOut' }}
                style={{ backgroundColor: 'var(--bg-editor)' }}
              >
                {hasTranspiled ? (
                  <MonacoEditor
                    height="100%"
                    language="python"
                    theme={isDark ? 'vs-dark' : 'vs'}
                    value={displayedCode || '# Generating...'}
                    onChange={(value) => {
                      if (!isStreaming) {
                        setDisplayedCode(value || '');
                        setTileCode(value || '');
                      }
                    }}
                    options={{
                      fontFamily: 'JetBrains Mono, Menlo, Monaco, monospace',
                      fontSize: 13,
                      lineHeight: 20,
                      fontLigatures: false,
                      minimap: {
                        enabled: isOutputExpanded,
                        scale: 1,
                        showSlider: 'always',
                      },
                      scrollBeyondLastLine: false,
                      padding: { top: 12, bottom: 12 },
                      renderLineHighlight: isOutputExpanded ? 'line' : 'none',
                      overviewRulerBorder: false,
                      hideCursorInOverviewRuler: true,
                      readOnly: isStreaming,
                      scrollbar: { vertical: 'auto', horizontal: 'hidden' },
                      cursorBlinking: 'smooth',
                      cursorSmoothCaretAnimation: 'on',
                    }}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center" style={{ color: 'var(--text-muted)' }}>
                    <div className="text-center">
                      <p className="text-sm mb-2">Click Transpile to convert your CUDA code</p>
                      <p className="text-xs opacity-60">Generated cuTile Python will appear here</p>
                    </div>
                  </div>
                )}
              </motion.div>
            </motion.div>
          </div>

          {/* Transpile Button */}
          <div className="flex justify-center mt-6">
            <button
              onClick={handleTranspile}
              disabled={isTranspiling || !cudaCode.trim()}
              className="rn-button-pro gap-2 px-8 py-3 text-base rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>{isTranspiling ? 'Transpiling...' : 'Transpile'}</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          {/* Scientific Visualization - always visible */}
          <ScientificVisualization
            isActive={isTranspiling}
            currentStage={currentStage}
            totalStages={totalStages}
            isDark={isDark}
          />

          {/* Pattern Evidence - after transpile completes */}
          {result && result.pattern.evidence && result.pattern.evidence.length > 0 && !isTranspiling && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mt-6"
            >
              <h3 className="text-xs font-medium uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                Pattern Evidence
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
                {result.pattern.evidence.slice(0, 6).map((ev, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 + index * 0.1 }}
                    className="p-3 rounded"
                    style={{
                      backgroundColor: isDark ? '#1b1913' : '#f5f5f5',
                      border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                    }}
                  >
                    <div
                      className="text-lg font-bold mb-1"
                      style={{ color: ev.weight > 0.5 ? '#76B900' : 'var(--text-secondary)' }}
                    >
                      {Math.round(ev.weight * 100)}%
                    </div>
                    <div className="text-xs font-medium uppercase" style={{ color: 'var(--text-primary)' }}>
                      {ev.type}
                    </div>
                    <div className="text-xs mt-1 truncate" style={{ color: 'var(--text-muted)' }}>
                      {ev.description}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Analysis Info */}
          {result && !isTranspiling && (result.memoryAnalysis || result.semanticAnalysis || result.diagnostics) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-2"
            >
              {result.memoryAnalysis && (
                <div
                  className="p-3 rounded"
                  style={{
                    backgroundColor: isDark ? '#1b1913' : '#f5f5f5',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <h4 className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                    Memory Analysis
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-muted)' }}>Coalescing</span>
                      <span style={{ color: result.memoryAnalysis.globalMemory.coalescingScore > 0.7 ? '#76B900' : 'var(--text-secondary)' }}>
                        {Math.round(result.memoryAnalysis.globalMemory.coalescingScore * 100)}%
                      </span>
                    </div>
                    {result.memoryAnalysis.tileRecommendation?.recommended?.tileSize && (
                      <div className="flex justify-between text-xs">
                        <span style={{ color: 'var(--text-muted)' }}>Recommended Tile</span>
                        <span style={{ color: 'var(--text-primary)' }}>
                          {result.memoryAnalysis.tileRecommendation.recommended.tileSize}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {result.semanticAnalysis && (
                <div
                  className="p-3 rounded"
                  style={{
                    backgroundColor: isDark ? '#1b1913' : '#f5f5f5',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <h4 className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                    Semantic Analysis
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span style={{ color: 'var(--text-muted)' }}>Parallelism</span>
                      <span style={{ color: 'var(--text-primary)' }}>{result.semanticAnalysis.parallelismType}</span>
                    </div>
                    {result.semanticAnalysis.reductionVariables?.length > 0 && (
                      <div className="flex justify-between text-xs">
                        <span style={{ color: 'var(--text-muted)' }}>Reductions</span>
                        <span style={{ color: 'var(--text-primary)' }}>
                          {result.semanticAnalysis.reductionVariables.map(r => r.operation).join(', ')}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {result.diagnostics && result.diagnostics.length > 0 && (
                <div
                  className="p-3 rounded"
                  style={{
                    backgroundColor: isDark ? '#1b1913' : '#f5f5f5',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                  }}
                >
                  <h4 className="text-xs font-medium uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                    Diagnostics
                  </h4>
                  <div className="space-y-1 max-h-20 overflow-y-auto">
                    {result.diagnostics.slice(0, 3).map((diag, idx) => (
                      <div key={idx} className="text-xs flex items-start gap-1">
                        <span style={{ color: 'var(--text-secondary)' }}>
                          [{diag.code}]
                        </span>
                        <span style={{ color: 'var(--text-muted)' }} className="truncate">
                          {diag.message.split(':').slice(-1)[0].trim()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Info Banner */}
          <div
            className="mt-6 p-4 rounded"
            style={{
              backgroundColor: isDark ? 'rgba(245, 158, 11, 0.1)' : 'rgb(254, 252, 232)',
              border: `1px solid ${isDark ? 'rgba(245, 158, 11, 0.3)' : 'rgb(253, 230, 138)'}`,
            }}
          >
            <p
              className="text-sm text-center"
              style={{ color: isDark ? '#F59E0B' : 'rgb(146, 64, 14)' }}
            >
              <strong>Note:</strong> cuTile requires NVIDIA Blackwell GPUs (compute capability 10.x+).
              This tool generates cuTile Python code from CUDA SIMT kernels.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 border-t" style={{ borderColor: 'var(--border-primary)', backgroundColor: 'var(--bg-secondary)' }}>
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span>Open source by</span>
              <a href="https://rightnowai.co" target="_blank" rel="noopener noreferrer" className="text-[#76B900] hover:underline font-medium">
                RightNow AI
              </a>
              <span style={{ color: 'var(--border-primary)' }}>·</span>
              <a
                href="https://docs.nvidia.com/cuda/cutile-python/"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline transition-colors"
                style={{ color: 'var(--text-muted)' }}
              >
                cuTile Docs
              </a>
            </div>
            <a
              href="https://discord.gg/sSJqgNnq6X"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#5865F2] text-white text-xs font-medium rounded hover:bg-[#4752c4] transition-colors"
            >
              <DiscordIcon />
              Join Discord
            </a>
          </div>
        </div>
      </footer>

      {/* Promo Modal */}
      <PromoModal isDark={isDark} />
    </div>
  );
}
