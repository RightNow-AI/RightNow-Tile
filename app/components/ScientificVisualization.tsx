'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';

interface ScientificVisualizationProps {
  isActive: boolean;
  currentStage: number;
  totalStages: number;
  isDark: boolean;
  onComplete?: () => void;
}

interface TooltipInfo {
  title: string;
  description: string;
}

const TOOLTIPS: Record<string, TooltipInfo> = {
  cuda: {
    title: 'CUDA Source',
    description: 'Input CUDA SIMT kernel code with __global__ functions, thread indexing, shared memory, and synchronization primitives.',
  },
  extractor: {
    title: 'AST Extractor',
    description: 'Extracts kernel signatures, parameters, memory accesses, loops, and sync points from CUDA source.',
  },
  enhanced_parser: {
    title: 'Enhanced Parser',
    description: 'Advanced parsing with 150+ CUDA intrinsics detection, index pattern analysis, warp operations, and tensor core intrinsics.',
  },
  semantic: {
    title: 'Semantic Analyzer',
    description: 'Analyzes reduction variables, induction variables, data dependencies, race conditions, and barrier divergence.',
  },
  memory: {
    title: 'Memory Analyzer',
    description: 'Analyzes coalescing efficiency, bank conflicts, register pressure, and generates tile size recommendations.',
  },
  pattern: {
    title: 'Pattern Detector',
    description: '18 pattern matchers with 60+ variants. Core: Elementwise, GEMM, Reduction, Scan, Stencil. ML/DL: Convolution, Pooling, Normalization, Fused. LLM: Attention, RoPE, KV Cache, Embedding, Quantization. Specialized: Sparse, Histogram, Sorting, FFT.',
  },
  ir_builder: {
    title: 'IR Builder',
    description: 'Constructs intermediate representation with tile operations, memory layouts, and reduction operations.',
  },
  optimizer: {
    title: 'IR Optimizer',
    description: 'Optimizes tile configurations based on pattern variant and memory analysis. Generates performance hints.',
  },
  templates: {
    title: 'Template CodeGen',
    description: '14 template files with 60+ variant-specific generators. Supports Flash Attention, RoPE, KV Cache, Quantization, Convolution, Normalization, and more.',
  },
  diagnostics: {
    title: 'Diagnostics',
    description: 'Comprehensive error/warning system with diagnostic codes (E1xx-W5xx), suggestions, and performance hints.',
  },
  output: {
    title: 'cuTile Output',
    description: 'Generated cuTile Python code optimized for NVIDIA Blackwell GPUs with @ct.kernel, ct.load(), ct.reduce(), and more.',
  },
};

export default function ScientificVisualization({
  isActive,
  currentStage,
  isDark,
}: ScientificVisualizationProps) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const handleMouseEnter = (nodeId: string, e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setTooltipPos({ x: rect.left + rect.width / 2, y: rect.bottom + 8 });
    setHoveredNode(nodeId);
  };

  const handleMouseLeave = () => {
    setHoveredNode(null);
  };

  // Stage mapping for animation (10 stages now)
  const getNodeStatus = (nodeIndex: number) => {
    if (!isActive) return 'idle';
    if (nodeIndex < currentStage) return 'completed';
    if (nodeIndex === currentStage) return 'active';
    return 'pending';
  };

  const strokeColor = isDark ? '#e5e5e5' : '#1a1a1a';
  const fillColor = isDark ? '#1a1a1a' : '#ffffff';
  const textColor = isDark ? '#e5e5e5' : '#1a1a1a';
  const mutedColor = isDark ? '#666666' : '#888888';
  const greenColor = '#76B900';

  return (
    <div className="w-full mt-8 select-none">
      {/* Header */}
      <div className="px-4 py-2 flex items-center justify-between">
        <span
          className="text-xs font-mono"
          style={{ color: mutedColor }}
        >
          Transpilation Pipeline
        </span>
        {isActive && (
          <span className="flex items-center gap-1.5 text-xs font-mono" style={{ color: greenColor }}>
            <motion.span
              className="w-1.5 h-1.5 rounded-full bg-[#76B900]"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            Processing...
          </span>
        )}
      </div>

      {/* Flowchart SVG */}
      <div className="p-4 overflow-x-auto">
        <svg
          viewBox="0 0 1100 380"
          className="w-full h-auto min-w-[900px]"
          style={{ maxHeight: '380px' }}
        >
          <defs>
            {/* Arrow markers */}
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill={strokeColor} />
            </marker>
            <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill={greenColor} />
            </marker>
          </defs>

          {/* Analysis Layer boundary */}
          <rect
            x="195"
            y="15"
            width="390"
            height="170"
            fill="none"
            stroke={mutedColor}
            strokeWidth="1.5"
            strokeDasharray="6 3"
            rx="4"
            opacity="0.6"
          />
          <text x="390" y="32" textAnchor="middle" fill={mutedColor} fontSize="10" fontFamily="monospace" fontWeight="600">
            Analysis Layer
          </text>

          {/* Blackwell GPUs boundary */}
          <rect
            x="620"
            y="15"
            width="460"
            height="350"
            fill="none"
            stroke={greenColor}
            strokeWidth="2"
            strokeDasharray="8 4"
            rx="4"
          />
          <text x="850" y="32" textAnchor="middle" fill={greenColor} fontSize="11" fontFamily="monospace" fontWeight="600">
            Blackwell GPU Target
          </text>

          {/* ===== ROW 1: Main Pipeline ===== */}

          {/* CUDA Source */}
          <g
            onMouseEnter={(e) => handleMouseEnter('cuda', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <rect x="20" y="90" width="85" height="50" fill={fillColor} stroke={strokeColor} strokeWidth="1.5" />
            <text x="62" y="112" textAnchor="middle" fill={textColor} fontSize="10" fontFamily="monospace">CUDA</text>
            <text x="62" y="126" textAnchor="middle" fill={textColor} fontSize="10" fontFamily="monospace">Source</text>
          </g>

          {/* AST Extractor */}
          <g
            onMouseEnter={(e) => handleMouseEnter('extractor', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="135"
              y="90"
              width="85"
              height="50"
              fill={getNodeStatus(0) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(0) === 'completed' ? greenColor : getNodeStatus(0) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(0) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(0) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="177" y="112" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">AST</text>
            <text x="177" y="125" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Extractor</text>
            {getNodeStatus(0) === 'completed' && <text x="210" y="100" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* Enhanced Parser */}
          <g
            onMouseEnter={(e) => handleMouseEnter('enhanced_parser', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="250"
              y="90"
              width="85"
              height="50"
              fill={getNodeStatus(1) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(1) === 'completed' ? greenColor : getNodeStatus(1) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(1) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(1) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="292" y="112" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Enhanced</text>
            <text x="292" y="125" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Parser</text>
            {getNodeStatus(1) === 'completed' && <text x="325" y="100" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* ===== ROW 2: Analysis Branch ===== */}

          {/* Semantic Analyzer */}
          <g
            onMouseEnter={(e) => handleMouseEnter('semantic', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="365"
              y="50"
              width="90"
              height="45"
              fill={getNodeStatus(2) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(2) === 'completed' ? greenColor : getNodeStatus(2) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(2) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(2) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="410" y="70" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Semantic</text>
            <text x="410" y="83" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Analyzer</text>
            {getNodeStatus(2) === 'completed' && <text x="445" y="60" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* Memory Analyzer */}
          <g
            onMouseEnter={(e) => handleMouseEnter('memory', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="365"
              y="125"
              width="90"
              height="45"
              fill={getNodeStatus(3) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(3) === 'completed' ? greenColor : getNodeStatus(3) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(3) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(3) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="410" y="145" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Memory</text>
            <text x="410" y="158" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Analyzer</text>
            {getNodeStatus(3) === 'completed' && <text x="445" y="135" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* Pattern Detector - Diamond */}
          <g
            onMouseEnter={(e) => handleMouseEnter('pattern', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.polygon
              points="490,115 550,70 610,115 550,160"
              fill={getNodeStatus(4) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(4) === 'completed' ? greenColor : getNodeStatus(4) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(4) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(4) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="550" y="105" textAnchor="middle" fill={textColor} fontSize="8" fontFamily="monospace">Pattern</text>
            <text x="550" y="117" textAnchor="middle" fill={textColor} fontSize="8" fontFamily="monospace">Detector</text>
            <text x="550" y="129" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace">(18 types)</text>
            {getNodeStatus(4) === 'completed' && <text x="595" y="85" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* ===== ROW 3: Generation Pipeline ===== */}

          {/* IR Builder */}
          <g
            onMouseEnter={(e) => handleMouseEnter('ir_builder', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="650"
              y="90"
              width="80"
              height="50"
              fill={getNodeStatus(5) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(5) === 'completed' ? greenColor : getNodeStatus(5) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(5) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(5) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="690" y="112" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">IR</text>
            <text x="690" y="125" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Builder</text>
            {getNodeStatus(5) === 'completed' && <text x="720" y="100" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* IR Optimizer */}
          <g
            onMouseEnter={(e) => handleMouseEnter('optimizer', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="760"
              y="90"
              width="80"
              height="50"
              fill={getNodeStatus(6) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(6) === 'completed' ? greenColor : getNodeStatus(6) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(6) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(6) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="800" y="112" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">IR</text>
            <text x="800" y="125" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Optimizer</text>
            {getNodeStatus(6) === 'completed' && <text x="830" y="100" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* Template CodeGen */}
          <g
            onMouseEnter={(e) => handleMouseEnter('templates', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="870"
              y="90"
              width="90"
              height="50"
              fill={getNodeStatus(7) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(7) === 'completed' ? greenColor : getNodeStatus(7) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(7) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(7) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="915" y="108" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Template</text>
            <text x="915" y="121" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">CodeGen</text>
            <text x="915" y="133" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace">(variants)</text>
            {getNodeStatus(7) === 'completed' && <text x="950" y="100" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* Diagnostics */}
          <g
            onMouseEnter={(e) => handleMouseEnter('diagnostics', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <motion.rect
              x="870"
              y="170"
              width="90"
              height="45"
              fill={getNodeStatus(8) === 'active' ? `${greenColor}22` : fillColor}
              stroke={getNodeStatus(8) === 'completed' ? greenColor : getNodeStatus(8) === 'active' ? greenColor : strokeColor}
              strokeWidth={getNodeStatus(8) === 'active' ? 2 : 1.5}
              animate={getNodeStatus(8) === 'active' ? { strokeOpacity: [1, 0.5, 1] } : {}}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <text x="915" y="190" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Diagnostics</text>
            <text x="915" y="203" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace">(E/W/I codes)</text>
            {getNodeStatus(8) === 'completed' && <text x="950" y="180" fill={greenColor} fontSize="11">✓</text>}
          </g>

          {/* cuTile Output */}
          <g
            onMouseEnter={(e) => handleMouseEnter('output', e)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: 'pointer' }}
          >
            <path
              d={`M 990 85 L 990 155 Q 990 165 1000 165 L 1050 165 Q 1060 165 1060 155 L 1060 95 L 1050 85 L 990 85 M 1050 85 L 1050 95 L 1060 95`}
              fill={fillColor}
              stroke={getNodeStatus(8) === 'completed' ? greenColor : strokeColor}
              strokeWidth="1.5"
            />
            <path
              d="M 990 155 Q 980 155 980 165 Q 980 175 990 175 L 1050 175 Q 1060 175 1060 165"
              fill="none"
              stroke={getNodeStatus(8) === 'completed' ? greenColor : strokeColor}
              strokeWidth="1.5"
            />
            <text x="1025" y="115" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">cuTile</text>
            <text x="1025" y="130" textAnchor="middle" fill={textColor} fontSize="9" fontFamily="monospace">Output</text>
            <line x1="1000" y1="142" x2="1045" y2="142" stroke={mutedColor} strokeWidth="1" />
            <line x1="1000" y1="150" x2="1035" y2="150" stroke={mutedColor} strokeWidth="1" />
          </g>

          {/* ===== PATTERN LABELS ===== */}
          <g transform="translate(210, 200)">
            {/* Core Patterns */}
            <text x="0" y="0" textAnchor="start" fill={mutedColor} fontSize="7" fontFamily="monospace" fontWeight="600">Core:</text>
            <rect x="30" y="-8" width="40" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="50" y="2" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">GEMM</text>
            <rect x="75" y="-8" width="50" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="100" y="2" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Reduction</text>
            <rect x="130" y="-8" width="35" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="147" y="2" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Scan</text>
            <rect x="170" y="-8" width="40" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="190" y="2" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Stencil</text>

            {/* ML/DL Patterns */}
            <text x="0" y="18" textAnchor="start" fill={mutedColor} fontSize="7" fontFamily="monospace" fontWeight="600">ML/DL:</text>
            <rect x="35" y="10" width="55" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="62" y="20" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Convolution</text>
            <rect x="95" y="10" width="40" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="115" y="20" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Pooling</text>
            <rect x="140" y="10" width="60" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="170" y="20" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Normalization</text>

            {/* LLM Patterns */}
            <text x="0" y="36" textAnchor="start" fill={greenColor} fontSize="7" fontFamily="monospace" fontWeight="600">LLM:</text>
            <rect x="25" y="28" width="50" height="14" fill="none" stroke={greenColor} strokeWidth="0.5" rx="2" />
            <text x="50" y="38" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Attention</text>
            <rect x="80" y="28" width="35" height="14" fill="none" stroke={greenColor} strokeWidth="0.5" rx="2" />
            <text x="97" y="38" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">RoPE</text>
            <rect x="120" y="28" width="48" height="14" fill="none" stroke={greenColor} strokeWidth="0.5" rx="2" />
            <text x="144" y="38" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">KV Cache</text>
            <rect x="173" y="28" width="35" height="14" fill="none" stroke={greenColor} strokeWidth="0.5" rx="2" />
            <text x="190" y="38" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Quant</text>

            {/* Specialized */}
            <text x="0" y="54" textAnchor="start" fill={mutedColor} fontSize="7" fontFamily="monospace" fontWeight="600">Other:</text>
            <rect x="30" y="46" width="40" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="50" y="56" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Sparse</text>
            <rect x="75" y="46" width="50" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="100" y="56" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Histogram</text>
            <rect x="130" y="46" width="40" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="150" y="56" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">Sorting</text>
            <rect x="175" y="46" width="30" height="14" fill="none" stroke={mutedColor} strokeWidth="0.5" rx="2" />
            <text x="190" y="56" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">FFT</text>
          </g>

          {/* ===== VARIANT INFO ===== */}
          <g transform="translate(650, 200)">
            <text x="90" y="0" textAnchor="middle" fill={mutedColor} fontSize="8" fontFamily="monospace">60+ Variants:</text>
            <text x="90" y="15" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">flash_attention | flash_attention_v2 | multi_head_attention</text>
            <text x="90" y="28" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">rope_standard | rope_neox | kvcache_paged | quant_int8</text>
            <text x="90" y="41" textAnchor="middle" fill={textColor} fontSize="6" fontFamily="monospace">layernorm | rmsnorm | conv_2d | conv_depthwise | bitonic_sort</text>
          </g>

          {/* ===== ARROWS ===== */}

          {/* CUDA -> Extractor */}
          <line x1="105" y1="115" x2="133" y2="115" stroke={strokeColor} strokeWidth="1.5" markerEnd="url(#arrowhead)" />

          {/* Extractor -> Enhanced Parser */}
          <line x1="220" y1="115" x2="248" y2="115" stroke={strokeColor} strokeWidth="1.5" markerEnd="url(#arrowhead)" />
          <text x="234" y="108" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace" fontStyle="italic">AST</text>

          {/* Enhanced Parser -> Semantic (branch up) */}
          <line x1="335" y1="100" x2="350" y2="100" stroke={mutedColor} strokeWidth="1" />
          <line x1="350" y1="100" x2="350" y2="72" stroke={mutedColor} strokeWidth="1" />
          <line x1="350" y1="72" x2="363" y2="72" stroke={mutedColor} strokeWidth="1" markerEnd="url(#arrowhead)" />

          {/* Enhanced Parser -> Memory (branch down) */}
          <line x1="335" y1="125" x2="350" y2="125" stroke={mutedColor} strokeWidth="1" />
          <line x1="350" y1="125" x2="350" y2="147" stroke={mutedColor} strokeWidth="1" />
          <line x1="350" y1="147" x2="363" y2="147" stroke={mutedColor} strokeWidth="1" markerEnd="url(#arrowhead)" />

          {/* Semantic -> Pattern */}
          <line x1="455" y1="72" x2="475" y2="72" stroke={mutedColor} strokeWidth="1" />
          <line x1="475" y1="72" x2="475" y2="95" stroke={mutedColor} strokeWidth="1" />
          <line x1="475" y1="95" x2="490" y2="108" stroke={mutedColor} strokeWidth="1" markerEnd="url(#arrowhead)" />

          {/* Memory -> Pattern */}
          <line x1="455" y1="147" x2="475" y2="147" stroke={mutedColor} strokeWidth="1" />
          <line x1="475" y1="147" x2="475" y2="125" stroke={mutedColor} strokeWidth="1" />
          <line x1="475" y1="125" x2="490" y2="118" stroke={mutedColor} strokeWidth="1" markerEnd="url(#arrowhead)" />

          {/* Pattern -> IR Builder */}
          <line x1="610" y1="115" x2="648" y2="115" stroke={greenColor} strokeWidth="1.5" markerEnd="url(#arrowhead-green)" />
          <text x="628" y="108" textAnchor="middle" fill={greenColor} fontSize="7" fontFamily="monospace">match</text>

          {/* IR Builder -> Optimizer */}
          <line x1="730" y1="115" x2="758" y2="115" stroke={strokeColor} strokeWidth="1.5" markerEnd="url(#arrowhead)" />
          <text x="744" y="108" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace" fontStyle="italic">IR</text>

          {/* Optimizer -> Templates */}
          <line x1="840" y1="115" x2="868" y2="115" stroke={strokeColor} strokeWidth="1.5" markerEnd="url(#arrowhead)" />

          {/* Memory Analysis -> Optimizer (feedback) */}
          <line x1="410" y1="170" x2="410" y2="280" stroke={mutedColor} strokeWidth="1" strokeDasharray="4 2" opacity="0.5" />
          <line x1="410" y1="280" x2="800" y2="280" stroke={mutedColor} strokeWidth="1" strokeDasharray="4 2" opacity="0.5" />
          <line x1="800" y1="280" x2="800" y2="142" stroke={mutedColor} strokeWidth="1" strokeDasharray="4 2" opacity="0.5" markerEnd="url(#arrowhead)" />
          <text x="600" y="292" textAnchor="middle" fill={mutedColor} fontSize="7" fontFamily="monospace" fontStyle="italic" opacity="0.7">tile optimization hints</text>

          {/* Templates -> Diagnostics */}
          <line x1="915" y1="140" x2="915" y2="168" stroke={strokeColor} strokeWidth="1.5" markerEnd="url(#arrowhead)" />

          {/* Templates -> Output */}
          <line x1="960" y1="115" x2="988" y2="115" stroke={greenColor} strokeWidth="1.5" markerEnd="url(#arrowhead-green)" />

          {/* Diagnostics -> Output (validation) */}
          <line x1="960" y1="192" x2="1025" y2="192" stroke={strokeColor} strokeWidth="1" strokeDasharray="3 2" />
          <line x1="1025" y1="192" x2="1025" y2="177" stroke={strokeColor} strokeWidth="1" strokeDasharray="3 2" markerEnd="url(#arrowhead)" />
          <text x="995" y="186" textAnchor="middle" fill={mutedColor} fontSize="6" fontFamily="monospace">validate</text>

        </svg>
      </div>

      {/* Tooltip */}
      {hoveredNode && TOOLTIPS[hoveredNode] && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: 'translateX(-50%)',
          }}
        >
          <div
            className="px-3 py-2 text-xs font-mono border shadow-lg max-w-[280px]"
            style={{
              backgroundColor: isDark ? '#1a1a1a' : '#fff',
              borderColor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.15)',
              color: textColor,
            }}
          >
            <div className="font-semibold mb-1" style={{ color: greenColor }}>
              {TOOLTIPS[hoveredNode].title}
            </div>
            <p style={{ color: mutedColor, lineHeight: 1.4 }}>
              {TOOLTIPS[hoveredNode].description}
            </p>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 flex items-center justify-between">
        <span className="text-[10px] font-mono" style={{ color: mutedColor }}>
          Hover over nodes for details
        </span>
        <span className="text-[10px] font-mono" style={{ color: mutedColor }}>
          CUDA SIMT → Semantic Analysis → Pattern Detection → Optimized cuTile
        </span>
      </div>
    </div>
  );
}
