/**
 * Enhanced CUDA Parser
 * Provides deeper analysis of CUDA source code beyond basic regex extraction
 */

import {
  ALL_INTRINSICS,
  IntrinsicInfo,
  PatternHint,
  buildIntrinsicsRegex,
  getIntrinsicInfo,
} from './intrinsics';

// Types for parsed CUDA constructs
export interface IntrinsicCall {
  name: string;
  info: IntrinsicInfo;
  args: string;
  line: number;
  patternHints: PatternHint[];
}

export interface IndexPattern {
  array: string;
  indexExpr: string;
  classification: 'linear' | '2d' | 'strided' | 'indirect' | 'complex';
  dependsOnThreadIdx: boolean;
  dependsOnBlockIdx: boolean;
  hasNeighborOffset: boolean;
  offsets: number[];
}

export interface WarpOperation {
  type: 'shuffle' | 'vote' | 'reduce' | 'sync';
  intrinsic: string;
  mask?: string;
  delta?: number;
}

export interface LoopAnalysis {
  variable: string;
  init: string;
  condition: string;
  update: string;
  hasStrideHalving: boolean;
  hasStrideDoubling: boolean;
  hasSyncInside: boolean;
  nestLevel: number;
  isReductionLoop: boolean;
  isScanLoop: boolean;
}

export interface EnhancedParseResult {
  intrinsicCalls: IntrinsicCall[];
  indexPatterns: IndexPattern[];
  warpOperations: WarpOperation[];
  loopAnalysis: LoopAnalysis[];
  reductionPatterns: ReductionPattern[];
  accumulationPatterns: AccumulationPattern[];
  threadIndexUsage: ThreadIndexUsage;
  sharedMemoryUsage: SharedMemoryUsage;
  patternSignals: PatternSignal[];
}

export interface ReductionPattern {
  variable: string;
  operation: 'sum' | 'max' | 'min' | 'prod' | 'and' | 'or' | 'xor';
  isWarpLevel: boolean;
  isBlockLevel: boolean;
  hasAtomic: boolean;
}

export interface AccumulationPattern {
  variable: string;
  expression: string;
  operation: 'add' | 'mul' | 'max' | 'min' | 'fma';
  inLoop: boolean;
}

export interface ThreadIndexUsage {
  usesThreadIdxX: boolean;
  usesThreadIdxY: boolean;
  usesThreadIdxZ: boolean;
  usesBlockIdxX: boolean;
  usesBlockIdxY: boolean;
  usesBlockIdxZ: boolean;
  usesBlockDimX: boolean;
  usesBlockDimY: boolean;
  usesBlockDimZ: boolean;
  globalIdExpressions: string[];
  is1D: boolean;
  is2D: boolean;
  is3D: boolean;
}

export interface SharedMemoryUsage {
  declarations: SharedMemoryDecl[];
  totalStaticBytes: number;
  hasDynamicAllocation: boolean;
  accessPatterns: SharedMemoryAccess[];
}

export interface SharedMemoryDecl {
  name: string;
  type: string;
  size: string | number;
  isDynamic: boolean;
}

export interface SharedMemoryAccess {
  array: string;
  indexExpr: string;
  hasBankConflictRisk: boolean;
}

export interface PatternSignal {
  pattern: PatternHint;
  signal: string;
  weight: number;
  source: string;
}

export class EnhancedCudaParser {
  private source: string = '';
  private lines: string[] = [];
  private intrinsicsRegex: RegExp;

  constructor() {
    this.intrinsicsRegex = buildIntrinsicsRegex();
  }

  /**
   * Parse CUDA source code for enhanced analysis
   */
  parse(source: string): EnhancedParseResult {
    this.source = source;
    this.lines = source.split('\n');

    return {
      intrinsicCalls: this.extractIntrinsics(),
      indexPatterns: this.analyzeIndexPatterns(),
      warpOperations: this.detectWarpOperations(),
      loopAnalysis: this.analyzeLoops(),
      reductionPatterns: this.detectReductionPatterns(),
      accumulationPatterns: this.detectAccumulationPatterns(),
      threadIndexUsage: this.analyzeThreadIndexUsage(),
      sharedMemoryUsage: this.analyzeSharedMemory(),
      patternSignals: this.gatherPatternSignals(),
    };
  }

  /**
   * Extract all intrinsic function calls
   */
  private extractIntrinsics(): IntrinsicCall[] {
    const calls: IntrinsicCall[] = [];
    const regex = new RegExp(this.intrinsicsRegex.source, 'g');

    let match;
    while ((match = regex.exec(this.source)) !== null) {
      const name = match[1];
      const info = getIntrinsicInfo(name);
      if (info) {
        // Find line number
        const line = this.source.substring(0, match.index).split('\n').length;
        // Extract arguments (simple extraction)
        const argsStart = match.index + match[0].length - 1;
        const args = this.extractBalancedParens(this.source, argsStart);

        calls.push({
          name,
          info,
          args,
          line,
          patternHints: info.patternHints,
        });
      }
    }

    return calls;
  }

  /**
   * Extract content within balanced parentheses
   */
  private extractBalancedParens(source: string, start: number): string {
    let depth = 0;
    let result = '';
    for (let i = start; i < source.length; i++) {
      const char = source[i];
      if (char === '(') depth++;
      else if (char === ')') {
        depth--;
        if (depth === 0) break;
      }
      if (depth > 0 && (depth > 1 || char !== '(')) {
        result += char;
      }
    }
    return result.trim();
  }

  /**
   * Analyze array index patterns
   */
  private analyzeIndexPatterns(): IndexPattern[] {
    const patterns: IndexPattern[] = [];
    // Match array access: identifier[expression]
    const arrayAccessRegex = /(\w+)\s*\[([^\[\]]+(?:\[[^\[\]]*\][^\[\]]*)*)\]/g;

    let match;
    while ((match = arrayAccessRegex.exec(this.source)) !== null) {
      const array = match[1];
      const indexExpr = match[2];

      // Skip common non-array identifiers
      if (['if', 'for', 'while', 'switch'].includes(array)) continue;

      patterns.push({
        array,
        indexExpr,
        classification: this.classifyIndexPattern(indexExpr),
        dependsOnThreadIdx: /threadIdx\s*\.\s*[xyz]/.test(indexExpr),
        dependsOnBlockIdx: /blockIdx\s*\.\s*[xyz]/.test(indexExpr),
        hasNeighborOffset: this.hasNeighborOffset(indexExpr),
        offsets: this.extractOffsets(indexExpr),
      });
    }

    return patterns;
  }

  /**
   * Classify the index pattern type
   */
  private classifyIndexPattern(expr: string): IndexPattern['classification'] {
    // Check for indirect access (array[array[x]])
    if (/\w+\s*\[\s*\w+\s*\[/.test(expr)) {
      return 'indirect';
    }

    // Check for 2D indexing (row * width + col or similar)
    if (/\*\s*\w+\s*\+/.test(expr) || /\w+\s*\*\s*\w+/.test(expr)) {
      // Could be 2D or strided
      if (/row|col|width|height|[ij].*\*.*[ij]/i.test(expr)) {
        return '2d';
      }
      // Check for stride pattern
      if (/stride|\*\s*\d+|>>|<</.test(expr)) {
        return 'strided';
      }
      return '2d';
    }

    // Check for strided access
    if (/\*\s*\d+|\*\s*stride|stride\s*\*/i.test(expr)) {
      return 'strided';
    }

    // Simple linear: tid, idx, i, threadIdx.x, etc.
    if (/^[\w\s\.\+\-]+$/.test(expr) && !this.hasNeighborOffset(expr)) {
      return 'linear';
    }

    return 'complex';
  }

  /**
   * Check if index has neighbor offset (stencil pattern)
   */
  private hasNeighborOffset(expr: string): boolean {
    // Look for patterns like: i+1, i-1, idx+width, idx-1
    return /[\w]+\s*[\+\-]\s*(?:\d+|\w+)/.test(expr);
  }

  /**
   * Extract numeric offsets from index expression
   */
  private extractOffsets(expr: string): number[] {
    const offsets: number[] = [];
    const offsetRegex = /[\+\-]\s*(\d+)/g;
    let match;
    while ((match = offsetRegex.exec(expr)) !== null) {
      const sign = expr[match.index] === '-' ? -1 : 1;
      offsets.push(sign * parseInt(match[1]));
    }
    return offsets;
  }

  /**
   * Detect warp-level operations
   */
  private detectWarpOperations(): WarpOperation[] {
    const operations: WarpOperation[] = [];

    // Warp shuffle
    const shuffleRegex = /__shfl(?:_(?:down|up|xor))?(?:_sync)?\s*\(/g;
    let match;
    while ((match = shuffleRegex.exec(this.source)) !== null) {
      const name = match[0].replace(/\s*\($/, '');
      operations.push({
        type: 'shuffle',
        intrinsic: name,
        delta: this.extractShuffleDelta(this.source, match.index),
      });
    }

    // Warp vote
    const voteRegex = /__(?:ballot|all|any)(?:_sync)?\s*\(/g;
    while ((match = voteRegex.exec(this.source)) !== null) {
      operations.push({
        type: 'vote',
        intrinsic: match[0].replace(/\s*\($/, ''),
      });
    }

    // Warp reduce
    const reduceRegex = /__reduce_(?:add|max|min|and|or|xor)_sync\s*\(/g;
    while ((match = reduceRegex.exec(this.source)) !== null) {
      operations.push({
        type: 'reduce',
        intrinsic: match[0].replace(/\s*\($/, ''),
      });
    }

    // Warp sync
    if (/__syncwarp/.test(this.source)) {
      operations.push({
        type: 'sync',
        intrinsic: '__syncwarp',
      });
    }

    return operations;
  }

  /**
   * Extract shuffle delta (offset) value
   */
  private extractShuffleDelta(source: string, pos: number): number | undefined {
    // Find the arguments
    const argsMatch = source.substring(pos).match(/\([^)]+\)/);
    if (!argsMatch) return undefined;

    const args = argsMatch[0];
    // For __shfl_down_sync(mask, val, delta) - delta is 3rd arg
    // For __shfl_down(val, delta) - delta is 2nd arg
    const parts = args.split(',');
    if (parts.length >= 2) {
      const deltaArg = parts[parts.length - 1].replace(/[()]/g, '').trim();
      const num = parseInt(deltaArg);
      if (!isNaN(num)) return num;
    }
    return undefined;
  }

  /**
   * Analyze loop structures
   */
  private analyzeLoops(): LoopAnalysis[] {
    const loops: LoopAnalysis[] = [];
    // Match for loops
    const forRegex = /for\s*\(\s*([^;]*);([^;]*);([^)]*)\)\s*\{/g;

    let match;
    while ((match = forRegex.exec(this.source)) !== null) {
      const init = match[1].trim();
      const condition = match[2].trim();
      const update = match[3].trim();

      // Extract loop variable
      const varMatch = init.match(/(?:int|unsigned|size_t)?\s*(\w+)\s*=/);
      const variable = varMatch ? varMatch[1] : '';

      // Find loop body
      const bodyStart = match.index + match[0].length;
      const bodyEnd = this.findMatchingBrace(this.source, bodyStart - 1);
      const body = this.source.substring(bodyStart, bodyEnd);

      loops.push({
        variable,
        init,
        condition,
        update,
        hasStrideHalving: />>=\s*1|\/=\s*2|=\s*\w+\s*\/\s*2/.test(update),
        hasStrideDoubling: /<<=\s*1|\*=\s*2|=\s*\w+\s*\*\s*2/.test(update),
        hasSyncInside: /__syncthreads|__syncwarp/.test(body),
        nestLevel: this.countNestLevel(this.source, match.index),
        isReductionLoop: this.isReductionLoop(body, variable),
        isScanLoop: this.isScanLoop(update, body),
      });
    }

    return loops;
  }

  /**
   * Find matching closing brace
   */
  private findMatchingBrace(source: string, start: number): number {
    let depth = 0;
    for (let i = start; i < source.length; i++) {
      if (source[i] === '{') depth++;
      else if (source[i] === '}') {
        depth--;
        if (depth === 0) return i;
      }
    }
    return source.length;
  }

  /**
   * Count loop nesting level
   */
  private countNestLevel(source: string, pos: number): number {
    const before = source.substring(0, pos);
    const opens = (before.match(/\bfor\s*\(/g) || []).length;
    const closes = (before.match(/\}/g) || []).length;
    return Math.max(0, opens - closes);
  }

  /**
   * Check if loop body indicates reduction
   */
  private isReductionLoop(body: string, loopVar: string): boolean {
    // Look for reduction patterns in loop body
    // arr[tid] += arr[tid + stride], arr[tid] = arr[tid] op arr[tid + stride]
    const reductionPatterns = [
      new RegExp(`\\w+\\s*\\[.*\\]\\s*\\+=\\s*\\w+\\s*\\[.*\\+.*${loopVar}`, 'i'),
      new RegExp(`\\w+\\s*\\[.*\\]\\s*=\\s*\\w+\\s*\\[.*\\]\\s*\\+\\s*\\w+\\s*\\[.*${loopVar}`, 'i'),
      /\w+\s*\+=\s*\w+\s*\[/,
      /\w+\s*=\s*max\s*\(\s*\w+\s*,/,
      /\w+\s*=\s*min\s*\(\s*\w+\s*,/,
    ];
    return reductionPatterns.some((p) => p.test(body));
  }

  /**
   * Check if loop is scan pattern (both doubling and halving phases)
   */
  private isScanLoop(update: string, body: string): boolean {
    // Scan typically has complex index arithmetic
    const hasComplexIndex = /\(\s*\w+\s*\+\s*1\s*\)\s*\*\s*\w+\s*\*\s*2/.test(body);
    const hasPowerOf2Ops = (/<</.test(body) || />>/.test(body)) && (/<<=/.test(update) || />>=/.test(update));
    return hasComplexIndex || hasPowerOf2Ops;
  }

  /**
   * Detect reduction patterns
   */
  private detectReductionPatterns(): ReductionPattern[] {
    const patterns: ReductionPattern[] = [];

    // Detect accumulation with atomics
    const atomicPatterns = [
      { regex: /atomicAdd\s*\(\s*(\w+)/, operation: 'sum' as const },
      { regex: /atomicMax\s*\(\s*(\w+)/, operation: 'max' as const },
      { regex: /atomicMin\s*\(\s*(\w+)/, operation: 'min' as const },
      { regex: /atomicAnd\s*\(\s*(\w+)/, operation: 'and' as const },
      { regex: /atomicOr\s*\(\s*(\w+)/, operation: 'or' as const },
      { regex: /atomicXor\s*\(\s*(\w+)/, operation: 'xor' as const },
    ];

    for (const { regex, operation } of atomicPatterns) {
      const match = this.source.match(regex);
      if (match) {
        patterns.push({
          variable: match[1],
          operation,
          isWarpLevel: false,
          isBlockLevel: true,
          hasAtomic: true,
        });
      }
    }

    // Detect warp-level reductions
    if (/__shfl_down|__shfl_xor|__reduce_\w+_sync/.test(this.source)) {
      // Try to find the operation from context
      const sumMatch = /(\w+)\s*[\+\=]\s*__shfl/.test(this.source);
      patterns.push({
        variable: 'warp_result',
        operation: sumMatch ? 'sum' : 'sum',
        isWarpLevel: true,
        isBlockLevel: false,
        hasAtomic: false,
      });
    }

    return patterns;
  }

  /**
   * Detect accumulation patterns (important for GEMM)
   */
  private detectAccumulationPatterns(): AccumulationPattern[] {
    const patterns: AccumulationPattern[] = [];

    // sum += a * b (GEMM pattern)
    const fmaRegex = /(\w+)\s*\+=\s*([^;]+\*[^;]+)/g;
    let match: RegExpExecArray | null;
    while ((match = fmaRegex.exec(this.source)) !== null) {
      patterns.push({
        variable: match[1],
        expression: match[2].trim(),
        operation: 'fma',
        inLoop: this.isInsideLoop(match.index),
      });
    }

    // sum += val
    const addRegex = /(\w+)\s*\+=\s*(\w+)\s*;/g;
    let addMatch: RegExpExecArray | null;
    while ((addMatch = addRegex.exec(this.source)) !== null) {
      // Skip if already matched as FMA
      if (!patterns.some((p) => p.variable === addMatch![1] && p.operation === 'fma')) {
        patterns.push({
          variable: addMatch[1],
          expression: addMatch[2],
          operation: 'add',
          inLoop: this.isInsideLoop(addMatch.index),
        });
      }
    }

    // max = max(max, val)
    const maxRegex = /(\w+)\s*=\s*max\s*\(\s*\1\s*,\s*([^)]+)\)/g;
    while ((match = maxRegex.exec(this.source)) !== null) {
      patterns.push({
        variable: match[1],
        expression: match[2].trim(),
        operation: 'max',
        inLoop: this.isInsideLoop(match.index),
      });
    }

    // min = min(min, val)
    const minRegex = /(\w+)\s*=\s*min\s*\(\s*\1\s*,\s*([^)]+)\)/g;
    while ((match = minRegex.exec(this.source)) !== null) {
      patterns.push({
        variable: match[1],
        expression: match[2].trim(),
        operation: 'min',
        inLoop: this.isInsideLoop(match.index),
      });
    }

    return patterns;
  }

  /**
   * Check if position is inside a loop
   */
  private isInsideLoop(pos: number): boolean {
    const before = this.source.substring(0, pos);
    const forOpens = (before.match(/\bfor\s*\([^)]*\)\s*\{/g) || []).length;
    const whileOpens = (before.match(/\bwhile\s*\([^)]*\)\s*\{/g) || []).length;
    const closes = (before.match(/\}/g) || []).length;
    return forOpens + whileOpens > closes;
  }

  /**
   * Analyze thread index usage
   */
  private analyzeThreadIndexUsage(): ThreadIndexUsage {
    const usage = {
      usesThreadIdxX: /threadIdx\s*\.\s*x/.test(this.source),
      usesThreadIdxY: /threadIdx\s*\.\s*y/.test(this.source),
      usesThreadIdxZ: /threadIdx\s*\.\s*z/.test(this.source),
      usesBlockIdxX: /blockIdx\s*\.\s*x/.test(this.source),
      usesBlockIdxY: /blockIdx\s*\.\s*y/.test(this.source),
      usesBlockIdxZ: /blockIdx\s*\.\s*z/.test(this.source),
      usesBlockDimX: /blockDim\s*\.\s*x/.test(this.source),
      usesBlockDimY: /blockDim\s*\.\s*y/.test(this.source),
      usesBlockDimZ: /blockDim\s*\.\s*z/.test(this.source),
      globalIdExpressions: [] as string[],
      is1D: false,
      is2D: false,
      is3D: false,
    };

    // Extract global ID expressions
    const globalIdRegex = /(?:int|unsigned|size_t)\s+(\w+)\s*=\s*(threadIdx[^;]+blockIdx[^;]+|blockIdx[^;]+threadIdx[^;]+)/g;
    let match;
    while ((match = globalIdRegex.exec(this.source)) !== null) {
      usage.globalIdExpressions.push(match[2].trim());
    }

    // Determine dimensionality
    const usesX = usage.usesThreadIdxX || usage.usesBlockIdxX;
    const usesY = usage.usesThreadIdxY || usage.usesBlockIdxY;
    const usesZ = usage.usesThreadIdxZ || usage.usesBlockIdxZ;

    usage.is3D = usesX && usesY && usesZ;
    usage.is2D = usesX && usesY && !usesZ;
    usage.is1D = usesX && !usesY && !usesZ;

    return usage;
  }

  /**
   * Analyze shared memory usage
   */
  private analyzeSharedMemory(): SharedMemoryUsage {
    const declarations: SharedMemoryDecl[] = [];
    const accessPatterns: SharedMemoryAccess[] = [];

    // Static shared memory: __shared__ type name[size]
    const staticRegex = /__shared__\s+([\w\s]+)\s+(\w+)\s*\[\s*(\d+|\w+)\s*\]/g;
    let match;
    while ((match = staticRegex.exec(this.source)) !== null) {
      const type = match[1].trim();
      const name = match[2];
      const sizeStr = match[3];
      const size = parseInt(sizeStr);

      declarations.push({
        name,
        type,
        size: isNaN(size) ? sizeStr : size,
        isDynamic: false,
      });
    }

    // Dynamic shared memory: extern __shared__ type name[]
    const dynamicRegex = /extern\s+__shared__\s+([\w\s]+)\s+(\w+)\s*\[\s*\]/g;
    while ((match = dynamicRegex.exec(this.source)) !== null) {
      declarations.push({
        name: match[2],
        type: match[1].trim(),
        size: 'dynamic',
        isDynamic: true,
      });
    }

    // Analyze access patterns for bank conflicts
    for (const decl of declarations) {
      const accessRegex = new RegExp(`${decl.name}\\s*\\[([^\\]]+)\\]`, 'g');
      while ((match = accessRegex.exec(this.source)) !== null) {
        const indexExpr = match[1];
        accessPatterns.push({
          array: decl.name,
          indexExpr,
          hasBankConflictRisk: this.hasBankConflictRisk(indexExpr),
        });
      }
    }

    // Calculate total static bytes
    let totalStaticBytes = 0;
    for (const decl of declarations) {
      if (!decl.isDynamic && typeof decl.size === 'number') {
        const typeSize = this.getTypeSize(decl.type);
        totalStaticBytes += decl.size * typeSize;
      }
    }

    return {
      declarations,
      totalStaticBytes,
      hasDynamicAllocation: declarations.some((d) => d.isDynamic),
      accessPatterns,
    };
  }

  /**
   * Check for bank conflict risk in shared memory access
   */
  private hasBankConflictRisk(indexExpr: string): boolean {
    // Bank conflicts occur when threads in a warp access different banks with stride
    // Simple heuristic: stride of 1 is good, stride of 32 or powers of 2 may conflict
    const hasStride = /\*\s*(?:32|16|8|4|2)\b/.test(indexExpr);
    const hasStrideVar = /\*\s*stride/.test(indexExpr);
    return hasStride || hasStrideVar;
  }

  /**
   * Get size of a CUDA type in bytes
   */
  private getTypeSize(type: string): number {
    const sizes: Record<string, number> = {
      char: 1,
      'unsigned char': 1,
      short: 2,
      'unsigned short': 2,
      int: 4,
      'unsigned int': 4,
      float: 4,
      double: 8,
      'long long': 8,
      'unsigned long long': 8,
      half: 2,
      __half: 2,
    };
    return sizes[type.toLowerCase()] || 4;
  }

  /**
   * Gather all pattern signals from analysis
   */
  private gatherPatternSignals(): PatternSignal[] {
    const signals: PatternSignal[] = [];

    // Signals from intrinsics
    for (const call of this.extractIntrinsics()) {
      for (const hint of call.patternHints) {
        signals.push({
          pattern: hint,
          signal: `intrinsic_${call.name}`,
          weight: 0.2,
          source: `line ${call.line}`,
        });
      }
    }

    // Signals from loops
    for (const loop of this.analyzeLoops()) {
      if (loop.hasStrideHalving) {
        signals.push({
          pattern: 'reduction',
          signal: 'stride_halving_loop',
          weight: 0.35,
          source: `loop var ${loop.variable}`,
        });
      }
      if (loop.hasStrideDoubling && loop.hasStrideHalving) {
        signals.push({
          pattern: 'scan',
          signal: 'up_down_sweep',
          weight: 0.4,
          source: `loop var ${loop.variable}`,
        });
      }
      if (loop.hasSyncInside) {
        signals.push({
          pattern: 'reduction',
          signal: 'sync_in_loop',
          weight: 0.2,
          source: `loop var ${loop.variable}`,
        });
      }
    }

    // Signals from index patterns
    const neighborCount = this.analyzeIndexPatterns().filter((p) => p.hasNeighborOffset).length;
    if (neighborCount >= 3) {
      signals.push({
        pattern: 'stencil',
        signal: 'multiple_neighbor_access',
        weight: 0.4,
        source: `${neighborCount} neighbor accesses`,
      });
    }

    // Signals from accumulation
    const fmaInLoop = this.detectAccumulationPatterns().filter((p) => p.operation === 'fma' && p.inLoop);
    if (fmaInLoop.length > 0) {
      signals.push({
        pattern: 'gemm',
        signal: 'fma_in_loop',
        weight: 0.25,
        source: `${fmaInLoop.length} FMA operations`,
      });
    }

    // Signals from thread usage
    const threadUsage = this.analyzeThreadIndexUsage();
    if (threadUsage.is2D) {
      signals.push({
        pattern: 'gemm',
        signal: '2d_thread_indexing',
        weight: 0.15,
        source: 'thread index analysis',
      });
      signals.push({
        pattern: 'stencil',
        signal: '2d_thread_indexing',
        weight: 0.1,
        source: 'thread index analysis',
      });
    }

    return signals;
  }
}

// Export singleton instance
export const cudaParser = new EnhancedCudaParser();
