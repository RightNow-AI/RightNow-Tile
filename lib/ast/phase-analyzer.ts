// Phase Analyzer - Detects multi-phase computation patterns in CUDA kernels
// Critical for complex kernels like Flash Attention, FFT, and fused operations

import { CudaKernelInfo, KernelArchetype, LoopInfo, MemoryAccess, SyncPoint } from './types';

export interface KernelPhase {
  id: number;
  type: PhaseType;
  startLine: number;
  endLine: number;
  pattern: KernelArchetype | null;  // What pattern this phase matches
  inputs: string[];                  // Arrays read in this phase
  outputs: string[];                 // Arrays written in this phase
  dependencies: number[];            // Phase IDs this depends on
  syncBefore: boolean;               // Has __syncthreads before
  syncAfter: boolean;                // Has __syncthreads after
  description: string;               // Human-readable description
}

export type PhaseType =
  | 'load'           // Data loading from global to shared/registers
  | 'compute'        // General computation
  | 'matmul'         // Matrix multiplication
  | 'reduce'         // Reduction operation
  | 'softmax'        // Softmax computation (exp, sum, normalize)
  | 'elementwise'    // Element-wise operations
  | 'store'          // Store results to global memory
  | 'sync'           // Synchronization barrier
  | 'accumulate'     // Running accumulation (e.g., online softmax)
  | 'transpose'      // Matrix transpose
  | 'fft_butterfly'  // FFT butterfly operation
  | 'unknown';

export interface MultiPhaseAnalysis {
  phases: KernelPhase[];
  isFusedKernel: boolean;
  isMultiPhaseKernel: boolean;
  dominantPattern: KernelArchetype;
  fusedPatterns: KernelArchetype[];  // e.g., ['gemm', 'reduction', 'gemm'] for attention
  phaseCount: number;
  hasOnlineSoftmax: boolean;         // Key indicator for Flash Attention
  hasBlockwiseIteration: boolean;    // Iterates over blocks of K/V
  confidence: number;
}

export class PhaseAnalyzer {
  /**
   * Analyze kernel to detect multi-phase computation patterns
   */
  analyze(kernel: CudaKernelInfo): MultiPhaseAnalysis {
    const phases: KernelPhase[] = [];
    const body = kernel.sourceText;
    const lines = body.split('\n');

    // Step 1: Find synchronization points as phase boundaries
    const syncIndices = this.findSyncBoundaries(kernel.syncPoints, lines);

    // Step 2: Segment code by sync points
    const segments = this.segmentBySync(body, syncIndices);

    // Step 3: Analyze each segment
    let phaseId = 0;
    for (const segment of segments) {
      const phase = this.analyzeSegment(segment, phaseId++, kernel);
      if (phase) {
        phases.push(phase);
      }
    }

    // Step 4: Build dependency graph
    this.buildDependencies(phases);

    // Step 5: Detect fused patterns
    const fusedPatterns = this.detectFusedPatterns(phases);
    const dominantPattern = this.determineDominantPattern(phases, fusedPatterns);

    // Step 6: Check for attention-specific patterns
    const hasOnlineSoftmax = this.detectOnlineSoftmax(kernel);
    const hasBlockwiseIteration = this.detectBlockwiseIteration(kernel);

    const isFusedKernel = fusedPatterns.length > 1;
    const isMultiPhaseKernel = phases.length > 2;

    return {
      phases,
      isFusedKernel,
      isMultiPhaseKernel,
      dominantPattern,
      fusedPatterns,
      phaseCount: phases.length,
      hasOnlineSoftmax,
      hasBlockwiseIteration,
      confidence: this.calculateConfidence(phases, fusedPatterns, hasOnlineSoftmax),
    };
  }

  private findSyncBoundaries(syncPoints: SyncPoint[], lines: string[]): number[] {
    const boundaries: number[] = [0]; // Start of kernel

    for (const sync of syncPoints) {
      if (sync.type === 'syncthreads') {
        boundaries.push(sync.line);
      }
    }

    boundaries.push(lines.length); // End of kernel
    return [...new Set(boundaries)].sort((a, b) => a - b);
  }

  private segmentBySync(body: string, syncIndices: number[]): Array<{
    startLine: number;
    endLine: number;
    code: string;
  }> {
    const lines = body.split('\n');
    const segments: Array<{ startLine: number; endLine: number; code: string }> = [];

    for (let i = 0; i < syncIndices.length - 1; i++) {
      const start = syncIndices[i];
      const end = syncIndices[i + 1];
      const code = lines.slice(start, end).join('\n');

      if (code.trim()) {
        segments.push({
          startLine: start + 1,
          endLine: end,
          code,
        });
      }
    }

    return segments;
  }

  private analyzeSegment(
    segment: { startLine: number; endLine: number; code: string },
    phaseId: number,
    kernel: CudaKernelInfo
  ): KernelPhase | null {
    const code = segment.code;

    // Detect phase type based on code patterns
    const type = this.detectPhaseType(code);

    // Extract inputs and outputs
    const { inputs, outputs } = this.extractDataflow(code, kernel);

    // Detect pattern for this phase
    const pattern = this.detectPhasePattern(code, type);

    // Check for sync barriers
    const syncBefore = /__syncthreads/.test(code.split('\n')[0] || '');
    const syncAfter = /__syncthreads/.test(code.split('\n').slice(-1)[0] || '');

    return {
      id: phaseId,
      type,
      startLine: segment.startLine,
      endLine: segment.endLine,
      pattern,
      inputs,
      outputs,
      dependencies: [],
      syncBefore,
      syncAfter,
      description: this.generatePhaseDescription(type, pattern, inputs, outputs),
    };
  }

  private detectPhaseType(code: string): PhaseType {
    // Check for specific patterns in order of specificity

    // FFT butterfly pattern
    if (/twiddle|butterfly|__sincosf|__cosf.*__sinf/i.test(code) ||
        /W_r\s*=|W_i\s*=/.test(code)) {
      return 'fft_butterfly';
    }

    // Softmax pattern: exp + sum + divide
    if (/expf?\s*\(/.test(code) && /\//.test(code)) {
      const hasMax = /fmaxf?|max\s*\(/.test(code);
      const hasSum = /\+=/.test(code) || /sum/.test(code);
      if (hasMax || hasSum) {
        return 'softmax';
      }
    }

    // Matrix multiplication pattern
    if (this.hasMatmulPattern(code)) {
      return 'matmul';
    }

    // Reduction pattern
    if (this.hasReductionPattern(code)) {
      return 'reduce';
    }

    // Transpose pattern
    if (/\[.*\]\s*\[.*\]/.test(code) &&
        /\[\s*\w+\s*\]\s*\[\s*\w+\s*\].*=.*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/.test(code)) {
      // Check if indices are swapped
      return 'transpose';
    }

    // Accumulate pattern (running sum/max)
    if (/\+=|acc|accum/i.test(code) && !/for|while/.test(code)) {
      return 'accumulate';
    }

    // Load pattern
    if (this.isLoadPhase(code)) {
      return 'load';
    }

    // Store pattern
    if (this.isStorePhase(code)) {
      return 'store';
    }

    // Sync-only
    if (/^\s*__syncthreads\s*\(\s*\)\s*;?\s*$/.test(code.trim())) {
      return 'sync';
    }

    // General elementwise
    if (/[\+\-\*\/]/.test(code) && !/for|while/.test(code)) {
      return 'elementwise';
    }

    return 'compute';
  }

  private hasMatmulPattern(code: string): boolean {
    // Check for nested loops with accumulation
    const hasNestedLoop = /for.*for/.test(code.replace(/\n/g, ' '));
    const hasAccumulation = /\+=/.test(code);
    const hasMultiply = /\*/.test(code);

    // Check for explicit matmul-like pattern
    const hasMatmulAccess = /\[\s*\w+\s*\]\s*\[\s*\w+\s*\].*\*.*\[\s*\w+\s*\]\s*\[\s*\w+\s*\]/.test(code) ||
                           /\[\s*\w+\s*\+\s*\w+\s*\].*\*.*\[\s*\w+\s*\+\s*\w+\s*\]/.test(code);

    return (hasNestedLoop && hasAccumulation && hasMultiply) || hasMatmulAccess;
  }

  private hasReductionPattern(code: string): boolean {
    // Tree reduction with stride halving
    const hasStrideHalving = />>=\s*1|\/=\s*2/.test(code);
    const hasConditionalAccum = /if.*threadIdx.*\+=/.test(code.replace(/\n/g, ' '));

    // Warp shuffle reduction
    const hasWarpShuffle = /__shfl_down_sync|__shfl_xor_sync/.test(code);

    return hasStrideHalving || hasConditionalAccum || hasWarpShuffle;
  }

  private isLoadPhase(code: string): boolean {
    // Primarily reading from global memory to shared/registers
    const lines = code.split('\n');
    let loadCount = 0;
    let storeCount = 0;

    for (const line of lines) {
      if (/=\s*\w+\s*\[/.test(line) && !/\w+\s*\[\s*.*\s*\]\s*=/.test(line)) {
        loadCount++;
      }
      if (/\w+\s*\[\s*.*\s*\]\s*=/.test(line)) {
        storeCount++;
      }
    }

    return loadCount > storeCount && loadCount > 0;
  }

  private isStorePhase(code: string): boolean {
    // Primarily writing to global memory
    const lines = code.split('\n');
    let storeCount = 0;

    for (const line of lines) {
      if (/\w+\s*\[\s*.*\s*\]\s*=/.test(line) && !/=\s*\w+\s*\[/.test(line.split('=')[1] || '')) {
        storeCount++;
      }
    }

    return storeCount > 0 && !/for|while/.test(code);
  }

  private extractDataflow(code: string, kernel: CudaKernelInfo): {
    inputs: string[];
    outputs: string[];
  } {
    const inputs = new Set<string>();
    const outputs = new Set<string>();

    // Find all array accesses in this segment
    const accessRegex = /(\w+)\s*\[/g;
    let match;

    const lines = code.split('\n');
    for (const line of lines) {
      // Reset regex
      accessRegex.lastIndex = 0;

      while ((match = accessRegex.exec(line)) !== null) {
        const array = match[1];
        const afterAccess = line.substring(match.index + match[0].length);
        const beforeAccess = line.substring(0, match.index);

        // Skip if it's a declaration
        if (/(?:float|double|int|half|__shared__|__device__|const)\s*$/.test(beforeAccess.trim())) {
          continue;
        }

        // Check if write or read
        const fullAccessMatch = line.substring(match.index).match(/^\w+\s*\[[^\]]+\]/);
        if (fullAccessMatch) {
          const afterFull = line.substring(match.index + fullAccessMatch[0].length);
          if (/^\s*=(?!=)/.test(afterFull) || /^\s*[\+\-\*\/\&\|\^]=/.test(afterFull)) {
            outputs.add(array);
          }
          // Also check if it's read (could be both read and write for +=)
          if (!/^\s*=(?!=)/.test(afterFull) || /^\s*[\+\-\*\/\&\|\^]=/.test(afterFull)) {
            inputs.add(array);
          }
        }
      }
    }

    // Also check for shared memory as intermediate data
    for (const shared of kernel.sharedMemoryDecls) {
      if (code.includes(shared)) {
        if (code.includes(`${shared}[`) && /=\s*\w+/.test(code)) {
          // Being written to
          outputs.add(shared);
        }
        if (new RegExp(`=.*${shared}\\s*\\[`).test(code)) {
          // Being read from
          inputs.add(shared);
        }
      }
    }

    return {
      inputs: Array.from(inputs),
      outputs: Array.from(outputs),
    };
  }

  private detectPhasePattern(code: string, type: PhaseType): KernelArchetype | null {
    switch (type) {
      case 'matmul':
        return 'gemm';
      case 'reduce':
      case 'softmax':
        return 'reduction';
      case 'fft_butterfly':
        return 'fft';
      case 'elementwise':
        return 'elementwise';
      default:
        return null;
    }
  }

  private buildDependencies(phases: KernelPhase[]): void {
    for (let i = 0; i < phases.length; i++) {
      const currentPhase = phases[i];

      // Look at previous phases for dependencies
      for (let j = 0; j < i; j++) {
        const prevPhase = phases[j];

        // Check if current phase reads what previous phase wrote
        const hasDataDep = currentPhase.inputs.some(input =>
          prevPhase.outputs.includes(input)
        );

        if (hasDataDep) {
          currentPhase.dependencies.push(prevPhase.id);
        }
      }

      // If no explicit dependencies but previous phase exists, assume sequential dep
      if (currentPhase.dependencies.length === 0 && i > 0) {
        currentPhase.dependencies.push(phases[i - 1].id);
      }
    }
  }

  private detectFusedPatterns(phases: KernelPhase[]): KernelArchetype[] {
    const patterns: KernelArchetype[] = [];
    const seen = new Set<KernelArchetype>();

    for (const phase of phases) {
      if (phase.pattern && !seen.has(phase.pattern)) {
        patterns.push(phase.pattern);
        seen.add(phase.pattern);
      }
    }

    return patterns;
  }

  private determineDominantPattern(
    phases: KernelPhase[],
    fusedPatterns: KernelArchetype[]
  ): KernelArchetype {
    // Check for attention pattern first (has specific structure)
    const hasMatmul = phases.some(p => p.type === 'matmul');
    const hasSoftmax = phases.some(p => p.type === 'softmax');
    const hasAccumulate = phases.some(p => p.type === 'accumulate');

    if (hasMatmul && hasSoftmax) {
      return 'attention';
    }

    // Check for FFT
    if (phases.some(p => p.type === 'fft_butterfly')) {
      return 'fft';
    }

    // If multiple patterns, it's a fused kernel
    if (fusedPatterns.length > 1) {
      return 'fused';
    }

    // Return the most common pattern
    const patternCounts = new Map<KernelArchetype, number>();
    for (const phase of phases) {
      if (phase.pattern) {
        patternCounts.set(phase.pattern, (patternCounts.get(phase.pattern) || 0) + 1);
      }
    }

    let maxPattern: KernelArchetype = 'elementwise';
    let maxCount = 0;
    for (const [pattern, count] of patternCounts) {
      if (count > maxCount) {
        maxCount = count;
        maxPattern = pattern;
      }
    }

    return maxPattern;
  }

  private detectOnlineSoftmax(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Online softmax maintains running max and sum
    // Pattern: m_new = max(m_old, m_curr), then rescale
    const hasRunningMax = /m_\w*\s*=\s*(?:fmaxf?|max)\s*\(/.test(source) ||
                          /max_\w*\s*=\s*(?:fmaxf?|max)\s*\(/.test(source);

    const hasRescale = /\*\s*expf?\s*\(\s*\w+\s*-\s*\w+\s*\)/.test(source) ||
                       /expf?\s*\(\s*\w+\s*-\s*m/.test(source);

    const hasRunningSum = /l_\w*\s*[+*]=/.test(source) ||
                          /sum_\w*\s*[+*]=/.test(source);

    return hasRunningMax && (hasRescale || hasRunningSum);
  }

  private detectBlockwiseIteration(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Block-wise iteration typically has:
    // - Outer loop over blocks
    // - Block-sized loads
    // - Synchronization inside loop
    const hasBlockLoop = /for\s*\([^)]*block|for\s*\([^)]*BLOCK/i.test(source);
    const hasSyncInLoop = kernel.loops.some(loop => loop.containsSyncthreads);
    const hasBlockSizedAccess = /\[\s*\w+\s*\*\s*(?:BLOCK|block)/i.test(source);

    return (hasBlockLoop || hasSyncInLoop) && hasBlockSizedAccess;
  }

  private calculateConfidence(
    phases: KernelPhase[],
    fusedPatterns: KernelArchetype[],
    hasOnlineSoftmax: boolean
  ): number {
    let confidence = 0.5; // Base confidence

    // More phases = more confidence in multi-phase analysis
    if (phases.length >= 3) confidence += 0.1;
    if (phases.length >= 5) confidence += 0.1;

    // Detected fused patterns increases confidence
    if (fusedPatterns.length >= 2) confidence += 0.15;

    // Online softmax is a strong indicator
    if (hasOnlineSoftmax) confidence += 0.15;

    // All phases have patterns = higher confidence
    const phasesWithPatterns = phases.filter(p => p.pattern !== null).length;
    confidence += (phasesWithPatterns / Math.max(phases.length, 1)) * 0.1;

    return Math.min(confidence, 1.0);
  }

  private generatePhaseDescription(
    type: PhaseType,
    pattern: KernelArchetype | null,
    inputs: string[],
    outputs: string[]
  ): string {
    const patternStr = pattern ? ` (${pattern} pattern)` : '';
    const inputStr = inputs.length > 0 ? ` from [${inputs.join(', ')}]` : '';
    const outputStr = outputs.length > 0 ? ` to [${outputs.join(', ')}]` : '';

    const descriptions: Record<PhaseType, string> = {
      'load': `Load data${inputStr}${outputStr}`,
      'compute': `General computation${patternStr}${inputStr}${outputStr}`,
      'matmul': `Matrix multiplication${inputStr}${outputStr}`,
      'reduce': `Reduction${inputStr}${outputStr}`,
      'softmax': `Softmax computation${inputStr}${outputStr}`,
      'elementwise': `Element-wise operations${inputStr}${outputStr}`,
      'store': `Store results${outputStr}`,
      'sync': 'Thread synchronization barrier',
      'accumulate': `Running accumulation${inputStr}${outputStr}`,
      'transpose': `Matrix transpose${inputStr}${outputStr}`,
      'fft_butterfly': `FFT butterfly operation${inputStr}${outputStr}`,
      'unknown': `Unknown operation${inputStr}${outputStr}`,
    };

    return descriptions[type];
  }
}

export const phaseAnalyzer = new PhaseAnalyzer();
