/**
 * Histogram Pattern Matcher
 * Detects histogram computation patterns including atomic and privatized variants
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class HistogramMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Atomic increment to array with computed index - KEY indicator
    const atomicIncrementInfo = this.detectAtomicIncrement(source);
    if (atomicIncrementInfo.found) {
      addEvidence(evidence, 'atomic_increment', 0.35,
        `Atomic increment detected: ${atomicIncrementInfo.type}`);
    }

    // 2. Bin/bucket index calculation
    const binCalcInfo = this.detectBinCalculation(source);
    if (binCalcInfo.found) {
      addEvidence(evidence, 'bin_calculation', 0.25,
        `Bin index calculation: ${binCalcInfo.method}`);
    }

    // 3. Shared memory privatization (local histogram per block)
    const hasPrivatization = this.detectPrivatization(kernel, source);
    if (hasPrivatization) {
      addEvidence(evidence, 'shared_privatization', 0.20,
        'Shared memory privatization for local histogram');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Atomic operation target is indexed array (not scalar)
    const atomics = kernel.syncPoints.filter(s => s.type === 'atomic');
    const atomicToArray = this.hasAtomicToIndexedArray(source);
    if (atomicToArray) {
      addEvidence(evidence, 'atomic_to_array', 0.15,
        'Atomic operation targets indexed array position');
    }

    // 5. Input data range clamping/bounds checking
    const hasBoundsCheck = this.detectBoundsCheck(source);
    if (hasBoundsCheck) {
      addEvidence(evidence, 'bounds_check', 0.10,
        'Bin bounds checking detected');
    }

    // 6. Final reduction from shared to global (for privatized)
    const hasBlockReduction = this.detectBlockReduction(kernel, source);
    if (hasBlockReduction && hasPrivatization) {
      addEvidence(evidence, 'block_reduction', 0.10,
        'Block-level histogram reduction');
    }

    // 7. Name hints
    if (/histogram|hist|bucket|bin|count|frequency/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests histogram operation');
    }

    // 8. Integer division or modulo (common for binning)
    const hasDivMod = /\s\/\s*\d+|\s%\s*\d+|>>/.test(source);
    if (hasDivMod && binCalcInfo.found) {
      addEvidence(evidence, 'div_mod_binning', 0.05,
        'Division/modulo for bin calculation');
    }

    // 9. Comparison chains (for range-based binning)
    const hasComparisonChain = this.detectComparisonChain(source);
    if (hasComparisonChain) {
      addEvidence(evidence, 'comparison_binning', 0.10,
        'Comparison-based binning detected');
    }

    // === Negative Indicators ===

    // Stride halving suggests reduction
    if (kernel.loops.some(l => l.hasStrideHalving)) {
      addEvidence(evidence, 'stride_halving', -0.20,
        'Stride halving pattern (likely reduction)');
    }

    // Matrix multiply pattern suggests GEMM
    const hasMatMul = /\w+\s*\+=\s*\w+\[[^\]]+\]\s*\*\s*\w+\[[^\]]+\]/.test(source);
    if (hasMatMul) {
      addEvidence(evidence, 'matrix_multiply', -0.25,
        'Matrix multiply pattern (likely GEMM)');
    }

    // Neighbor offset suggests stencil
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length > 2) {
      addEvidence(evidence, 'neighbor_access', -0.15,
        'Neighbor access pattern (likely stencil)');
    }

    // No atomics at all strongly suggests not histogram
    if (atomics.length === 0 && !hasPrivatization) {
      addEvidence(evidence, 'no_atomic', -0.30,
        'No atomic operations or privatization');
    }

    const match = createPatternMatch('histogram', evidence, warnings);

    // Determine variant
    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(hasPrivatization, atomicIncrementInfo);
    }

    return match;
  }

  /**
   * Detect atomic increment patterns
   */
  private detectAtomicIncrement(source: string): { found: boolean; type: string } {
    // atomicAdd(&hist[bin], 1) or atomicInc(&hist[bin], max)
    if (/atomicAdd\s*\(\s*&?\s*\w+\s*\[[^\]]+\]\s*,\s*1\s*\)/.test(source)) {
      return { found: true, type: 'atomicAdd with 1' };
    }
    if (/atomicInc\s*\(\s*&?\s*\w+\s*\[[^\]]+\]/.test(source)) {
      return { found: true, type: 'atomicInc' };
    }
    // Generic atomicAdd to array
    if (/atomicAdd\s*\(\s*&?\s*\w+\s*\[[^\]]+\]/.test(source)) {
      return { found: true, type: 'atomicAdd to indexed array' };
    }
    return { found: false, type: '' };
  }

  /**
   * Detect bin/bucket calculation patterns
   */
  private detectBinCalculation(source: string): { found: boolean; method: string } {
    // Integer division: val / binWidth
    if (/\w+\s*\/\s*(?:bin_?width|bucket_?size|num_?bins|\d+)/i.test(source)) {
      return { found: true, method: 'integer division' };
    }
    // Right shift: val >> shift
    if (/\w+\s*>>\s*\d+/.test(source) && !/stride/.test(source)) {
      return { found: true, method: 'bit shift' };
    }
    // Multiply and truncate: (val * numBins) / range
    if (/\w+\s*\*\s*\w+\s*\//.test(source)) {
      return { found: true, method: 'scale and truncate' };
    }
    // Floor/cast: (int)(val * scale)
    if (/\(\s*int\s*\)\s*\(?\s*\w+\s*\*/.test(source)) {
      return { found: true, method: 'cast to int' };
    }
    // Direct modulo: val % numBins
    if (/\w+\s*%\s*(?:num_?bins|n_?bins|\w*bin|\d+)/i.test(source)) {
      return { found: true, method: 'modulo' };
    }
    // Variable named bin/bucket being assigned
    if (/(?:bin|bucket)\s*=\s*[^;]+/.test(source)) {
      return { found: true, method: 'explicit bin calculation' };
    }
    return { found: false, method: '' };
  }

  /**
   * Detect shared memory privatization for histogram
   */
  private detectPrivatization(kernel: CudaKernelInfo, source: string): boolean {
    // Must have shared memory
    if (kernel.sharedMemoryDecls.length === 0) return false;

    // Shared memory array with incrementing
    const hasSharedIncrement =
      /\w+_s\s*\[[^\]]+\]\s*\+\+/.test(source) ||
      /\w+_shared\s*\[[^\]]+\]\s*\+\+/.test(source) ||
      /s_?\w*\s*\[[^\]]+\]\s*\+=\s*1/.test(source);

    // Or atomicAdd to shared memory (rare but valid)
    const hasSharedAtomic = /atomicAdd\s*\(\s*&?\s*s_?\w*\s*\[/.test(source);

    // Or initialization of shared histogram to zero
    const hasSharedInit = /__shared__[^;]+\[\s*\d+\s*\][^;]*=\s*\{?\s*0/.test(source) ||
      /s_?\w*\s*\[\s*threadIdx/.test(source);

    return hasSharedIncrement || hasSharedAtomic || hasSharedInit;
  }

  /**
   * Check if atomic operation targets an indexed array
   */
  private hasAtomicToIndexedArray(source: string): boolean {
    // Pattern: atomicXxx(&array[index], ...)
    return /atomic\w+\s*\(\s*&?\s*\w+\s*\[[^\]]+\]/.test(source);
  }

  /**
   * Detect bounds checking for bins
   */
  private detectBoundsCheck(source: string): boolean {
    // if (bin < numBins) or if (bin >= 0 && bin < N)
    const patterns = [
      /if\s*\([^)]*(?:bin|bucket)\s*<\s*(?:num|n|max)/i,
      /if\s*\([^)]*(?:bin|bucket)\s*>=\s*0/i,
      /if\s*\([^)]*<\s*(?:num_?bins|n_?bins)/i,
      /(?:min|max|clamp)\s*\([^)]*(?:bin|bucket)/i,
    ];
    return patterns.some(p => p.test(source));
  }

  /**
   * Detect block reduction pattern (for privatized histogram)
   */
  private detectBlockReduction(kernel: CudaKernelInfo, source: string): boolean {
    // __syncthreads followed by atomic add from shared to global
    const hasSyncThenAtomic = /__syncthreads[\s\S]*atomicAdd/.test(source);

    // Or loop adding shared to global
    const hasReductionLoop = /for\s*\([^)]*\)[^}]*\w+\s*\[\s*\w+\s*\]\s*\+=\s*s_?\w*\s*\[/.test(source);

    return hasSyncThenAtomic || hasReductionLoop;
  }

  /**
   * Detect comparison chain for range-based binning
   */
  private detectComparisonChain(source: string): boolean {
    // Pattern: if (val < t1) bin = 0; else if (val < t2) bin = 1; ...
    const ifElseChain = /(if\s*\([^)]*<[^)]*\)[^}]*(?:bin|bucket)\s*=\s*\d+[^}]*){2,}/.test(source);

    // Or ternary chain: val < t1 ? 0 : val < t2 ? 1 : ...
    const ternaryChain = /\?\s*\d+\s*:\s*[^?]+\?\s*\d+\s*:/.test(source);

    return ifElseChain || ternaryChain;
  }

  /**
   * Determine the histogram variant
   */
  private determineVariant(
    hasPrivatization: boolean,
    atomicInfo: { found: boolean; type: string }
  ): PatternVariant {
    if (hasPrivatization) {
      return 'histogram_privatized';
    }
    if (atomicInfo.found) {
      return 'histogram_atomic';
    }
    return 'histogram_atomic'; // default
  }
}

export const histogramMatcher = new HistogramMatcher();
