/**
 * Sorting Pattern Matcher
 * Detects bitonic sort, radix sort, and merge sort patterns
 */

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class SortingMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // === Primary Indicators (high weight) ===

    // 1. Bitonic sort: compare-and-swap with specific index pattern
    const bitonicInfo = this.detectBitonicPattern(source);
    if (bitonicInfo.found) {
      addEvidence(evidence, 'bitonic_pattern', 0.35,
        `Bitonic sort pattern: ${bitonicInfo.type}`);
    }

    // 2. Radix sort: bit extraction and histogram
    const radixInfo = this.detectRadixPattern(source, kernel);
    if (radixInfo.found) {
      addEvidence(evidence, 'radix_pattern', 0.35,
        `Radix sort pattern: ${radixInfo.type}`);
    }

    // 3. Merge sort: merge operation with two sorted sequences
    const mergeInfo = this.detectMergePattern(source);
    if (mergeInfo.found) {
      addEvidence(evidence, 'merge_pattern', 0.30,
        'Merge sort pattern detected');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Compare-and-swap operation
    const hasCompareSwap = this.detectCompareSwap(source);
    if (hasCompareSwap) {
      addEvidence(evidence, 'compare_swap', 0.15,
        'Compare-and-swap operation detected');
    }

    // 5. Power-of-2 stride patterns (bitonic characteristic)
    const hasPow2Stride = /stride\s*[=*\/]\s*2|stride\s*>>=\s*1|stride\s*<<=\s*1/.test(source);
    if (hasPow2Stride && bitonicInfo.found) {
      addEvidence(evidence, 'pow2_stride', 0.10,
        'Power-of-2 stride pattern');
    }

    // 6. XOR indexing (bitonic network)
    const hasXorIndex = /\^\s*\d+|\^\s*stride|\^\s*step/.test(source);
    if (hasXorIndex) {
      addEvidence(evidence, 'xor_indexing', 0.15,
        'XOR-based indexing (bitonic network)');
    }

    // 7. Bit manipulation for key extraction
    const hasBitExtract = />>|&\s*0x|&\s*\(/.test(source) && /sort|key|radix/i.test(kernel.name);
    if (hasBitExtract) {
      addEvidence(evidence, 'bit_extract', 0.10,
        'Bit manipulation for key extraction');
    }

    // 8. Warp-level sorting primitives
    const hasWarpSort = /__ballot|__shfl|warp.*sort/i.test(source);
    if (hasWarpSort) {
      addEvidence(evidence, 'warp_sort', 0.15,
        'Warp-level sorting primitive');
    }

    // 9. Name hints
    if (/sort|bitonic|radix|merge|quick|heap/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests sorting');
    }

    // 10. Shared memory for local sorting
    if (kernel.sharedMemoryDecls.length > 0 && hasCompareSwap) {
      addEvidence(evidence, 'shared_sort', 0.10,
        'Shared memory for local sorting');
    }

    // === Negative Indicators ===

    // Reduction patterns
    if (kernel.loops.some(l => l.hasStrideHalving) && !bitonicInfo.found) {
      addEvidence(evidence, 'reduction_pattern', -0.20,
        'Stride halving without bitonic (likely reduction)');
    }

    // No comparison operations
    if (!/[<>]=?/.test(source) && !hasCompareSwap) {
      addEvidence(evidence, 'no_comparison', -0.25,
        'No comparison operations found');
    }

    const match = createPatternMatch('sorting' as any, evidence, warnings);

    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(bitonicInfo, radixInfo, mergeInfo);
    }

    return match;
  }

  /**
   * Detect bitonic sort pattern
   */
  private detectBitonicPattern(source: string): { found: boolean; type: string } {
    // Classic bitonic: compare elements at distance stride with alternating direction
    if (/\[\s*\w+\s*\^\s*\w+\s*\]/.test(source) && /stride|step|offset/.test(source)) {
      return { found: true, type: 'bitonic XOR indexing' };
    }

    // Compare-swap with direction flag
    if (/dir\s*\^|ascending|descending/i.test(source) && /swap|exchange/.test(source)) {
      return { found: true, type: 'directional compare-swap' };
    }

    // Bitonic merge pattern
    if (/for\s*\([^)]*step[^)]*\)\s*{[^}]*for\s*\([^)]*stride/.test(source)) {
      return { found: true, type: 'nested bitonic loops' };
    }

    // Log2(n) iterations with power-of-2 stride
    if (/for\s*\([^)]*k\s*=\s*2[^)]*k\s*\*=?\s*2/.test(source)) {
      return { found: true, type: 'log2(n) stride doubling' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect radix sort pattern
   */
  private detectRadixPattern(source: string, kernel: CudaKernelInfo): { found: boolean; type: string } {
    // Bit extraction with histogram
    const hasBitExtract = /\(\s*\w+\s*>>\s*\d+\s*\)\s*&\s*\d+/.test(source);
    const hasHistogram = /atomicAdd|histogram|count/i.test(source);

    if (hasBitExtract && hasHistogram) {
      return { found: true, type: 'histogram-based radix' };
    }

    // Prefix sum for scatter
    if (hasBitExtract && /prefix|scan|exclusive_sum/i.test(source)) {
      return { found: true, type: 'prefix-sum scatter' };
    }

    // Split operation (separate 0s and 1s)
    if (/\&\s*1|%\s*2/.test(source) && /split|partition/i.test(source)) {
      return { found: true, type: 'split-based radix' };
    }

    // Digit extraction loop
    if (/for\s*\([^)]*bit|for\s*\([^)]*digit|for\s*\([^)]*pass/i.test(source) && hasBitExtract) {
      return { found: true, type: 'multi-pass radix' };
    }

    return { found: false, type: '' };
  }

  /**
   * Detect merge sort pattern
   */
  private detectMergePattern(source: string): { found: boolean } {
    // Two-way merge with sorted input
    const hasMerge = /merge|left\s*<|right\s*<|mid/i.test(source);
    const hasTwoPointers = /left_idx|right_idx|i\s*<\s*mid|j\s*<\s*end/i.test(source);

    if (hasMerge && hasTwoPointers) {
      return { found: true };
    }

    // Bottom-up merge (iterative)
    if (/for\s*\([^)]*width[^)]*\)\s*{[^}]*merge/i.test(source)) {
      return { found: true };
    }

    return { found: false };
  }

  /**
   * Detect compare-and-swap
   */
  private detectCompareSwap(source: string): boolean {
    // Direct compare-swap pattern
    if (/if\s*\([^)]*[<>][^)]*\)\s*{[^}]*swap|min.*max|max.*min/i.test(source)) {
      return true;
    }

    // Conditional exchange
    if (/temp\s*=.*\[.*\].*\[.*\]\s*=.*\[.*\]\s*=\s*temp/.test(source)) {
      return true;
    }

    // Min/max swap
    if (/\w+\s*=\s*min\s*\([^)]+\)[^;]*\w+\s*=\s*max\s*\([^)]+\)/.test(source)) {
      return true;
    }

    return false;
  }

  /**
   * Determine sorting variant
   */
  private determineVariant(
    bitonicInfo: { found: boolean; type: string },
    radixInfo: { found: boolean; type: string },
    mergeInfo: { found: boolean }
  ): PatternVariant {
    if (bitonicInfo.found) {
      return 'bitonic_sort' as PatternVariant;
    }
    if (radixInfo.found) {
      return 'radix_sort' as PatternVariant;
    }
    if (mergeInfo.found) {
      return 'merge_sort' as PatternVariant;
    }
    return 'bitonic_sort' as PatternVariant; // default
  }
}

export const sortingMatcher = new SortingMatcher();
