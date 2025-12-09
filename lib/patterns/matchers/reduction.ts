// Reduction Pattern Matcher
// Detects tree reduction, parallel sum, and similar patterns
// Supports variants: tree_reduction, warp_shuffle, multi_block, segmented

import { CudaKernelInfo, PatternMatch, Evidence, PatternVariant } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class ReductionMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];
    const source = kernel.sourceText;

    // Track variant indicators
    const variantSignals = {
      hasWarpShuffle: false,
      hasMultiBlock: false,
      hasSegmented: false,
      hasTreeReduction: false,
    };

    // === Primary Indicators (high weight) ===

    // 1. Stride halving pattern (stride >>= 1 or stride /= 2) - Critical!
    const strideHalvingLoops = kernel.loops.filter(l => l.hasStrideHalving);
    if (strideHalvingLoops.length > 0) {
      addEvidence(evidence, 'stride_halving', 0.35,
        `Stride halving detected in ${strideHalvingLoops.length} loop(s)`);
      variantSignals.hasTreeReduction = true;
    }

    // 2. __syncthreads inside loop (essential for correctness)
    const syncInLoop = kernel.loops.some(l => l.containsSyncthreads);
    if (syncInLoop) {
      addEvidence(evidence, 'sync_in_reduction_loop', 0.20,
        '__syncthreads inside reduction loop');
    }

    // 3. Reduction accumulation pattern: arr[tid] += arr[tid + stride]
    const hasReductionAccum = this.detectReductionAccumulation(kernel);
    if (hasReductionAccum) {
      addEvidence(evidence, 'reduction_accumulation', 0.15,
        'arr[tid] += arr[tid + stride] pattern detected');
    }

    // === Secondary Indicators (medium weight) ===

    // 4. Shared memory for partial sums
    if (kernel.sharedMemoryDecls.length > 0) {
      addEvidence(evidence, 'shared_memory_accumulator', 0.15,
        'Shared memory for partial sums');
    }

    // 5. Atomic operation at end (multi-block reduction)
    const atomics = kernel.syncPoints.filter(s => s.type === 'atomic');
    if (atomics.length > 0) {
      const hasAtomicAdd = atomics.some(a => a.name.includes('atomicAdd'));
      if (hasAtomicAdd) {
        addEvidence(evidence, 'atomic_final_accumulation', 0.15,
          'atomicAdd for inter-block reduction');
        variantSignals.hasMultiBlock = true;
      } else {
        addEvidence(evidence, 'atomic_operation', 0.10,
          'Atomic operation detected');
      }
    }

    // 6. Thread ID comparison (tid < stride pattern)
    const hasTidStrideComparison = this.detectTidStrideComparison(source);
    if (hasTidStrideComparison) {
      addEvidence(evidence, 'tid_stride_guard', 0.10,
        'Thread divergence guard (tid < stride)');
    }

    // 7. Many reads, few writes (reduction semantics)
    const reads = kernel.memoryAccesses.filter(a => a.accessType === 'read').length;
    const writes = kernel.memoryAccesses.filter(a => a.accessType === 'write').length;
    if (reads > writes * 3 && writes > 0) {
      addEvidence(evidence, 'many_to_one', 0.10,
        `Many-to-one pattern: ${reads} reads, ${writes} writes`);
    }

    // 8. Warp-level reduction (__shfl_down, __shfl_xor)
    const hasWarpShuffle = this.detectWarpShuffle(source);
    if (hasWarpShuffle) {
      addEvidence(evidence, 'warp_shuffle', 0.15,
        'Warp shuffle instructions detected (warp-level reduction)');
      variantSignals.hasWarpShuffle = true;
    }

    // 8b. Segmented reduction detection
    const hasSegmented = this.detectSegmentedReduction(source);
    if (hasSegmented) {
      addEvidence(evidence, 'segmented_reduction', 0.20,
        'Segmented/per-row reduction pattern detected');
      variantSignals.hasSegmented = true;
    }

    // 9. Name hints
    if (/reduce|reduction|sum|max|min|average|mean/i.test(kernel.name)) {
      addEvidence(evidence, 'name_hint', 0.10,
        'Kernel name suggests reduction operation');
    }

    // 10. Decreasing loop bound (convergence)
    const hasDecreasingBound = kernel.loops.some(l =>
      l.condition.includes('>') && !l.condition.includes('>=')
    );
    if (hasDecreasingBound && strideHalvingLoops.length > 0) {
      addEvidence(evidence, 'convergent_loop', 0.05,
        'Loop converges toward completion');
    }

    // === Negative Indicators ===

    // Matrix multiply patterns suggest GEMM
    const hasMatrixMultiply = /\w+\s*\+=\s*\w+\[[^\]]+\]\s*\*\s*\w+\[[^\]]+\]/.test(source) &&
                             kernel.loops.length >= 2;
    if (hasMatrixMultiply) {
      addEvidence(evidence, 'matrix_multiply', -0.20,
        'Matrix multiply pattern (likely GEMM)');
    }

    // Neighbor access suggests stencil
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length > 3) {
      addEvidence(evidence, 'neighbor_access', -0.15,
        'Multiple neighbor accesses (likely stencil)');
    }

    // Both stride doubling AND halving suggests scan
    const hasStrideDoubling = kernel.loops.some(l => l.hasStrideDoubling);
    if (strideHalvingLoops.length > 0 && hasStrideDoubling) {
      addEvidence(evidence, 'dual_stride', -0.15,
        'Both stride patterns (may be scan)');
      warnings.push('Has both doubling and halving - could be scan');
    }

    const match = createPatternMatch('reduction', evidence, warnings);

    // Determine variant
    if (match.confidence > 0.3) {
      match.variant = this.determineVariant(variantSignals);
    }

    return match;
  }

  /**
   * Determine the reduction variant based on detected signals
   */
  private determineVariant(signals: {
    hasWarpShuffle: boolean;
    hasMultiBlock: boolean;
    hasSegmented: boolean;
    hasTreeReduction: boolean;
  }): PatternVariant {
    // Priority: segmented > warp_shuffle > multi_block > tree
    if (signals.hasSegmented) {
      return 'segmented';
    }
    if (signals.hasWarpShuffle) {
      return 'warp_shuffle';
    }
    if (signals.hasMultiBlock) {
      return 'multi_block';
    }
    return 'tree_reduction';
  }

  /**
   * Detect segmented reduction (per-row or per-segment)
   */
  private detectSegmentedReduction(source: string): boolean {
    // Per-row reduction pattern: loop over rows, inner reduction
    const hasRowLoop = /for\s*\([^)]*row[^)]*\)/i.test(source);
    const hasInnerReduction = /\+=\s*\w+\s*\[/.test(source);

    // Or segment-based: segment_start, segment_end
    const hasSegmentBounds = /segment_?(?:start|end|size)/i.test(source);

    // Or row-wise output: output[row] = ...
    const hasRowOutput = /\w+\s*\[\s*row\s*\]\s*=/.test(source);

    return (hasRowLoop && hasInnerReduction) || hasSegmentBounds || hasRowOutput;
  }

  private detectReductionAccumulation(kernel: CudaKernelInfo): boolean {
    const source = kernel.sourceText;

    // Pattern: sdata[tid] += sdata[tid + stride]
    const reductionPattern1 = /\w+\s*\[\s*\w+\s*\]\s*\+=\s*\w+\s*\[\s*\w+\s*\+\s*\w+\s*\]/;

    // Pattern: sdata[tid] = sdata[tid] + sdata[tid + stride]
    const reductionPattern2 = /\w+\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*\w+\s*\]\s*\+\s*\w+\s*\[\s*\w+\s*\+/;

    return reductionPattern1.test(source) || reductionPattern2.test(source);
  }

  private detectTidStrideComparison(source: string): boolean {
    // Pattern: if (tid < stride) or if (threadIdx.x < stride)
    const pattern1 = /if\s*\(\s*\w+\s*<\s*stride/i;
    const pattern2 = /if\s*\(\s*threadIdx\.x\s*<\s*\w+/;

    return pattern1.test(source) || pattern2.test(source);
  }

  private detectWarpShuffle(source: string): boolean {
    const shufflePatterns = [
      /__shfl_down/,
      /__shfl_xor/,
      /__shfl_sync/,
      /__shfl_up/,
      /__reduce_add_sync/,
      /__reduce_max_sync/,
      /__reduce_min_sync/,
    ];

    return shufflePatterns.some(p => p.test(source));
  }
}

export const reductionMatcher = new ReductionMatcher();
