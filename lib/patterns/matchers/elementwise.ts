// Elementwise Pattern Matcher
// Detects embarrassingly parallel kernels with 1:1 input-output mapping

import { CudaKernelInfo, PatternMatch, Evidence } from '../../ast/types';
import { PatternMatcher, createPatternMatch, addEvidence } from '../types';

export class ElementwiseMatcher implements PatternMatcher {
  match(kernel: CudaKernelInfo): PatternMatch {
    const evidence: Evidence[] = [];
    const warnings: string[] = [];

    // === Primary Indicators (presence indicates elementwise) ===

    // 1. Linear thread indexing (tid = threadIdx.x + blockIdx.x * blockDim.x)
    if (kernel.threadIndexUsage.globalIdExpression) {
      addEvidence(evidence, 'linear_thread_index', 0.25,
        'Simple global thread ID calculation detected');
    } else if (kernel.threadIndexUsage.usesThreadIdxX && kernel.threadIndexUsage.usesBlockIdxX) {
      addEvidence(evidence, 'thread_block_indexing', 0.15,
        'Uses threadIdx.x and blockIdx.x for indexing');
    }

    // 2. Direct array access with thread index (no offsets)
    const directAccesses = kernel.memoryAccesses.filter(a => !a.hasNeighborOffset);
    const totalAccesses = kernel.memoryAccesses.length;

    if (totalAccesses > 0) {
      const directRatio = directAccesses.length / totalAccesses;
      if (directRatio >= 0.8) {
        addEvidence(evidence, 'direct_array_access', 0.25,
          `${Math.round(directRatio * 100)}% direct arr[tid] access pattern`);
      } else if (directRatio >= 0.5) {
        addEvidence(evidence, 'mostly_direct_access', 0.15,
          `${Math.round(directRatio * 100)}% direct array access`);
      }
    }

    // 3. Simple arithmetic (no complex loop structures)
    if (kernel.loops.length === 0) {
      addEvidence(evidence, 'no_loops', 0.15,
        'No loop structures (pure element mapping)');
    } else if (kernel.loops.length === 1 && !kernel.loops[0].containsSyncthreads) {
      addEvidence(evidence, 'simple_loop', 0.10,
        'Single simple loop without synchronization');
    }

    // === Absence-Based Indicators ===

    // 4. No shared memory (typical for elementwise)
    if (kernel.sharedMemoryDecls.length === 0) {
      addEvidence(evidence, 'no_shared_memory', 0.10,
        'No shared memory usage');
    }

    // 5. No __syncthreads (no thread cooperation)
    const syncCount = kernel.syncPoints.filter(s => s.type === 'syncthreads').length;
    if (syncCount === 0) {
      addEvidence(evidence, 'no_synchronization', 0.10,
        'No __syncthreads calls');
    }

    // 6. No atomic operations
    const atomicCount = kernel.syncPoints.filter(s => s.type === 'atomic').length;
    if (atomicCount === 0) {
      addEvidence(evidence, 'no_atomics', 0.05,
        'No atomic operations');
    }

    // 7. No neighbor access patterns
    const neighborAccesses = kernel.memoryAccesses.filter(a => a.hasNeighborOffset);
    if (neighborAccesses.length === 0) {
      addEvidence(evidence, 'no_neighbor_access', 0.10,
        'No neighbor/offset access patterns');
    }

    // 8. Balanced reads/writes (1:1 mapping)
    const reads = kernel.memoryAccesses.filter(a => a.accessType === 'read').length;
    const writes = kernel.memoryAccesses.filter(a => a.accessType === 'write').length;
    if (reads > 0 && writes > 0 && Math.abs(reads - writes) <= 1) {
      addEvidence(evidence, 'balanced_io', 0.10,
        '1:1 input-output element mapping');
    }

    // === Negative Indicators (reduce confidence) ===

    // Nested loops suggest other patterns
    const nestedLoops = kernel.loops.filter(l => l.nestLevel > 0);
    if (nestedLoops.length > 0) {
      addEvidence(evidence, 'nested_loops', -0.20,
        'Nested loops detected (may not be elementwise)');
      warnings.push('Nested loops may indicate GEMM or other pattern');
    }

    // Stride patterns suggest reduction/scan
    const strideLoops = kernel.loops.filter(l => l.hasStrideHalving || l.hasStrideDoubling);
    if (strideLoops.length > 0) {
      addEvidence(evidence, 'stride_pattern', -0.25,
        'Stride pattern detected (likely reduction/scan)');
    }

    // 2D thread indexing suggests matrix operations
    if (kernel.threadIndexUsage.usesThreadIdxY && kernel.threadIndexUsage.usesBlockIdxY) {
      addEvidence(evidence, '2d_indexing', -0.15,
        '2D thread indexing (may be GEMM/stencil)');
    }

    // Neighbor access suggests stencil
    if (neighborAccesses.length > 2) {
      addEvidence(evidence, 'has_neighbor_access', -0.30,
        'Multiple neighbor accesses (likely stencil)');
    }

    // Shared memory suggests cooperative algorithms
    if (kernel.sharedMemoryDecls.length > 0) {
      addEvidence(evidence, 'has_shared_memory', -0.15,
        'Shared memory present (cooperative pattern)');
    }

    // Elementwise is the default/fallback pattern - ensure minimum confidence
    const result = createPatternMatch('elementwise', evidence, warnings);
    result.confidence = Math.max(0.40, result.confidence); // Minimum baseline

    return result;
  }
}

export const elementwiseMatcher = new ElementwiseMatcher();
